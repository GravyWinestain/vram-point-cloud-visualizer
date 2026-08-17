"""
telemetry.py — Live GPU + LLM telemetry for the spec port.

Async design: each source has its own daemon thread that polls on a
fixed cadence and atomically updates a last-known-good value. The
render thread just calls `snapshot()` and never blocks on a subprocess
or HTTP call. This is the fix for the "jerky" report — the previous
sync version was stalling the render loop on every poll.
"""

import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Optional


# ─── Probes (each runs in its own thread) ───

def _run_nvidia_smi() -> Optional[tuple[float, float, float, float]]:
    """Returns (util, vram_used, vram_total, temp_c) or None on failure."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        out = subprocess.check_output(
            [
                binary,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        )
        line = out.decode().strip().splitlines()[0]
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 4:
            return None
        return (
            float(parts[0]),   # util %
            float(parts[1]),   # vram used MB
            float(parts[2]),   # vram total MB
            float(parts[3]),   # temp °C
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ValueError, IndexError, OSError):
        return None


_PROBE_PROMPT = "hi"
_PROBE_MODEL = "qwen2.5-coder:7b"
_OLLAMA_URL = "http://127.0.0.1:11434"


def _run_ollama_generate() -> Optional[float]:
    """Returns tokens/sec from a one-shot /api/generate call, or None."""
    body = json.dumps({
        "model": _PROBE_MODEL,
        "prompt": _PROBE_PROMPT,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, TimeoutError, OSError):
        return None
    if not data.get("done"):
        return None
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    if eval_duration_ns <= 0 or eval_count <= 0:
        return None
    return eval_count / (eval_duration_ns / 1e9)


# ─── Normalisation ───

TEMP_MIN_C = 25.0
TEMP_SPAN_C = 60.0
TPS_MAX = 60.0


def _norm_util(util_pct: float) -> float:
    return max(0.0, min(1.0, util_pct / 100.0))


def _norm_vram(used_mb: float, total_mb: float) -> float:
    if total_mb <= 0:
        return 0.0
    return max(0.0, min(1.0, used_mb / total_mb))


def _norm_temp(temp_c: float) -> float:
    return max(0.0, min(1.0, (temp_c - TEMP_MIN_C) / TEMP_SPAN_C))


def _norm_tps(tps: float) -> float:
    return max(0.0, min(1.0, tps / TPS_MAX))


# ─── Public API ───

class TelemetrySource:
    """Background-pollable source of normalised telemetry.

    On `start()` two daemon threads begin polling at fixed cadences:
      - nvidia-smi at 4 Hz
      - Ollama at 1 Hz (heavier call, less frequent)
    Each thread updates its own atomically-readable value. The render
    thread never blocks — `snapshot()` just reads the cached values.
    """

    def __init__(self):
        self._util = 0.0
        self._temp = 0.3
        self._vram = 0.0
        self._tps = 0.0
        self._has_nvidia = False
        self._has_ollama = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t_nvidia: Optional[threading.Thread] = None
        self._t_ollama: Optional[threading.Thread] = None

    def _set_nvidia(self, util: float, temp: float, vram: float) -> None:
        with self._lock:
            self._util = util
            self._temp = temp
            self._vram = vram
            self._has_nvidia = True

    def _set_ollama(self, tps_norm: float) -> None:
        with self._lock:
            self._tps = tps_norm
            self._has_ollama = True

    def _nvidia_loop(self) -> None:
        # Prime once at start.
        smi = _run_nvidia_smi()
        if smi is not None:
            util_pct, vram_used, vram_total, temp_c = smi
            self._set_nvidia(_norm_util(util_pct), _norm_temp(temp_c),
                             _norm_vram(vram_used, vram_total))
        while not self._stop.is_set():
            self._stop.wait(0.25)  # 4 Hz
            if self._stop.is_set():
                return
            smi = _run_nvidia_smi()
            if smi is None:
                continue
            util_pct, vram_used, vram_total, temp_c = smi
            self._set_nvidia(_norm_util(util_pct), _norm_temp(temp_c),
                             _norm_vram(vram_used, vram_total))

    def _ollama_loop(self) -> None:
        # Prime once at start.
        tps = _run_ollama_generate()
        if tps is not None:
            self._set_ollama(_norm_tps(tps))
        while not self._stop.is_set():
            self._stop.wait(1.0)  # 1 Hz
            if self._stop.is_set():
                return
            tps = _run_ollama_generate()
            if tps is None:
                continue
            self._set_ollama(_norm_tps(tps))

    def start(self) -> None:
        """Spawn background polling threads."""
        if self._t_nvidia is not None:
            return  # already started
        self._t_nvidia = threading.Thread(
            target=self._nvidia_loop, name="telemetry-nvidia", daemon=True
        )
        self._t_ollama = threading.Thread(
            target=self._ollama_loop, name="telemetry-ollama", daemon=True
        )
        self._t_nvidia.start()
        self._t_ollama.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._t_nvidia, self._t_ollama):
            if t is not None:
                t.join(timeout=2.0)

    def snapshot(self) -> tuple[float, float, float, float]:
        """Return (gpu_usage, gpu_temp, vram_usage, tokens_per_sec), all 0..1.

        Lock-free read: the four fields are only ever assigned to, never
        mutated in place, so a torn read is harmless.
        """
        with self._lock:
            return (self._util, self._temp, self._vram, self._tps)
