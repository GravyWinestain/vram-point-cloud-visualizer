"""
ollama_driver.py — Streaming Ollama heartbeat driver for the spec port.

This is what connects the visualizer to the LLM *for real*, as the spec
intends. Where telemetry.py watches the system passively, this module
actively drives Ollama in a loop, streaming /api/generate responses and
classifying each frame of the lifecycle:

  - "prefill"   → first response after a request lands; the LLM is
                  processing the prompt. Map to Mode 1 (Context Prefill
                  Vortex).
  - "generating"→ tokens are streaming back; report live tokens/sec.
                  Map to Mode 2 (Harmonic Token Orbits).
  - "idle"      → between requests. Map to Mode 0 (Aizawa Idle).
  - "done"      → request finished cleanly; record final stats.

The driver exposes a thread-safe phase snapshot the render loop can read
without blocking. Cadence defaults to ~10 seconds between requests
(long enough to read prefill + generation as distinct phases).

Thread model: one daemon thread runs the request loop. Inside the loop
we use streaming HTTP with a short per-request timeout so the thread
exits cleanly on stop. Each NDJSON line read updates atomically-readable
fields and sets an Event on phase transitions.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Deque, Optional, Sequence, Tuple


# ─── Defaults ───

_DEFAULT_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen2.5-coder:7b"
# Rotating real prompts. Each cycle the model actually answers one of
# these, so the heartbeat is non-empty work — the response text drives
# the visual identity of each beat. Pass a custom list via __init__
# to override.
_DEFAULT_PROMPTS = (
    "Describe the colour blue in exactly three sentences.",
    "Write one sentence about a forest at night.",
    "Invent a name for a small star.",
    "What does silence sound like? Answer in one sentence.",
    "Tell me a single unusual fact about octopuses.",
    "Compose a haiku about rain.",
    "One sentence: what is consciousness?",
    "Describe the smell of rain on warm asphalt.",
    "Name a feeling you cannot translate to English.",
    "What is the loneliest number? Answer in one sentence.",
)
# How long to pause between requests. Long enough that prefill and
# generation show up as distinct visual phases, short enough that the
# driver feels like a heartbeat.
_DEFAULT_INTERVAL_S = 8.0
# Per-request timeout: covers slow cold loads of large models.
_REQUEST_TIMEOUT_S = 60.0
# Normalisation ceiling for eval tokens/sec.
_EVAL_TPS_MAX = 80.0
# Prefill is reported in tokens/sec too (prompt tokens / prompt eval
# duration). Heavy prompts can hit thousands of tok/s on a small model;
# 4000 is a reasonable visual cap.
_PREFILL_TPS_MAX = 4000.0


# ─── Helpers ───

def _norm(value: float, ceiling: float) -> float:
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, value / ceiling))


# ─── Driver ───

class OllamaDriver:
    """Streaming Ollama heartbeat.

    Lifecycle (called by the engine):
        driver = OllamaDriver()
        driver.start()         # spawns background thread
        ...
        phase, tps, prefill_tps, last_prompt, last_eval = driver.snapshot()
        ...
        driver.stop()          # signal thread to exit, join
    """

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        model: str = _DEFAULT_MODEL,
        prompts: Optional[Sequence[str]] = None,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ):
        self.url = url.rstrip("/")
        self.model = model
        # Rotating prompts: each cycle uses the next one so the model
        # does visibly different work each beat.
        self._prompts: Sequence[str] = (
            list(prompts) if prompts is not None else list(_DEFAULT_PROMPTS)
        )
        self._prompt_index = 0
        self.prompt: str = self._prompts[0]
        self.interval_s = interval_s

        # Phase state. Values are read on every render frame, so we
        # hold them under a lock to keep torn reads off the table.
        self._lock = threading.Lock()
        self._phase = "idle"          # "idle" | "prefill" | "generating" | "done" | "error"
        self._eval_tps_norm = 0.0     # live eval tokens/sec, normalised
        self._prefill_tps_norm = 0.0  # live prefill tokens/sec, normalised
        self._last_prompt_eval_count = 0
        self._last_prompt_eval_duration_ns = 0
        self._last_eval_count = 0
        self._last_eval_duration_ns = 0
        self._request_count = 0
        self._error_count = 0
        # Per-token events the engine reads each frame to decide
        # whether to fire a GPU shockwave. Each entry is (mono_time,
        # token_hash). Capped at 32 so we never hold more than a
        # couple of seconds of token history.
        self._recent_token_events: Deque[Tuple[float, int]] = deque(maxlen=32)
        # Hash of the response text so far — drives a colour tint so
        # each beat looks distinct.
        self._response_hue = 0.0
        # Latest full response text (trimmed). Useful for debugging
        # and for any future UI overlay.
        self._response_text = ""

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─── Public API ───

    def start(self) -> None:
        if self._thread is not None:
            return  # already started
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ollama-driver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> Tuple[
        str, float, float, int, int,
        Tuple[Tuple[float, int], ...], float, str
    ]:
        """Return (phase, eval_tps_norm, prefill_tps_norm,
        last_eval_count, last_prompt_eval_count,
        recent_token_events, response_hue, response_text).

        Cheap, lock-protected read. The render thread calls this every
        frame and never blocks on network I/O.

        recent_token_events is a snapshot of the last few (mono_time,
        token_hash) pairs, copied so the caller can iterate without
        holding the lock.
        """
        with self._lock:
            return (
                self._phase,
                self._eval_tps_norm,
                self._prefill_tps_norm,
                self._last_eval_count,
                self._last_prompt_eval_count,
                tuple(self._recent_token_events),
                self._response_hue,
                self._response_text,
            )

    # ─── Internal ───

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            if self._phase != phase:
                self._phase = phase

    def _record_prefill(self, count: int, duration_ns: int) -> None:
        with self._lock:
            self._last_prompt_eval_count = count
            self._last_prompt_eval_duration_ns = duration_ns
            if duration_ns > 0 and count > 0:
                prefill_tps = count / (duration_ns / 1e9)
                self._prefill_tps_norm = _norm(prefill_tps, _PREFILL_TPS_MAX)

    def _record_eval(self, count: int, duration_ns: int) -> None:
        with self._lock:
            self._last_eval_count = count
            self._last_eval_duration_ns = duration_ns
            if duration_ns > 0 and count > 0:
                eval_tps = count / (duration_ns / 1e9)
                self._eval_tps_norm = _norm(eval_tps, _EVAL_TPS_MAX)

    def _bump_request(self) -> None:
        with self._lock:
            self._request_count += 1
            # Reset per-request state so the previous response's hue
            # doesn't linger across beats.
            self._response_text = ""
            self._response_hue = 0.0
            self._recent_token_events.clear()

    def _record_token(self, token_text: str) -> None:
        """Push a (mono_time, hash) event for one streamed token.

        Also recomputes the running response hue from the cumulative
        text so the GPU can tint the current generation distinctly
        from the previous one.
        """
        with self._lock:
            self._recent_token_events.append(
                (time.monotonic(), hash(token_text))
            )
            self._response_text += token_text
            # Cheap deterministic hue: hash the *full* running
            # response text. Different prompts → different content →
            # different hue. Rolling the hash gives a smooth
            # transition across tokens in the same response.
            self._response_hue = (hash(self._response_text) & 0xFFFF) / 65535.0

    def _bump_error(self) -> None:
        with self._lock:
            self._error_count += 1
            self._phase = "error"

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    @property
    def error_count(self) -> int:
        with self._lock:
            return self._error_count

    # ─── Streaming call ───

    def _stream_once(self) -> None:
        """Run a single streaming /api/generate and update state.

        We treat the first line of the response as the prefill phase
        and every subsequent `response` line as generating phase. The
        final line has `done:true` and contains the authoritative
        eval_count/eval_duration/prompt_eval_count/prompt_eval_duration.
        """
        # Rotate to the next prompt for this beat.
        with self._lock:
            self.prompt = self._prompts[self._prompt_index]
            self._prompt_index = (self._prompt_index + 1) % len(self._prompts)
        body = json.dumps({
            "model": self.model,
            "prompt": self.prompt,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as exc:
            # Don't crash the thread; just count and fall back to idle.
            self._bump_error()
            print(f"[ollama] request failed: {exc}")
            return

        self._bump_request()
        self._set_phase("prefill")
        first_line = True
        try:
            # Read one line at a time so we can react mid-stream.
            for raw in resp:
                if self._stop.is_set():
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # First non-empty line arrives before eval tokens start.
                if first_line:
                    first_line = False
                    # Some Ollama versions emit prompt_eval stats on the
                    # very first line; capture if present.
                    pe_count = data.get("prompt_eval_count", 0)
                    pe_dur = data.get("prompt_eval_duration", 0)
                    if pe_count and pe_dur:
                        self._record_prefill(pe_count, pe_dur)
                    # Switch to generating on any non-done line that
                    # carries a `response` payload.
                    if data.get("response") and not data.get("done"):
                        self._set_phase("generating")
                    continue
                # Subsequent lines.
                if data.get("done"):
                    # Final authoritative stats.
                    pe_count = data.get("prompt_eval_count", 0)
                    pe_dur = data.get("prompt_eval_duration", 0)
                    if pe_count and pe_dur:
                        self._record_prefill(pe_count, pe_dur)
                    e_count = data.get("eval_count", 0)
                    e_dur = data.get("eval_duration", 0)
                    if e_count and e_dur:
                        self._record_eval(e_count, e_dur)
                    self._set_phase("done")
                    break
                if data.get("response"):
                    self._set_phase("generating")
                    # Record this token for the per-frame shockwave.
                    self._record_token(data["response"])
                    # Some streams include eval_running totals mid-stream;
                    # capture if present so tps is fresh by the time
                    # done:true arrives.
                    e_count = data.get("eval_count", 0)
                    e_dur = data.get("eval_duration", 0)
                    if e_count and e_dur:
                        self._record_eval(e_count, e_dur)
        finally:
            try:
                resp.close()
            except Exception:
                pass

    # ─── Main loop ───

    def _loop(self) -> None:
        # Brief warmup so the GL window appears before we hammer Ollama.
        if self._stop.wait(1.0):
            return
        while not self._stop.is_set():
            self._stream_once()
            # Hold the "done" phase for a moment so the render loop can
            # observe it and map it to Mode 2 (Harmonic Orbits — the
            # eval stats we just recorded are still relevant). Then
            # drop back to idle so the engine can map this to Mode 0
            # (Aizawa) between beats.
            self._set_phase("done")
            if self._stop.wait(1.0):
                return
            self._set_phase("idle")
            if self._stop.wait(self.interval_s):
                return