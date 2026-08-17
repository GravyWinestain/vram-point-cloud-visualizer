# compute.glsl + point_cloud.py — Standalone GLSL spec port

This directory contains two things side-by-side:

1. **`src/main.rs`** — the live, working Rust/eframe binary
   `cuda_monitor`. 12 patterns, CPU physics, 12 000 particles, runs
   in an eframe window. This is the app you've been iterating on.

2. **`compute.glsl` + `point_cloud.py`** — a separate, spec-faithful
   port of `point_cloud_spec.docx` to Python + PyOpenGL + GLFW.
   131 072 particles, GPU compute shader, 4 modes (Aizawa, Context
   Prefill, Token Orbits, Thermal Chaos). The Rust binary does not
   use or know about this code.

## Why two things in one repo

The Rust binary and the spec describe two different architectures:

| | Rust binary (`main.rs`) | Spec port (`point_cloud.py`) |
|---|---|---|
| Language | Rust | Python 3 |
| Windowing | eframe (wgpu on Linux) | GLFW + PyOpenGL |
| Physics | CPU, per-frame loop | GPU, compute shader |
| Particles | 12 000 | 131 072 |
| Patterns | 12 (manually tuned) | 4 (spec-defined) |

The Rust binary is the practical visual tool. The spec port is the
reference implementation that matches the `point_cloud_spec.docx`
architecture 1:1. Keeping them in one repo lets the spec evolve
without forking.

## Running the spec port

```bash
# One-time, in the repo root:
python3 -m venv .venv
.venv/bin/pip install glfw PyOpenGL numpy

# Run (the harness expects to find compute.glsl in cwd):
.venv/bin/python3 -u point_cloud.py
```

Then:
- `Space` cycles to the next pattern (0 → 1 → 2 → 3 → 0)
- `1..4` jumps directly to a pattern
- `Esc` quits

The harness logs one line per 60 frames so you can confirm it's
running. Telemetry values are sine-wave stand-ins — wire them to
real `nvidia-smi` and Ollama probes when you want the visuals to
respond to actual GPU/LLM load.

## Layout

```
compute.glsl     # GLSL 4.30 compute shader (read at runtime)
point_cloud.py   # Python harness: SSBO, dispatch, render
```
