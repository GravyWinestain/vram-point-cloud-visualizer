# GPU Tuning — Reversibility Checkpoint

Date: 2026-08-05
Host: Groucho | GPU: NVIDIA GeForce RTX 4060 Ti 16GB
Decision: **Left GPU at stock.** Persistence mode enabled (the one safe free win).
NO clock locks, NO undervolt, NO overclock applied.

## Why stock was the right call
- Throttle reason = **SW Power Cap** — card drops from 3105→2670 MHz because it
  must stay under the 165W BIOS hard-cap. Locking a HIGHER clock raises power,
  so it cannot recover headroom.
- Workload = **LLM inference** (Ollama llama-server), which is memory-bandwidth
  bound, not SM-clock bound. Clock tuning gains ~zero for token generation.
- Persistence mode was the genuine free gain and it is now ON.

## Current verified state (rollback reference)
- Driver: 595.84, CUDA 13.2
- Power limit: 165W / 165W (max — cannot raise, VBIOS locked)
- Max SM clock: 3105 MHz | Max mem clock: 9001 MHz
- Idle: 210 MHz, ~7W
- Persistence mode: **Enabled** (was OFF — this was the real fix)
- No clock lock set | No Applications Clocks override

## Revert commands if ever needed
   Lock GPU clock :  sudo nvidia-smi -lgc <MHz>     # e.g. 2670
   Reset clocks   :  sudo nvidia-smi -rgc
   Disable persis :  sudo nvidia-smi -pm 0

## LACT status
- Installed at /usr/bin/lact, config /etc/lact/config.yaml (version 5,
  admin_group: sudo).
- NOTE: LACT is primarily an AMD-GPU tool. On NVIDIA 4060 Ti it has no
  voltage/clock leverage — the card's VBIOS locks power at 165W. Do not rely
  on LACT for NVIDIA tuning.

## Known blocker (future GPU work)
- CUDA nvcc (13.1 at /usr/local/cuda-13.1, default /usr/local/cuda)
  CANNOT compile CUDA kernels on this host:
    gcc 15.2 / modern glibc declare rsqrt/rsqrtf as `noexcept` in
    mathcalls.h; CUDA 13.1 bundled crt math_functions.h lacks noexcept
    -> "error: exception specification is incompatible".
    Tried: -D__STRICT_ANSI__, -D__USE_FIXED_PROTOTYPES__, -std=c++11,
            -ccbin g++-14. All fail. Needs CUDA 13.2 install or header patch.
- gpuburn.cu (CUDA benchmark) kept in project dir for future use.

## Project state
- cuda_monitor Rust GUI (utils/): INCOMPLETE main.rs (no eframe::App impl).
  Working pre-visualizer binary at target/release/cuda_monitor.
  Rollback = keep the broken source + working binary untouched.
