# Project Goal: Migrate Simulation Logic to GPU Compute Shader using PointCloud Specification

## 🎯 Overview
The objective is to convert the core physics and color update logic residing in the existing Python code (`main.rs`) into native GLSL compute shader programming constructs, as mandated by the technical specification provided. The entire visualization must run on the GPU via a single **Compute Shader** dispatched once per frame, ensuring optimal state management using **Shader Storage Buffer Objects (SSBOs)**.

The final deliverable will be an updated `COMPUTE_SHADER_SRC` string containing all logic inside the `main()` function body. The Python wrapper code will only be responsible for setting uniforms and dispatching the compute job.

## 🧪 Current State / Reference Code
The simulation structure is derived from:
- **Source File:** `/home/sfarrant/dev/RustroverProjects/utils/src/main.rs`
- **Specification Guide:** The technical document outlining mandatory GLSL structures, SSBO layouts, and required Mode behaviors (Aizawa, Context Prefill, etc.).

## 🛠️ Task Breakdown (The Refactoring Mandate)
Follow the following steps sequentially:

### Phase 1: Math Isolation & Translation (Most Critical)
1.  **Isolate Core Physics:** For every `update()` method in the Python code (`VortexPattern`, `CylinderPattern`, etc.), extract the pure mathematical calculations involving:
    *   Position updates ($\text{pos} += \text{force} * \Delta t$).
    *   Force/Attractor calculations (e.g., $\text{force} = \text{target} - (\text{velocity} * C)$).
2.  **GLSL Grammar Conversion:** Translate these equations into valid, clean GLSL syntax suitable for insertion between `lines 133` and `lines 171` of the compute shader's `main()` function. Pay extreme attention to floating-point precision (`float`, `vec3`) and mathematical functions (using standard GLSL built-ins like `sin()`, `cos()`, `normalize()`).

### Phase 2: Data Structure Enforcement
1.  **SSBO Binding:** Ensure that the Python wrapper correctly sets up the SSBO using the exact $48$-byte layout dictated by the specification's **Memory Usage Table** ($\text{vec3} + \text{float} + \text{vec3} + \text{float}$, etc.).
2.  **Uniform Management:** The `main()` loop must correctly pass all time-dependent parameters (`u_time`, `u_delta_time`, `u_gpu_usage`, etc.) as uniforms, ensuring the shader can read them for calculations.

### Phase 3: Procedural Logic Integration
1.  **Update Shader Modes:** Rewrite the entire block inside the compute shader's `main()` function to use `if/else if` structures based on `u_pattern_mode`. Each block must encapsulate the logic from one of the five patterns (Aizawa, Vortex, etc.).
2.  **Color Calculation Shift:** The color blending logic ($\text{Option<(f32, f32, f32)>}$ blocks) must be translated to GLSL mix/lerp functions using uniforms derived from telemetry like `u_gpu_temp` and `u\_tokens\_per\_sec`.

### Phase 4: Error Handling & Finalization
1.  **Self-Correction:** The final shader code MUST NOT rely on Python-specific helper methods (`fastrand::f32()`, etc.). All randomization, if absolutely necessary, must be replaced with deterministically seeded GLSL math or documented as a required external step (like initializing base positions).

**In summary: Treat this task as writing the definitive C++ / GLSL version of the simulation described in Python.**