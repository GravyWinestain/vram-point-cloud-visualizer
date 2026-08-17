// compute.glsl — Standalone GLSL 4.30 compute shader implementing the
// four-mode Interactive GPU/LLM Point Cloud Visualizer spec.
//
// Modes (set via u_pattern_mode):
//   0 = Aizawa Attractor     (Idle/Breathing, temp-driven)
//   1 = Context Prefill Vortex (Prompt Processing, util-driven)
//   2 = Harmonic Token Orbits (Active Generation, tokens/sec-driven)
//   3 = Thermal Chaos         (High Load, high temp)
//
// All physics is computed entirely on the GPU. The host (Python) only
// sets uniforms and dispatches glDispatchCompute(N/256, 1, 1).
//
// Particle SSBO layout (48 bytes per particle, std430):
//   offset  0  vec4  position (xyz = pos, w = phase/life)
//   offset 16  vec4  velocity (xyz = vel, w = mass/damping)
//   offset 32  vec4  color    (rgba)
//
// The host is expected to pre-fill the SSBO with initial positions
// (random sphere of radius 0.5..3.0), zero velocities, and a neutral
// color. This shader updates position, velocity, and color every frame.

#version 430 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// ─── SSBO ───
struct Particle {
    vec4 position; // xyz: position, w: life/phase
    vec4 velocity; // xyz: velocity, w: mass
    vec4 color;    // rgba: color
    vec4 anchor;   // xyz: pattern anchor (e.g. GridWave grid position),
                   // w: reserved (currently 0)
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

// ─── Uniforms ───
uniform float u_time;
uniform float u_delta_time;
uniform int   u_pattern_mode;

// Telemetry (all 0..1, normalised by the host)
uniform float u_gpu_usage;       // 0..1
uniform float u_gpu_temp;        // 0..1
uniform float u_tokens_per_sec;  // 0..1
uniform float u_vram_usage;      // 0..1

// Per-token reaction. xyz = unit-sphere origin of the most recent
// shockwave (token-hash-derived), w = intensity in [0, 1]. Decays to
// 0 within ~250ms when no new token lands; the host clamps it.
uniform vec4  u_token_burst;
// 0..1 hue rotation. The shader rotates the per-mode palette by this
// amount so each response looks visibly distinct.
uniform float u_response_hue;

// ─── Hash for stochastic helpers (no Python rand equivalent) ───
//
// Deterministic per-particle pseudo-noise used by the colour pulse and
// the simplex-ish turbulence in Mode 3. Cheaper than a real hash, and
// the spec doesn't require cryptographic randomness.
float hash11(float n) {
    return fract(sin(n) * 43758.5453123);
}

vec3 hash33(vec3 p) {
    // Cheap 3D noise used for the turbulence field in Thermal Chaos.
    // Per-axis scaled sine products — same trick the spec's reference
    // Python uses for noise3D.
    return vec3(
        sin(p.y * 2.5 + u_time) * cos(p.z * 2.5 + u_time),
        sin(p.z * 2.5 + u_time) * cos(p.x * 2.5 + u_time),
        sin(p.x * 2.5 + u_time) * cos(p.y * 2.5 + u_time)
    );
}

// ─── HSV rotation ───
//
// Rotate the hue of an RGB colour by `hueShift` turns. Used to tint
// every mode's base palette per-response so the user can *see* that
// the model has switched to a new prompt. Saturation and value are
// preserved.
vec3 rgb_hue_shift(vec3 rgb, float hueShift) {
    const mat3 toYIQ = mat3(0.299, 0.587, 0.114,
                            0.596,-0.274,-0.322,
                            0.211,-0.523, 0.312);
    const mat3 toRGB = mat3(1.0,  0.956,  0.621,
                            1.0, -0.272, -0.647,
                            1.0, -1.106,  1.703);
    vec3 yiq = toYIQ * rgb;
    float angle = hueShift * 6.2831853;
    float ca = cos(angle);
    float sa = sin(angle);
    yiq.yz = mat2(ca, -sa, sa, ca) * yiq.yz;
    return clamp(toRGB * yiq, 0.0, 1.0);
}

// ─── Aizawa attractor (Mode 0) ───
//
// Continuous strange attractor. The 6 constants a..f are the canonical
// Aizawa parameters that produce a visually rich 3D trajectory. Scaled
// to keep the field bounded near the origin so it reads as a "breathing
// blob" rather than a runaway orbit.
vec3 aizawaForce(vec3 p) {
    const float a = 0.95;
    const float b = 0.7;
    const float c = 0.6;
    const float d = 3.5;
    const float e = 0.25;
    const float f = 0.1;

    float dx = (p.z - b) * p.x - d * p.y;
    float dy = d * p.x + (p.z - b) * p.y;
    float dz = c + a * p.z - (p.z * p.z * p.z) / 3.0
             - (p.x * p.x + p.y * p.y) * (1.0 + e * p.z)
             + f * p.z * (p.x * p.x * p.x);
    return vec3(dx, dy, dz);
}

// ─── Main ───
//
// Per-particle physics + colour. Each mode branch produces a vec3 force
// (or zero, for the unsupported raw spec) and a vec3 baseColor. After
// integration we apply the spec's hard physics rules (semi-implicit
// Euler, hard wall at length 8 with elastic reflection) and set the
// colour with a per-particle alpha pulse.

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= particles.length()) return;

    Particle p = particles[idx];
    vec3 pos = p.position.xyz;
    vec3 vel = p.velocity.xyz;
    vec3 force = vec3(0.0);
    vec3 baseColor = vec3(0.0);

    // ─── Mode dispatch ───
    if (u_pattern_mode == 0) {
        // Aizawa — strange attractor, damped.
        vec3 target = aizawaForce(pos * 0.5) * 0.8;
        force = target - vel * 1.5;
        // Cool → warm gradient driven by temperature.
        baseColor = mix(vec3(0.1, 0.4, 0.8), vec3(0.0, 0.9, 0.7), u_gpu_temp);
    } else if (u_pattern_mode == 1) {
        // Context Prefill — inward radial pull + tangential swirl.
        // The pull strength scales with util so a busy GPU sucks
        // particles inward faster.
        vec3 pull = -normalize(pos) * (2.0 + u_gpu_usage * 3.0);
        vec3 swirl = vec3(-pos.y, pos.x, sin(u_time * 2.0)) * 4.0;
        force = pull + swirl - vel * 0.8;
        // Red → orange as util rises.
        baseColor = mix(vec3(0.9, 0.2, 0.1), vec3(1.0, 0.6, 0.0), u_gpu_usage);
    } else if (u_pattern_mode == 2) {
        // Harmonic Token Orbits — Lissajous grid, frequency tied to
        // tokens/sec, grid extent tied to VRAM.
        float freq = 1.0 + u_tokens_per_sec * 5.0;
        vec3 gridTarget = vec3(
            sin(float(idx) * 0.01 + u_time * freq),
            cos(float(idx) * 0.02 + u_time * freq * 0.5),
            sin(float(idx) * 0.03 + u_time * freq * 1.5)
        ) * (1.5 + u_vram_usage * 2.0);
        force = (gridTarget - pos) * 10.0 - vel * 2.0;
        // Green → cyan as tokens/sec rises.
        baseColor = mix(vec3(0.2, 0.8, 0.2), vec3(0.0, 1.0, 0.8), u_tokens_per_sec);
    } else if (u_pattern_mode == 3) {
        // Thermal Chaos — 3D noise turbulence + soft spherical pull.
        vec3 turbulence = hash33(pos * 1.5) * (1.0 + u_gpu_temp * 4.0);
        vec3 spherePull = (normalize(pos) * 2.5 - pos) * 2.0;
        force = turbulence + spherePull - vel * 0.5;
        // Magenta → red as temp rises.
        baseColor = mix(vec3(0.8, 0.0, 0.5), vec3(1.0, 0.1, 0.1), u_gpu_temp);
    } else if (u_pattern_mode == 4) {
        // GridWave — concentric radial ripple on a static square grid.
        // Replicates Rust pattern 11 (key `-`). Particles are pulled
        // toward their per-particle (anchor.x, anchor.y) on the XY
        // plane, with a z-displacement driven by sin(dist*6 - t*2*(1+a))
        // that produces expanding circular ripples. Amplitude and wave
        // speed scale with GPU utilisation, so an idle GPU shows a
        // flat grid and a busy GPU shows a churning sea of ripples.
        vec3 a = particles[idx].anchor.xyz;
        float dist = length(a.xy);
        float act = u_gpu_usage;
        float wave = sin(dist * 6.0 - u_time * 2.0 * (1.0 + act)) * act * 0.5;
        vec3 gridTarget = vec3(a.x, a.y, wave);
        // Spring with the same stiffness (0.06) and damping (0.89) as
        // the Rust version — same physical feel, just on the GPU.
        force = (gridTarget - pos) * 15.0 - vel * 4.5;
        // Cool palette with a warm tint at high utilisation. Tinted
        // toward blue-green to mirror the rust-mode's vibe.
        baseColor = mix(vec3(0.15, 0.5, 0.95), vec3(0.2, 0.9, 0.8), act);
    }

    // ─── Integrate (semi-implicit Euler, per spec) ───
    vel += force * u_delta_time;
    pos += vel * u_delta_time;

    // ─── Per-token shockwave ───
    // When the model streams a fresh token, the host injects a unit-
    // sphere origin and an intensity that decays within ~250ms. We
    // apply an outward radial force from that origin, scaled by
    // intensity, and brighten the colour so the eye sees a discrete
    // "pulse" on every token. Falls to zero when no new tokens land.
    if (u_token_burst.w > 0.0) {
        vec3 away = pos - u_token_burst.xyz;
        float d = length(away);
        if (d > 0.0001) {
            // Stronger push when close to the origin; falls off with
            // distance so distant particles get a softer nudge.
            float falloff = exp(-d * 0.25);
            vel += (away / d) * u_token_burst.w * falloff * 60.0 * u_delta_time;
        }
    }

    // ─── Hard physics rule: 8-unit sphere rebound ───
    // If a particle escapes the visualisation bubble, snap it back to
    // the surface and damp the velocity. This matches the spec
    // exactly (no respawn, no wrap — just elastic reflection).
    if (length(pos) > 8.0) {
        pos = normalize(pos) * 8.0;
        vel *= -0.5;
    }

    // ─── Per-particle alpha pulse ───
    // 0.6 base + 0.4 * sin(t + idx) so the field "breathes" on top
    // of the mode-driven base colour. Index-based phase keeps the
    // pattern from synchronising into a single blinking field.
    float alpha = 0.6 + 0.4 * sin(u_time + float(idx));

    // ─── Response-driven hue rotation ───
    // Each new response has its own hash-derived hue; rotating every
    // particle's base colour by that amount gives the whole field a
    // distinct palette per beat. Token-burst intensity also lifts the
    // alpha floor so a freshly-arrived token briefly floods the field
    // with bright colour.
    baseColor = rgb_hue_shift(baseColor, u_response_hue);
    alpha = min(1.0, alpha + u_token_burst.w * 0.3);

    // ─── Write back ───
    particles[idx].position = vec4(pos, p.position.w);
    particles[idx].velocity = vec4(vel, p.velocity.w);
    particles[idx].color    = vec4(baseColor, alpha);
}
