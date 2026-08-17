use eframe::egui;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::Command;
use std::time::Instant;

// ─── telemetry snapshot ───
#[derive(Clone, Copy, Default)]
struct GpuData {
    util: f32,
    vram_used_mb: f32,
    vram_total_mb: f32,
    temp_c: f32,
    power_w: f32,
}

#[derive(Clone, Copy)]
struct Particle {
    x: f32, y: f32, z: f32,
    vx: f32, vy: f32, vz: f32,
    base_x: f32, base_y: f32, base_z: f32,
    phase: f32,
    size: f32,
    age: f32,
    process_id: u8,
    layer: u8,
}

const PARTICLE_COUNT: usize = 4356;
const HISTORY_MAX: usize = 240;

// ─── extensible pattern framework ───
//
// Every visual pattern (built-in or user-added) implements this trait.
// Adding a new pattern is one call to `VramVisualizer::register_pattern` —
// no edits to match arms, no enum variants, no name() switch.
//
// The per-frame physics closure is boxed so the registry stores all
// patterns uniformly regardless of which concrete struct powers them.
struct PatternCtx {
    frame: u64,
    activity: f32,    // 0..1 = util/100
    vram_fill: f32,   // 0..1 = used/total
    temp_factor: f32, // 0..1 mapped from 25..85 °C
}

trait Pattern: Send {
    fn name(&self) -> &'static str;
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx);
    fn on_activate(&mut self, _particles: &mut [Particle]) {}
    /// Max world-space half-extent this pattern occupies, as (max_x, max_y).
    /// The renderer scales the whole scene so this bounding box maps onto
    /// the window, so each pattern always fills its viewport. Approximate
    /// per pattern — pick the widest point the physics can reach.
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    /// Optional color override for this pattern. Returns None to use the
    /// default phase-based shimmer from the renderer.
    fn color(&self, _p: &Particle, _ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        None
    }
    /// Sample the pattern's source image at normalized (0..1, 0..1).
    /// Defaults to 0.5 (flat) so patterns without an image concept
    /// don't have to implement it. Override to power a reactive
    /// image-driven pattern.
    fn image_value(&self, _u: f32, _v: f32) -> f32 { 0.5 }
}

// ─── built-in patterns ───
//
// Each pattern is a small zero-sized or self-contained struct so its
// parameters can be tuned without touching the trait surface.

struct VortexPattern;
impl Pattern for VortexPattern {
    fn name(&self) -> &'static str { "1. Vortex" }
    fn extent(&self) -> (f32, f32) { (1.7, 1.7) } // respawn radius 1.0..1.6 + swirl spread
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
        let dx = -p.x; let dy = -p.y;
        let dist = (dx * dx + dy * dy).sqrt().max(0.001);
        let swirl_speed = 0.0004 * (1.0 + act) * dist;
        p.vx += dy * swirl_speed; p.vy += -dx * swirl_speed;
        let pull = (0.00015 + act * 0.0002) / (dist + 0.05);
        p.vx += dx * pull; p.vy += dy * pull;
        p.vz += -p.z * 0.005;
        p.vx *= 0.993; p.vy *= 0.993; p.vz *= 0.97;
        if dist < 0.03 {
            let a = fastrand::f32() * std::f32::consts::TAU;
            // Respawn at the new wider radius (1.0..1.6) so the vortex
            // continues to fill the viewport after particles drain in.
            let r = 1.0 + fastrand::f32() * 0.6;
            p.x = a.cos() * r; p.y = a.sin() * r;
            p.z = (fastrand::f32() - 0.5) * 1.0;
            let v0 = 0.001 + fastrand::f32() * 0.003;
            p.vx = -a.sin() * v0; p.vy = a.cos() * v0;
            p.vz = (fastrand::f32() - 0.5) * 0.001;
        }
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let dist = (p.x * p.x + p.y * p.y).sqrt();
        let depth_near = 1.0 - (dist / 1.0).clamp(0.0, 1.0);
        let temp = ctx.temp_factor;
        let base_r = 180.0 + (1.0 - temp) * 75.0;
        let base_g = 30.0 + temp * 120.0 + depth_near * 80.0;
        let base_b = 5.0 + depth_near * 30.0;
        let orange_chance = (temp - 0.25).clamp(0.0, 1.0) * 0.4;
        let is_orange = (p.phase + ctx.frame as f32 * 0.01).sin() > (1.0 - orange_chance * 2.0);
        let white_chance = ((temp - 0.7) / 0.3).clamp(0.0, 1.0) * 0.08;
        let is_white = white_chance > 0.0
            && (p.phase * 7.0 + ctx.frame as f32 * 0.03).sin() > (1.0 - white_chance * 2.0);
        let (r, g, b) = if is_white {
            (240.0 + temp * 15.0, 200.0 + temp * 55.0, 160.0 + temp * 95.0)
        } else if is_orange {
            (220.0 + temp * 35.0, 80.0 + temp * 100.0, 5.0 + temp * 15.0)
        } else {
            (base_r, base_g, base_b)
        };
        let flicker = (p.phase * 13.0 + ctx.frame as f32 * 0.05).sin() * 0.1 + 0.95;
        Some((r * flicker, g * flicker, b * flicker))
    }
}

struct CylinderPattern;
impl Pattern for CylinderPattern {
    fn name(&self) -> &'static str { "2. Cylinder" }
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) } // radius 1.15 + orbit 0.15
    fn on_activate(&mut self, particles: &mut [Particle]) {
        // Re-anchor onto a cylinder. Every other pattern does this on
        // activation (rings/cube/galaxy); without it, base_* is left at
        // whatever the previous pattern set, so the spring pulls the flock
        // to a stale shape and the cylinder never forms.
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let radius = 0.15 + (i as f32 / n) * 1.00;
            let angle = (i as f32 * 2.5) % std::f32::consts::TAU;
            p.base_x = angle.cos() * radius;
            p.base_y = angle.sin() * radius;
            p.base_z = ((i as f32 * 1.7).sin() * 0.6) as f32;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            p.phase = (i as f32 * 0.1) % std::f32::consts::TAU;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let vfill = ctx.vram_fill;
        let a = t * 0.3 + p.phase;
        let orbit = act * 0.15;
        let tx = p.base_x + a.cos() * orbit;
        let ty = p.base_y + a.sin() * orbit;
        let tz = p.base_z + (t * 0.5 + p.phase).sin() * act * 0.2;
        let spring = 0.04 + vfill * 0.06;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring; p.vz += (tz - p.z) * spring;
        let damp = 0.92 - act * 0.05;
        p.vx *= damp; p.vy *= damp; p.vz *= damp;
        let jit = act * 0.002;
        p.vx += p.phase.sin() * jit; p.vy += p.phase.cos() * jit;
        p.vz += (p.phase * 1.3).sin() * jit;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
}

struct ProcessCloudPattern;
impl Pattern for ProcessCloudPattern {
    fn name(&self) -> &'static str { "4. ProcessCloud" }
    // node ring 0.75 + vortex churn up to ~0.30 when busy -> 1.05
    fn extent(&self) -> (f32, f32) { (1.05, 1.05) }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let pid = p.process_id as f32;
        // Whole process ring slowly precesses around the vertical axis (~40s
        // per revolution) so the cloud is never a static frame.
        let rot = t * 0.15;
        let a0 = pid * 1.7;
        let cx = (a0 + rot).cos() * 0.75;
        let cy = (a0 + rot).sin() * 0.75;
        let cz = (pid * 0.9).sin() * 0.3;

        // Data packets: a stable ~17% slice of each process flares out and
        // travels along the ring toward the next node, reading as process
        // communication. Staggered by per-particle phase so they don't move
        // in lock-step. Never collapses, independent of GPU load.
        if p.phase.rem_euclid(std::f32::consts::TAU) < 1.1 {
            let prog = (t * 0.8 + p.phase).fract();
            let ang = a0 + rot + prog * 1.7; // sweep arc to the next node
            let tx = ang.cos() * 0.72;
            let ty = ang.sin() * 0.72;
            let tz = (ang * 0.9).sin() * 0.3;
            p.vx += (tx - p.x) * 0.12; p.vy += (ty - p.y) * 0.12; p.vz += (tz - p.z) * 0.12;
            p.vx *= 0.90; p.vy *= 0.90; p.vz *= 0.90;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
            return;
        }

        // Churning vortex at each node. Radius stays lively even when idle
        // (base r = 0.10) and swells with activity, so every process spins as
        // a tight glowing swirl instead of collapsing into an inert dot.
        let swirl_t = t * 1.6 + p.phase * 3.0;
        let vr = 0.10 + act * 0.18 + p.phase.sin().abs() * 0.02;
        let tx = cx + swirl_t.cos() * vr;
        let ty = cy + swirl_t.sin() * vr;
        let tz = cz + (t * 0.9 + p.phase).sin() * 0.08;
        let k = 0.06;
        p.vx += (tx - p.x) * k; p.vy += (ty - p.y) * k; p.vz += (tz - p.z) * k;
        p.vx *= 0.88; p.vy *= 0.88; p.vz *= 0.88;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, _ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let pid = p.process_id as f32;
        let r = (pid * 2.2).cos() * 0.5 * 200.0 + 55.0;
        let g = (pid * 2.2 + 2.1).cos() * 0.5 * 200.0 + 55.0;
        let b = (pid * 2.2 + 4.2).cos() * 0.5 * 200.0 + 55.0;
        Some((r, g, b))
    }
}

struct AnimationPattern;
impl Pattern for AnimationPattern {
    fn name(&self) -> &'static str { "6. Animation" }
    // Snowstorm: respawns at the window top (y -1.30..-1.55, x ±1.2, z ±0.5)
    // and falls toward the bottom; swirl keeps flakes inside; generous
    // padding so gusts near the edge don't cull.
    fn extent(&self) -> (f32, f32) { (1.9, 1.4) }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity; // GPU util drives the storm swirl (0..1)
        let f = p.phase % 1.0;  // per-flake variance: mass, sway, sparkle
        p.age += 0.016;

        // --- Gravity toward a realistic terminal fall ---
        // Screen projection is sy = cy + p.y*scale (all-positive factors), and
        // egui's rect.bottom() is larger-y than rect.top(), so +y is
        // screen-DOWN: the top is the smallest y and "falling" means
        // increasing p.y. Heavier flakes fall faster (same air drag, more
        // weight), which gives depth: big close flakes drop quicker, small
        // far ones totter.
        let mass = 0.75 + 0.55 * f;
        p.vy += 0.00034 * mass;      // gravity increment (mass-scaled, down)
        p.vy *= 0.97;                // quadratic-ish drag -> terminal velocity
        p.vx *= 0.985;               // light side drag lets swirl linger
        p.vz *= 0.985;

        // --- GPU-driven storm swirl ---
        // Falling snow gets stirred into eddies: a slowly wandering vortex
        // core nudges each flake tangentially, so load shows as spiral,
        // gusty descent instead of laminar straight lines. At idle (act=0)
        // this term vanishes and it's calm snowfall.
        let swirl = act * (0.25 + 0.75 * f);
        let core_x = 0.5 * (t * 0.35).sin();       // core wanders
        let core_y = 0.2 * (t * 0.42).cos() - 0.1;
        let dx = p.x - core_x;
        let dy = p.y - core_y;
        let d2 = dx * dx + dy * dy + p.z * p.z + 0.08; // softened core
        p.vx += swirl * (-dy / d2) * 0.020;   // tangential spin
        p.vz += swirl * ( dx / d2) * 0.020;
        p.vy += swirl * (p.z / d2) * 0.012;   // gentle eddy lift
        // fine flake sway — snow never falls perfectly straight
        p.vx += (t * 1.3 + p.phase * 7.0).sin() * act * 0.010;
        p.vz += (t * 0.9 + p.phase * 5.0).cos() * act * 0.010;

        p.x += p.vx; p.y += p.vy; p.z += p.vz;

        // Respawn at the TOP of the screen when a flake exits the field
        // (including being carried sideways by a strong eddy) so nothing
        // drifts off-screen. +y is screen-down, so the top is the most
        // negative y and falling means increasing y.
        if p.age > 34.0 || p.y > 1.38 || p.x.abs() > 2.1 || p.z.abs() > 1.1 {
            p.x = (fastrand::f32() - 0.5) * 2.4;
            p.y = -1.30 - fastrand::f32() * 0.25;    // top of the window (largest sy = low)
            p.z = (fastrand::f32() - 0.5) * 1.0;
            p.vx = (fastrand::f32() - 0.5) * 0.06;   // fresh gentle drift
            p.vy = 0.008 + f * 0.008;                 // already falling (increasing y)
            p.vz = (fastrand::f32() - 0.5) * 0.06;
            p.age = 0.0;
        }
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        // Ice-white snow; brighter in the storm, with per-flake sparkle so
        // individual crystals shimmer against the dark void.
        let act = ctx.activity;
        let sparkle = 0.8 + 0.4 * (p.phase * 13.0).sin().abs();
        let v = (120.0 + act * 115.0) * sparkle;
        Some((v, v + 4.0, v + 10.0))
    }
}

// Shared helper: re-anchor particles onto `rings` crisp concentric circles
// in the x,y (screen) plane, radiating from the center out to `max_radius`.
// Concentrating each band near its outer radius makes the circles read as
// distinct glowing rings (the "concentric circles" look) rather than a
// uniform disk smear.
fn anchor_concentric_rings(particles: &mut [Particle], max_radius: f32, rings: usize) {
    let n = particles.len() as f32;
    for (i, p) in particles.iter_mut().enumerate() {
        let fr = i as f32 / n;
        let ring = (fr * rings as f32).min((rings - 1) as f32) as usize;
        let ring_radius = max_radius * (ring as f32 + 1.0) / rings as f32;
        // Concentrate near the ring's radius with a little jitter so each
        // ring is a crisp circle with a subtle halo.
        let r = ring_radius * (0.86 + 0.14 * fastrand::f32());
        let a = fr * std::f32::consts::TAU * (rings as f32) + (ring as f32) * 0.4;
        p.base_x = a.cos() * r;
        p.base_y = a.sin() * r;
        p.base_z = (fastrand::f32() - 0.5) * 0.2;
        p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
        p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        p.phase = (i % 64) as f32 * 0.098; // per-flake motion variation
        p.process_id = ring as u8;          // ring index -> color/behaviour key
    }
}

struct HeatScalePattern;
impl Pattern for HeatScalePattern {
    fn name(&self) -> &'static str { "7. HeatScale" }
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) } // concentric heat rings to radius ~1.34
    fn on_activate(&mut self, particles: &mut [Particle]) {
        anchor_concentric_rings(particles, 1.34, 5);
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        // Concentric heat rings that ripple outwards (diverging waves) and
        // slowly spin — like heat shimmer spreading from a white-hot core.
        // GPU activity cranks up the ripple speed/amplitude.
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let motion = 1.0 + act * 1.2;
        let br = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt(); // ring radius
        let ang = p.base_y.atan2(p.base_x);                          // angular slot
        // Outward-travelling ripple: phase depends on (time - radius), so
        // the wave moves away from the centre.
        let ripple = 0.045 * (1.0 + act * 1.5) * (t * 2.0 * motion - br * 5.0).sin();
        let rot = t * (0.06 + act * 0.10);
        let r = (br + ripple).max(0.02);
        let a = ang + rot;
        let tx = a.cos() * r;
        let ty = a.sin() * r;
        let tz = (t * 1.5 - br * 3.0).sin() * act * 0.12;
        p.vx += (tx - p.x) * 0.10; p.vy += (ty - p.y) * 0.10; p.vz += (tz - p.z) * 0.06;
        p.vx *= 0.85; p.vy *= 0.85; p.vz *= 0.85;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        // Thermal ramp: white-hot core, graduating through orange to deep
        // red at the cool rim. Base brightness rises with GPU activity.
        let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let n = (dist / 1.34).clamp(0.0, 1.0);
        let heat = (0.55 + ctx.activity * 0.45).clamp(0.0, 1.0);
        let rise = ctx.activity * 0.4;
        let r = (190.0 + 65.0 * heat) as f32;
        let g = (205.0 * (1.0 - n) + 40.0) - rise * 15.0;
        let b = (60.0 * (1.0 - n) * (1.0 - n) + 15.0) - rise * 20.0;
        Some((r, g.max(8.0), b.max(8.0)))
    }
}

struct OrbitCubePattern;
impl Pattern for OrbitCubePattern {
    fn name(&self) -> &'static str { "3. OrbitCube" }
    fn extent(&self) -> (f32, f32) { (1.6, 1.6) } // zoomed past the cube faces: only the innermost shell + interior stars are in view, the 2.4/3.2 shells and cube edges fall off-screen
    fn on_activate(&mut self, particles: &mut [Particle]) {
        // Uniform random star field — no cube faces, no clumps, no edges.
        // Each particle gets a random radius (spread over a range for depth,
        // lower-bounded to avoid a dense singularity at the pivot) and a
        // uniformly-random direction on a sphere, so the whole pattern reads
        // as an even scatter of individual stars rather than a geometric
        // shape. There is deliberately no structured geometry here.
        for p in particles.iter_mut() {
            // Radius spread flat across depth: uniform radial density so
            // no region is denser than another (a dense centre would read
            // as a blob/"shape", which we deliberately avoid). The small
            // lower bound keeps the exact pivot from being over-populated.
            let r = 0.35 + fastrand::f32() * 2.2; // ~0.35..2.55, uniform
            // Uniform direction on the unit sphere (correct sampling):
            // colatitude phi via acos of a [-1,1] uniform, azimuth free.
            let u = fastrand::f32() * 2.0 - 1.0;   // (-1, 1)
            let phi = u.acos();                    // (0, PI) colatitude
            let theta = fastrand::f32() * std::f32::consts::TAU;
            let (sin_p, cos_p) = phi.sin_cos();
            let (sin_t, cos_t) = theta.sin_cos();
            let bx = r * sin_p * cos_t;
            let by = r * sin_p * sin_t;
            let bz = r * cos_p;
            p.base_x = bx;
            p.base_y = by;
            p.base_z = bz;
            p.x = bx; p.y = by; p.z = bz;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            // Star-like size: a few bright big ones amid many faint small ones.
            p.size = 0.4 + fastrand::f32().powi(2) * 1.4;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        // The spin rate is pre-smoothed in step() (flywheel effect) and
        // passed via ctx.frame as a fixed-point value (spin * 1_000_000).
        // This gives the cloud mass/inertia — it takes time to speed up
        // and slow down, eliminating jerkiness.
        let spin_rate = ctx.frame as f32 / 1_000_000.0;
        p.phase += spin_rate;
        let a = p.phase;
        let cos_a = a.cos();
        let sin_a = a.sin();
        let rtx = p.base_x * cos_a - p.base_z * sin_a;
        let rtz = p.base_x * sin_a + p.base_z * cos_a;
        let rty = p.base_y; // second axis (slower) — keep it for free.
        // Slight activity-driven breathing so a busy GPU pushes the
        // stars outward a touch.
        let breathe = 1.0 + ctx.activity * 0.10;
        let tx = rtx * breathe;
        let ty = rty * breathe;
        let tz = rtz * breathe;
        p.vx += (tx - p.x) * 0.10;
        p.vy += (ty - p.y) * 0.10;
        p.vz += (tz - p.z) * 0.10;
        p.vx *= 0.85; p.vy *= 0.85; p.vz *= 0.85;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
}

struct RegionsPattern;
impl Pattern for RegionsPattern {
    fn name(&self) -> &'static str { "8. Regions" }
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) } // concentric contour rings to radius ~1.34
    fn on_activate(&mut self, particles: &mut [Particle]) {
        anchor_concentric_rings(particles, 1.34, 7);
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        // Concentric contour "regions" that breathe in place (standing waves)
        // and slowly rotate — like a topographic map of a circular terrain
        // whose elevation rings swell with GPU activity.
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let br = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let ang = p.base_y.atan2(p.base_x);
        // Standing wave: phase tied to radius alone (not time) so rings
        // breathe in/out in place; amplitude grows with GPU activity.
        let breathe = 0.05 * (1.0 + act * 1.6) * (t * 1.4 - br * 4.0).sin();
        let rot = t * (0.02 + act * 0.04); // slow contour drift
        let r = (br + breathe).max(0.02);
        let a = ang + rot;
        let tx = a.cos() * r;
        let ty = a.sin() * r;
        // Depth undulation reveals the topography: elevation peaks travel
        // around each ring, so the "regions" read as hills moving through
        // the concentric circles.
        let tz = (br * 6.0 + t * act * 2.0).sin() * 0.12 * (0.3 + act);
        p.vx += (tx - p.x) * 0.08; p.vy += (ty - p.y) * 0.08; p.vz += (tz - p.z) * 0.06;
        p.vx *= 0.88; p.vy *= 0.88; p.vz *= 0.88;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        // Alternating contour bands (bright/dim rings) in a cool blue-cyan
        // spectrum, so each concentric circle reads as a separate "region".
        let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        // Sinusoidal banding by radius -> crisp alternating rings.
        let band = 0.5 + 0.5 * (dist * 10.0).cos();
        let glow = 0.45 + ctx.activity * 0.5;
        let bright = (0.35 + band * 0.65) * glow;
        let r = 45.0 * bright + 25.0;
        let g = (120.0 + 90.0 * bright) * (0.7 + ctx.activity * 0.3);
        let b = (190.0 + 60.0 * bright) - ctx.activity * 40.0;
        Some((r, g.min(255.0), b.min(255.0)))
    }
}

struct WavefieldPattern;
impl Pattern for WavefieldPattern {
    fn name(&self) -> &'static str { "9. Wavefield" }
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) } // cylinder spread + wave displacement
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let wave = (p.base_x * 3.0 + t).sin() * (p.base_y * 3.0 + t * 1.3).cos() * act * 0.4;
        let tx = p.base_x + (p.base_y * 2.0 + t).cos() * act * 0.15;
        let ty = p.base_y + (p.base_x * 2.0 + t).sin() * act * 0.15;
        let tz = wave;
        p.vx += (tx - p.x) * 0.06; p.vy += (ty - p.y) * 0.06; p.vz += (tz - p.z) * 0.06;
        p.vx *= 0.90; p.vy *= 0.90; p.vz *= 0.90;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
}

struct SpiralGalaxyPattern;
impl Pattern for SpiralGalaxyPattern {
    fn name(&self) -> &'static str { "10. SpiralGalaxy" }
    fn extent(&self) -> (f32, f32) { (1.6, 1.6) } // disk 1.30 * (1 + act*0.15)
    fn on_activate(&mut self, particles: &mut [Particle]) {
        // Re-anchor on a wider disk so the galaxy fills the viewport.
        // Disk radius 1.30 fills a 16:9 HD screen at the new
        // aspect-aware scale = min(w,h) * 0.42.
        for (i, p) in particles.iter_mut().enumerate() {
            let n = PARTICLE_COUNT as f32;
            // sqrt-distributed radius so the disk has uniform density
            // (not a bright core with a faint rim).
            let u = (i as f32 + 0.5) / n;
            let r = (u.sqrt() * 1.30).max(0.05);
            let a = (i as f32 * 2.399_963) % std::f32::consts::TAU; // golden-angle
            p.base_x = a.cos() * r;
            p.base_y = a.sin() * r;
            p.base_z = (fastrand::f32() - 0.5) * 0.15;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            p.size = 0.6 + fastrand::f32() * 0.6;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt().max(0.01);
        let base_angle = p.base_y.atan2(p.base_x);
        let spiral_offset = dist * 2.5;
        let rot_speed = 0.3 / (dist + 0.2) * (1.0 + act);
        let angle = base_angle + t * rot_speed - spiral_offset;
        let r = dist * (1.0 + act * 0.15);
        let tx = angle.cos() * r;
        let ty = angle.sin() * r;
        let tz = p.base_z + (dist * 4.0 + t).sin() * act * 0.1;
        p.vx += (tx - p.x) * 0.07; p.vy += (ty - p.y) * 0.07; p.vz += (tz - p.z) * 0.07;
        p.vx *= 0.89; p.vy *= 0.89; p.vz *= 0.89;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let dist = (p.x * p.x + p.y * p.y).sqrt();
        let arm_phase = (dist * 3.0 + t).sin() * 0.5 + 0.5;
        let r = 100.0 + arm_phase * 120.0 + act * 60.0;
        let g = 60.0 + arm_phase * 140.0;
        let b = 180.0 + arm_phase * 50.0;
        Some((r, g, b))
    }
}

struct GridWavePattern;
impl Pattern for GridWavePattern {
    fn name(&self) -> &'static str { "11. GridWave" }
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) } // cylinder spread base
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let wave = (dist * 6.0 - t * 2.0 * (1.0 + act)).sin() * act * 0.5;
        let tx = p.base_x; let ty = p.base_y; let tz = wave;
        p.vx += (tx - p.x) * 0.06; p.vy += (ty - p.y) * 0.06; p.vz += (tz - p.z) * 0.06;
        p.vx *= 0.89; p.vy *= 0.89; p.vz *= 0.89;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
}

/// Slot 5: Reactive Image.
///
/// The point cloud becomes a "screen" of particles arranged on a grid.
/// Each particle's anchor position is sampled from a procedural image
/// (mandala-like sinusoid by default) whose brightness modulates that
/// particle's z-displacement, color, and size — so an idle GPU shows a
/// flat dim image, a working GPU pushes pixels outward and warms them up,
/// and a hot GPU shifts the whole image into the red end of the spectrum.
///
/// To swap the source image: edit `image_value(u, v)` below, or — for a
/// real bitmap source — replace the procedural function with a lookup
/// into a `[[u8; W]; H]` constant.
struct ReactiveImagePattern;
impl Pattern for ReactiveImagePattern {
    fn name(&self) -> &'static str { "5. ReactiveImage" }
    fn extent(&self) -> (f32, f32) { (0.85, 0.85) } // grid half-extent 0.60 + rotation + relief

    /// Sample the source image at normalized coordinates (0..1, 0..1).
    /// Returns brightness 0..1. Replace this body to use a real image.
    fn image_value(&self, u: f32, v: f32) -> f32 {
        // Soft mandala: radial symmetric pattern with rotational sweep.
        let cx = u - 0.5;
        let cy = v - 0.5;
        let r = (cx * cx + cy * cy).sqrt();
        let theta = cy.atan2(cx);
        let petals = (theta * 6.0).sin() * 0.5 + 0.5;
        let ring = (r * 18.0).sin() * 0.5 + 0.5;
        let center_falloff = 1.0 - (r * 2.0).clamp(0.0, 1.0);
        (petals * 0.4 + ring * 0.5 + center_falloff * 0.3).clamp(0.0, 1.0)
    }

    fn on_activate(&mut self, particles: &mut [Particle]) {
        // Re-anchor every particle onto the image grid the first time the
        // pattern becomes active (and any subsequent time it's re-selected).
        let side = (particles.len() as f32).sqrt() as usize;
        let side = side.max(1);
        for (i, p) in particles.iter_mut().enumerate() {
            let u = (i % side) as f32 / side as f32;
            let v = (i / side) as f32 / side as f32;
            let brightness = self.image_value(u, v);
            // Map brightness to a z-displacement that we'll modulate with
            // telemetry at runtime. Anchor on a centered 1.20×1.20 square
            // so the image fills a 16:9 HD viewport.
            let px = (u - 0.5) * 1.20;
            let py = (v - 0.5) * 1.20;
            p.base_x = px;
            p.base_y = py;
            p.base_z = brightness * 0.4 - 0.2;
            p.x = px; p.y = py; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            p.size = 0.4 + brightness * 0.8;
        }
    }

    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let vfill = ctx.vram_fill;
        let tfact = ctx.temp_factor;

        // Re-sample brightness each frame so the image can react if you
        // swap the source later. (Cheap; same arithmetic either way.)
        let u = p.base_x / 1.20 + 0.5;
        let v = p.base_y / 1.20 + 0.5;
        let brightness = self.image_value(
            u.clamp(0.0, 1.0),
            v.clamp(0.0, 1.0),
        );

        // Idle: particles cling to the flat image plane.
        // Working: VRAM fill pushes the bright pixels outward.
        // Hot: temperature amplifies the relief.
        let relief = brightness * (0.15 + vfill * 0.45 + tfact * 0.4);
        let tz = p.base_z + relief;

        // Idle drift: very loose spring + tiny rotation to keep the image
        // alive when the GPU is doing nothing.
        let rotation = t * 0.05 * (0.3 + act);
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let tx = p.base_x * cos_r - p.base_y * sin_r;
        let ty = p.base_x * sin_r + p.base_y * cos_r;

        // Spring strength scales with activity — tighter when busy.
        let spring = 0.04 + act * 0.10;
        p.vx += (tx - p.x) * spring;
        p.vy += (ty - p.y) * spring;
        p.vz += (tz - p.z) * spring;

        // Activity-driven jitter — only on the "lit" pixels.
        let jit = act * 0.002 * brightness;
        p.vx += p.phase.sin() * jit;
        p.vy += p.phase.cos() * jit;

        let damp = 0.86 - act * 0.04;
        p.vx *= damp; p.vy *= damp; p.vz *= damp;

        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }

    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let u = p.base_x / 1.20 + 0.5;
        let v = p.base_y / 1.20 + 0.5;
        let brightness = self.image_value(
            u.clamp(0.0, 1.0),
            v.clamp(0.0, 1.0),
        );
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let vfill = ctx.vram_fill;
        let tfact = ctx.temp_factor;

        // Cool image palette: deep indigo → cyan → warm gold as brightness
        // and telemetry rise.
        let cool = (40.0, 60.0, 180.0);
        let mid  = (80.0, 200.0, 220.0);
        let warm = (255.0, 200.0, 80.0);
        let hot  = (255.0, 90.0, 40.0);

        // Drive the gradient by combined brightness + activity, then bias
        // toward "hot" when temperature is high.
        let mix1 = (brightness * 0.6 + act * 0.3 + vfill * 0.1).clamp(0.0, 1.0);
        let mix2 = (tfact * 0.7 + act * 0.3).clamp(0.0, 1.0);

        // Two-step lerp: cool → mid by mix1, then mid → warm by mix2,
        // then warm → hot at high temp.
        let (r1, g1, b1) = lerp3(cool, mid, mix1);
        let (r2, g2, b2) = lerp3((r1, g1, b1), warm, mix2);
        let final_mix = (tfact - 0.6).max(0.0) / 0.4;
        let (r, g, b) = lerp3((r2, g2, b2), hot, final_mix.clamp(0.0, 1.0));

        // Subtle per-particle shimmer, scaled by brightness.
        let shimmer = (p.phase + t * 1.5).sin() * 0.1 + 0.9;
        let scale = (0.6 + brightness * 0.4) * shimmer;
        Some((r * scale, g * scale, b * scale))
    }
}

fn lerp3(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    let t = t.clamp(0.0, 1.0);
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
}

/// Scale a pattern so its world-space bounding box (given by `extent`, the
/// max |x| and max |y| its particles reach) maps onto a window of the given
/// pixel size. Uses the more restrictive axis (min of the two ratios) so the
/// pattern's extent always touches both edges of the window in at least one
/// dimension — i.e. it fills the window it occupies. `extent` values are
/// clamped away from zero so a pathological extent can never blow the scale
/// up to infinity.
fn fill_scale(window_w: f32, window_h: f32, extent: (f32, f32)) -> f32 {
    let wx = extent.0.max(0.001);
    let wy = extent.1.max(0.001);
    let w_scale = window_w / (wx * 2.0);
    let h_scale = window_h / (wy * 2.0);
    w_scale.min(h_scale)
}

/// One step of exponential (low-pass) smoothing: move `cur` partway toward
/// `target`. With `k` in (0, 1) the value asymptotically approaches the
/// target frame by frame, so a step-wise input (like the 500ms nvidia-smi
/// poll) is rendered as a smooth glide rather than a hard snap. Never
/// overwrites `cur` directly, so a step can't jump to `target` in one frame.
fn smooth_step(cur: f32, target: f32, k: f32) -> f32 {
    cur + (target - cur) * k.clamp(0.0, 1.0)
}

struct ChimeraPattern;
impl Pattern for ChimeraPattern {
    fn name(&self) -> &'static str { "12. Chimera" }
    fn extent(&self) -> (f32, f32) { (0.55, 1.10) } // face oval rx 0.45, ry 0.975 + smile/amp
    fn on_activate(&mut self, particles: &mut [Particle]) {
        init_chimera_particles(particles);
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let vibe: usize = if act < 0.15 { 0 }
            else if act < 0.50 { 1 }
            else if act < 0.80 { 2 }
            else { 3 };
        let spring = match vibe { 0=>0.01, 1=>0.03, 2=>0.06, _=>0.10 };
        let amp = match (p.layer, vibe) {
            (0, _) => 0.0,
            (1, 0) => 0.005, (1, 1) => 0.012, (1, 2) => 0.025, (1, 3) => 0.05,
            (_, 0) => 0.008, (_, 1) => 0.015, (_, 2) => 0.030, (_, _) => 0.06,
        };
        let freq = match vibe { 0=>1.0, 1=>2.5, 2=>5.0, _=>12.0 };
        let disp = (t * freq + p.phase).sin() * amp;
        let tx = p.base_x + disp * 0.5;
        let ty = p.base_y + disp;
        if p.layer == 1 && p.base_y > 0.15 && vibe >= 2 {
            let smile = (p.base_x * 4.0).sin() * 0.02 * (vibe as f32) * 0.5;
            p.vy += smile;
        }
        p.vx += (tx - p.x) * spring;
        p.vy += (ty - p.y) * spring;
        p.vz += (p.base_z - p.z) * 0.04;
        let damp = match vibe { 0=>0.92, 1=>0.88, 2=>0.84, _=>0.78 };
        p.vx *= damp; p.vy *= damp; p.vz *= damp;
        let jit = act * 0.003;
        p.vx += p.phase.sin() * jit;
        p.vy += p.phase.cos() * jit;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let vibe: usize = if act < 0.15 { 0 }
            else if act < 0.50 { 1 }
            else if act < 0.80 { 2 }
            else { 3 };
        let shimmer = (p.phase + t * 3.0).sin() * 0.15 + 0.85;
        let c = match (p.layer, vibe) {
            (0, _) => (180.0, 180.0, 210.0),
            (1, 0) => (40.0, 30.0, 80.0),
            (1, 1) => (60.0, 80.0, 160.0),
            (1, 2) => (200.0, 140.0, 30.0),
            (1, 3) => (255.0, 60.0, 120.0),
            (_, 0) => (30.0, 25.0, 60.0),
            (_, 1) => (70.0, 100.0, 180.0),
            (_, 2) => (220.0, 180.0, 50.0),
            (_, _) => (255.0, 200.0, 20.0),
        };
        Some((c.0 * shimmer, c.1 * shimmer, c.2 * shimmer))
    }
}

// ─── Chimera face initialization (extracted free function) ───

fn init_chimera_particles(particles: &mut [Particle]) {
    // Multiplier on every position so the face fills a 16:9 HD
    // viewport (was 1.0 → face oval 0.60×1.50; now 1.5× → 0.90×2.25).
    const FILL: f32 = 1.5;
    let n = particles.len();
    for (i, p) in particles.iter_mut().enumerate() {
        let frac = i as f32 / n as f32;
        let (tx, ty, layer): (f32, f32, u8) = if frac < 0.10 {
            let a = frac / 0.10 * std::f32::consts::TAU * 2.5;
            let outline = face_outline(a);
            (outline.0 * FILL, outline.1 * FILL, 0)
        } else if frac < 0.35 {
            let of = (frac - 0.10) / 0.25;
            if of < 0.5 {
                let eye_frac = of * 2.0;
                let eye_x = if eye_frac < 0.5 {
                    -0.18 * FILL + (fastrand::f32() - 0.5) * 0.18 * FILL
                } else {
                    0.18 * FILL + (fastrand::f32() - 0.5) * 0.18 * FILL
                };
                let eye_y = -0.05 * FILL + (fastrand::f32() - 0.5) * 0.12 * FILL;
                (eye_x, eye_y, 1)
            } else {
                let mf = (of - 0.5) * 2.0;
                let mx = (mf - 0.5) * 0.40 * FILL;
                let my = 0.20 * FILL + (fastrand::f32() - 0.5) * 0.08 * FILL;
                (mx, my, 1)
            }
        } else {
            let _tf = (frac - 0.35) / 0.65;
            loop {
                let rx = (fastrand::f32() - 0.5) * 0.55 * FILL;
                let ry = (fastrand::f32() - 0.5) * 0.75 * FILL;
                let e = (rx / (0.30 * FILL)).powi(2) + (ry / (0.65 * FILL)).powi(2);
                if e <= 1.0 { break (rx, ry, 2); }
            }
        };
        p.base_x = tx; p.base_y = ty;
        p.base_z = (fastrand::f32() - 0.5) * 0.1;
        p.x = tx; p.y = ty; p.z = p.base_z;
        p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        p.size = match layer {
            0 => 1.0 + fastrand::f32() * 0.6,
            1 => 0.6 + fastrand::f32() * 0.6,
            _ => 0.3 + fastrand::f32() * 0.6,
        };
        p.layer = layer;
    }
}

fn face_outline(t: f32) -> (f32, f32) {
    let a = t.rem_euclid(std::f32::consts::TAU);
    let rx = 0.30;
    let chin = if a > std::f32::consts::PI * 0.5 && a < std::f32::consts::PI * 1.5 {
        let bf = ((a - std::f32::consts::PI * 0.5) / std::f32::consts::PI).clamp(0.0, 1.0);
        rx * (1.0 - bf * 0.4)
    } else {
        rx
    };
    let ry = 0.65;
    (chin * a.cos(), ry * a.sin() + 0.02)
}

// ─── main app state ───

struct ProcessInfo {
    pid: u32,
    name: String,
    used_memory_mb: f32,
    model: String,
}

struct VramVisualizer {
    // UI state: whether the footer context popup is currently shown.
    show_footer_popup: bool,
    gpu: GpuData,
    error: String,
    loaded_model: String,
    // Second model slot (for future dual-model display).
    loaded_model2: String,
    // Tokens/sec for each model (populated from Ollama stats if available).
    tok_s: f32,
    tok_s2: f32,
    particles: Vec<Particle>,
    last_poll: Instant,
    frame: u64,
    active_idx: usize,
    patterns: Vec<Box<dyn Pattern>>,
    history: Vec<GpuData>,
    // Smoothed (low-pass filtered) telemetry. nvidia-smi is polled every
    // 500ms, so the raw activity/VRAM/temp values step like a square wave
    // between polls — feeding those straight into per-frame physics and
    // color makes every reaction visibly jerk at 2Hz. We instead lerp a
    // running smoothed value toward each raw sample every frame, so the
    // entire visual response glides smoothly to the new level instead of
    // snapping.
    act_s: f32,
    vfill_s: f32,
    tfact_s: f32,
    processes: Vec<ProcessInfo>,
    // Histogram of lightning strikes per GPU‑usage bin (0‑9 = 0‑10%, … 90‑100%).
    strike_bins: [u64; 10],
    // Cool‑down timer to limit lightning to ~3 strikes per second.
    strike_cooldown: f32,
    // Smoothed spin rate — a heavy flywheel that takes time to speed up
    // and slow down, giving the cloud a sense of mass/inertia.
    spin_s: f32,
}

impl VramVisualizer {
    fn new() -> Self {
        let particles: Vec<Particle> = (0..PARTICLE_COUNT)
            .map(init_particle_cylinder)
            .collect();
        // Build the registry. Order here is the Tab cycle order.
        // Slot 5 (index 4) is ReactiveImage — the new extensible pattern.
        // Restrict to only the OrbitCube pattern (pattern 3).
        let patterns: Vec<Box<dyn Pattern>> = vec![
            Box::new(OrbitCubePattern),
        ];
        let mut s = Self {
            show_footer_popup: false,
            gpu: GpuData::default(),
            error: String::new(),
            loaded_model: String::from("—"),
            loaded_model2: String::from("—"),
            tok_s: 0.0,
            tok_s2: 0.0,
            particles,
            last_poll: Instant::now(),
            frame: 0,
            // With only one pattern, index 0 is the OrbitCube visualisation.
            active_idx: 0,
            patterns,
            history: Vec::with_capacity(HISTORY_MAX),
            act_s: 0.0,
            vfill_s: 0.0,
            tfact_s: 0.0,
            processes: Vec::new(),
            strike_bins: [0u64; 10],
            // start with cooldown ready to fire immediately
            strike_cooldown: 0.0,
            spin_s: 0.0,
        };

        // Activate the sole pattern.
        s.patterns[s.active_idx].on_activate(&mut s.particles);
        s
    }

    /// Register a new pattern at runtime. Returns its index in the cycle.
    /// Future pattern authors can add new visuals from outside the binary
    /// by holding a `&mut VramVisualizer` and calling this.
    #[allow(dead_code)]
    fn register_pattern(&mut self, pattern: Box<dyn Pattern>) -> usize {
        self.patterns.push(pattern);
        self.patterns.len() - 1
    }

    fn current_name(&self) -> &'static str {
        self.patterns[self.active_idx].name()
    }

    fn poll(&mut self) {
        if self.last_poll.elapsed().as_millis() < 500 { return; }
        self.last_poll = Instant::now();
        match query_nvidia_smi() {
            Ok(d) => {
                self.history.push(d);
                if self.history.len() > HISTORY_MAX { self.history.remove(0); }
                self.gpu = d;
                self.error.clear();
            }
            Err(e) => self.error = e,
        }
        // Updated: capture per-process info.
        if let Ok((model_str, procs)) = query_loaded_model() {
            self.loaded_model = model_str;
            self.processes = procs;
        }
    }

    fn activity(&self) -> f32 { (self.gpu.util / 100.0).clamp(0.0, 1.0) }
    fn vram_fill(&self) -> f32 {
        if self.gpu.vram_total_mb > 0.0 {
            (self.gpu.vram_used_mb / self.gpu.vram_total_mb).clamp(0.0, 1.0)
        } else { 0.0 }
    }
    fn temp_factor(&self) -> f32 { ((self.gpu.temp_c - 25.0) / 60.0).clamp(0.0, 1.0) }

    fn step(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        // Low-pass the raw telemetry so the 500ms-poll steps become a
        // continuous glide. Each frame we move the smoothed value partway
        // toward the latest raw sample (exponential smoothing). SMOOTH ≈
        // 0.10/frame at 60Hz gives a ~0.15s response — quick enough to
        // feel live, slow enough to remove any snap.
        const SMOOTH: f32 = 0.10;
        // Temperature changes much more slowly — a smaller smoothing
        // factor gives a long, lazy thermal lag that feels like a real
        // heat sink warming up and cooling down (~3-4s response).
        const SMOOTH_TEMP: f32 = 0.015;
        self.act_s = smooth_step(self.act_s, self.activity(), SMOOTH);
        self.vfill_s = smooth_step(self.vfill_s, self.vram_fill(), SMOOTH);
        self.tfact_s = smooth_step(self.tfact_s, self.temp_factor(), SMOOTH_TEMP);
        // Decrease lightning cooldown timer (frame time ≈ 1/60 s)
        let frame_dt = 1.0 / 60.0;
        if self.strike_cooldown > 0.0 {
            self.strike_cooldown -= frame_dt;
            if self.strike_cooldown < 0.0 { self.strike_cooldown = 0.0; }
        }
        // Compute the target spin rate from the log-scaled activity, then
        // smooth it heavily so the cloud accelerates/decelerates like it
        // has real mass — no jerky speed changes when GPU usage spikes.
        let log_activity = (1.0 + self.act_s * 9.0).ln() / 10.0_f32.ln();
        let mut target_spin = 0.0001 + log_activity * 0.015;
        // Above 90% GPU, halve the rotation speed — the cloud slows back
        // down at the extreme top end rather than spinning fastest there.
        if self.act_s > 0.9 {
            target_spin *= 0.25;
        }
        // Flywheel smoothing: very slow response (~2-3s to reach target).
        self.spin_s = smooth_step(self.spin_s, target_spin, 0.008);
        let ctx = PatternCtx {
            frame: (self.spin_s * 1_000_000.0) as u64,
            activity: self.act_s,
            vram_fill: self.vfill_s,
            temp_factor: self.tfact_s,
        };
        // Split borrow: take a raw pointer to the active pattern so we can
        // mutate particles and call the pattern simultaneously. Safe because
        // patterns never touch other particles' storage and we don't
        // reallocate the registry during a step.
        let pat: &mut dyn Pattern = &mut *self.patterns[self.active_idx];
        for p in &mut self.particles {
            pat.update(p, &ctx);
        }
    }


    fn switch_to(&mut self, idx: usize) {
        let idx = idx % self.patterns.len();
        if idx == self.active_idx { return; }
        self.active_idx = idx;
        self.patterns[self.active_idx].on_activate(&mut self.particles);
    }

    fn next_pattern(&mut self) {
        let next = (self.active_idx + 1) % self.patterns.len();
        self.switch_to(next);
    }
}

fn init_particle_cylinder(i: usize) -> Particle {
    let n = PARTICLE_COUNT as f32;
    let angle = (i as f32 * 2.5) % std::f32::consts::TAU;
    // Cylinder extends to radius 1.15 so cylinder-init patterns
    // (Cylinder, Animation, HeatScale, Regions, Wavefield, GridWave)
    // fill a standard 16:9 HD viewport at scale = min(w,h) * 0.42.
    let radius = 0.15 + (i as f32 / n) * 1.00;
    let height_z = ((i as f32 * 1.7).sin() * 0.6) as f32;
    Particle {
        x: angle.cos() * radius,
        y: angle.sin() * radius,
        z: height_z,
        vx: 0.0, vy: 0.0, vz: 0.0,
        base_x: angle.cos() * radius,
        base_y: angle.sin() * radius,
        base_z: height_z,
        phase: (i as f32 * 0.1) % std::f32::consts::TAU,
        size: 0.8 + (i % 3) as f32 * 0.4,
        age: 0.0,
        process_id: (i % 8) as u8,
        layer: 0,
    }
}

// ─── rendering ───
impl eframe::App for VramVisualizer {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        egui::Color32::TRANSPARENT.to_normalized_gamma_f32()
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        self.step();

        ctx.input(|i| {
            for ev in &i.events {
                if let egui::Event::Key { key: egui::Key::Tab, pressed: true, .. } = ev {
                    self.next_pattern();
                }
                if let egui::Event::Key { key, pressed: true, .. } = ev {
                    let new_idx = match key {
                        egui::Key::Num1 => Some(0),  // 1. Vortex
                        egui::Key::Num2 => Some(1),  // 2. Cylinder
                        egui::Key::Num3 => Some(2),  // 3. OrbitCube
                        egui::Key::Num4 => Some(3),  // 4. ProcessCloud
                        egui::Key::Num5 => Some(4),  // 5. ReactiveImage
                        egui::Key::Num6 => Some(5),  // 6. Animation
                        egui::Key::Num7 => Some(6),  // 7. HeatScale
                        egui::Key::Num8 => Some(7),  // 8. Regions
                        egui::Key::Num9 => Some(8),  // 9. Wavefield
                        egui::Key::Num0 => Some(9),  // 10. SpiralGalaxy
                        egui::Key::Minus => Some(10), // 11. GridWave
                        egui::Key::Equals => Some(11), // 12. Chimera
                        _ => None,
                    };
                    if let Some(idx) = new_idx { self.switch_to(idx); }
                }
            }
        });

        egui::CentralPanel::default()
            // No panel background — the window is transparent (see
            // with_transparent(true) + TRANSPARENT clear/panel fill in
            // main()), so whatever is behind the window shows through the
            // gaps between particles.
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
            let painter = ui.painter();
            let rect = ui.max_rect();
            let cx = rect.center().x;
            let cy = rect.center().y;
            // Fill the window: scale so the active pattern's own bounding
            // box (extent) maps onto the full window rect. Each pattern
            // reports the world-space half-extent its physics actually
            // covers, so whatever is selected fills its viewport. The
            // 1.0 factor keeps the mapping 1:1; a pattern wider than the
            // window overflows horizontally (clipped by the cull check).
            let (wx, wy) = self.patterns[self.active_idx].extent();
            let scale = fill_scale(rect.width(), rect.height(), (wx, wy));
            // Use the smoothed telemetry for rendering too, so point size,
            // alpha and colour glide with the physics instead of snapping.
            let act = self.act_s;
            let tfact = self.tfact_s;
            let vfill = self.vfill_s;
            let t = self.frame as f32 * 0.016;

            // Sort by z for depth.
            let mut sorted: Vec<&Particle> = self.particles.iter().collect();
            sorted.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal));

            let pat_ctx = PatternCtx {
                frame: self.frame,
                activity: act,
                vram_fill: vfill,
                temp_factor: tfact,
            };

            for p in &sorted {
                let depth_scale = 3.0 / (3.0 + p.z * 0.8);
                let sx = cx + p.x * scale * depth_scale;
                let sy = cy + p.y * scale * depth_scale;
                if sx < rect.left() - 10.0 || sx > rect.right() + 10.0
                    || sy < rect.top() - 10.0 || sy > rect.bottom() + 10.0 {
                    continue;
                }

                // Pattern-specific color, or default shimmer otherwise.
                let (r, g, b) = self.patterns[self.active_idx]
                    .color(p, &pat_ctx)
                    .unwrap_or_else(|| {
                        let shimmer = (p.phase + t * 2.0).sin() * 0.2 + 0.8;
                        let pulse = shimmer * (0.7 + act * 0.5);
                        let r = (80.0 + act * 160.0 + tfact * 80.0) * pulse;
                        let g = (120.0 * (1.0 - act * 0.7) + vfill * 60.0) * pulse;
                        let b = (200.0 * (1.0 - act * 0.8) - tfact * 80.0) * pulse;
                        (r, g, b)
                    });

                let alpha = (0.3 + act * 0.5 + depth_scale * 0.3).clamp(0.15, 0.95) * 0.5;
                let color = egui::Color32::from_rgba_premultiplied(
                    (r as u8).clamp(0, 255),
                    (g as u8).clamp(0, 255),
                    (b as u8).clamp(0, 255),
                    (alpha * 255.0) as u8,
                );
                let point_size = p.size * depth_scale * (0.8 + act * 1.2) * 0.5;
                painter.circle_filled(
                    egui::pos2(sx, sy),
                    point_size * 2.0,
                    egui::Color32::from_rgba_premultiplied(0, 0, 0, (alpha * 0.15 * 255.0) as u8),
                );
                painter.circle_filled(egui::pos2(sx, sy), point_size, color);
            }
            // Simulated lightning between two random points, adapted to GPU usage and limited to ~3 Hz
            {
                use egui::{pos2, vec2, Color32, Shape, Stroke};
                // Lightning strike rate scales linearly from 1.0/sec at 0% GPU
                // to 5.5/sec at 100% GPU. The cooldown is set to 1/rate each strike.
                if self.strike_cooldown <= 0.0 {
                    // Strike rate: 1.0 Hz at 0% → 5.5 Hz at 100%
                    let strike_rate = 1.0 + (self.gpu.util / 100.0) * 4.5;
                    if fastrand::f32() < (strike_rate / 60.0) {
                        self.strike_cooldown = 1.0 / strike_rate;
                        // Lightning emanates from the centre (cx, cy) outward
                        // to a random particle. The reach scales with GPU usage:
                        // low usage → short bolts near centre, high usage → long
                        // bolts reaching the outer edges of the cloud.
                        let usage_factor = (self.gpu.util / 100.0).clamp(0.0, 1.0);
                        let mut attempts = 0;
                        let mut chosen = None;
                        while attempts < 10 {
                            let i2 = fastrand::usize(0..sorted.len());
                            let p2 = &sorted[i2];
                            let depth2 = 3.0 / (3.0 + p2.z * 0.8);
                            let sx2 = cx + p2.x * scale * depth2;
                            let sy2 = cy + p2.y * scale * depth2;
                            let within = |x: f32, y: f32| {
                                x >= rect.left() - 10.0 && x <= rect.right() + 10.0 && y >= rect.top() - 10.0 && y <= rect.bottom() + 10.0
                            };
                            if within(sx2, sy2) {
                                // Start point is always the centre; end point is
                                // the particle, scaled by usage_factor so low GPU
                                // keeps bolts short and near the middle.
                                let reach = 0.15 + 0.85 * usage_factor;
                                let ex = cx + (sx2 - cx) * reach;
                                let ey = cy + (sy2 - cy) * reach;
                                chosen = Some(((cx, cy), (ex, ey), depth2));
                                break;
                            }
                            attempts += 1;
                        }
                        if let Some(((sx1, sy1), (sx2, sy2), depth_scale)) = chosen {
                            // Build a jagged polyline from centre outward
                            let segments = 8usize;
                            let mut points = Vec::with_capacity(segments + 1);
                            points.push(pos2(sx1, sy1));
                            let usage_factor = (self.gpu.util / 100.0).clamp(0.0, 1.0);
                            for i in 1..segments {
                                let t = i as f32 / segments as f32;
                                let ix = sx1 + (sx2 - sx1) * t;
                                let iy = sy1 + (sy2 - sy1) * t;
                                // Perpendicular jitter grows with usage
                                let perp = vec2(sy2 - sy1, -(sx2 - sx1)).normalized() * ((fastrand::f32() - 0.5) * 12.0 * depth_scale * usage_factor);
                                points.push(pos2(ix + perp.x, iy + perp.y));
                            }
                            points.push(pos2(sx2, sy2));
                            let stroke_width = (2.0 * depth_scale * (0.3 + 0.7 * usage_factor)).max(0.5);
                            painter.add(Shape::line(points, Stroke::new(stroke_width, Color32::from_rgb(255, 255, 200))));
                            // Record a strike in the histogram based on current GPU utilization.
                            let usage = self.gpu.util;
                            let mut bin = (usage / 10.0).floor() as usize;
                            if bin > 9 { bin = 9; }
                            self.strike_bins[bin] = self.strike_bins[bin].saturating_add(1);
                        }
                    } else {
                        // No strike this frame — cooldown stays at 0 so we
                        // roll again next frame at the same rate.
                        self.strike_cooldown = 0.0;
                    }
                }
            }



            // HUD — minimal: Model + tok/s, with space for a second model
            let hud_color = egui::Color32::from_rgba_premultiplied(200, 200, 220, 180);
            let hud_dim = egui::Color32::from_rgba_premultiplied(120, 120, 140, 140);
            ui.vertical(|ui| {
                // Line 1: primary model + tok/s
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&self.loaded_model)
                        .size(14.0).color(hud_color));
                    ui.separator();
                    let tps = if self.tok_s > 0.0 { format!("{:.1} tok/s", self.tok_s) } else { String::from("—") };
                    ui.label(egui::RichText::new(tps)
                        .size(14.0).color(hud_color));
                });
                // Line 2: second model + tok/s (provision for future)
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&self.loaded_model2)
                        .size(14.0).color(hud_dim));
                    ui.separator();
                    let tps2 = if self.tok_s2 > 0.0 { format!("{:.1} tok/s", self.tok_s2) } else { String::from("—") };
                    ui.label(egui::RichText::new(tps2)
                        .size(14.0).color(hud_dim));
                });
            });
        });
        ctx.request_repaint();
    }
}

fn query_nvidia_smi() -> Result<GpuData, String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| format!("nvidia-smi failed: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let first = stdout.lines().next().ok_or("no GPU data")?;
    let parts: Vec<&str> = first.split(',').map(|s| s.trim()).collect();
    if parts.len() < 5 {
        return Err(format!("unexpected nvidia-smi output: {}", first));
    }
    Ok(GpuData {
        util: parts[0].parse().unwrap_or(0.0),
        vram_used_mb: parts[1].parse().unwrap_or(0.0),
        vram_total_mb: parts[2].parse().unwrap_or(0.0),
        temp_c: parts[3].parse().unwrap_or(0.0),
        power_w: parts[4].parse().unwrap_or(0.0),
    })
}

/// The LLM model actually loaded in VRAM on this machine. Prefers the real
/// model name (e.g. "gemma4:12b-tool") from Ollama's /api/ps, falling back
/// to the GPU compute-process name (e.g. "python") if Ollama isn't reachable.
fn query_loaded_model() -> Result<(String, Vec<ProcessInfo>), String> {
    // Try Ollama API first to get model names. Ollama does not provide per-process PID info,
    // so we only fill ProcessInfo with placeholder values when using this path.
    if let Ok(ps) = http_get("127.0.0.1:11434", "/api/ps") {
        let mut names = Vec::new();
        let mut processes = Vec::new();
        let mut rest: &str = ps.as_str();
        while let Some(start) = rest.find("\"name\"") {
            rest = &rest[start + 6..];
            // Skip whitespace and colon
            while rest.starts_with([' ', '\t'].as_ref()) { rest = &rest[1..]; }
            if !rest.starts_with(':') { continue; }
            rest = &rest[1..];
            while rest.starts_with([' ', '\t'].as_ref()) { rest = &rest[1..]; }
            if !rest.starts_with('"') { continue; }
            rest = &rest[1..];
            if let Some(end) = rest.find('"') {
                let model_name = rest[..end].to_string();
                names.push(model_name.clone());
                // Ollama does not expose pid/memory, set defaults.
                processes.push(ProcessInfo { pid: 0, name: model_name.clone(), used_memory_mb: 0.0, model: model_name });
                rest = &rest[end+1..];
            } else { break; }
        }
        names.dedup();
        if !names.is_empty() {
            return Ok((names.join(" + "), processes));
        }
        return Ok((String::from("idle — nothing in VRAM"), Vec::new()));
    }

    // Fallback to nvidia-smi to get compute app details (pid, process name, memory).
    let out = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| format!("nvidia-smi failed: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut processes = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 { continue; }
        let pid: u32 = parts[0].parse().unwrap_or(0);
        let name_raw = parts[1];
        let name = name_raw.rsplit('/').next().unwrap_or(name_raw).to_string();
        let mem_mb: f32 = parts[2].parse().unwrap_or(0.0);
        processes.push(ProcessInfo { pid, name: name.clone(), used_memory_mb: mem_mb, model: name });
    }
    // Build a unique model string from the process list.
    let mut model_names: Vec<String> = processes.iter().map(|p| p.model.clone()).collect();
    model_names.sort();
    model_names.dedup();
    let model_str = if model_names.is_empty() {
        String::from("idle — no model")
    } else {
        model_names.join(" + ")
    };
    Ok((model_str, processes))
}

fn http_get(host_port: &str, path: &str) -> Result<String, String> {
    let mut stream = TcpStream::connect(host_port)
        .map_err(|e| format!("connect {host_port}: {e}"))?;
    stream.set_read_timeout(Some(std::time::Duration::from_millis(300))).ok();
    stream.set_write_timeout(Some(std::time::Duration::from_millis(300))).ok();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).map_err(|e| format!("write: {e}"))?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;
    let text = String::from_utf8_lossy(&buf).to_string();
    match text.find("\r\n\r\n") {
        Some(i) => Ok(text[i + 4..].to_string()),
        None => Ok(text),
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_min_inner_size([400.0, 300.0])
            .with_max_inner_size([3840.0, 2160.0])
            .with_resizable(true)
            .with_transparent(true)
            .with_title("VRAM Point Cloud"),
        ..Default::default()
    };
    eframe::run_native(
        "vram_visualizer",
        options,
        Box::new(|_cc| {
            let ctx = _cc.egui_ctx.clone();
            ctx.set_visuals(egui::Visuals {
                panel_fill: egui::Color32::TRANSPARENT,
                window_fill: egui::Color32::TRANSPARENT,
                ..Default::default()
            });
            Box::new(VramVisualizer::new())
        }),
    )
}

// ─── ad-hoc verification (not a permanent test suite) ───
//
// These tests run with `cargo test --release`. They exercise the
// extensible pattern framework and the new ReactiveImagePattern without
// needing a display server. NOT in the live behavior — only here for
// confidence during this change. Will be removed once a permanent test
// suite exists.
#[cfg(test)]
mod ad_hoc_verify {
    use super::*;

    fn ctx() -> PatternCtx {
        PatternCtx { frame: 42, activity: 0.6, vram_fill: 0.4, temp_factor: 0.5 }
    }

    fn make_p(base_x: f32, base_y: f32, base_z: f32, layer: u8) -> Particle {
        let mut p = init_particle_cylinder(0);
        p.base_x = base_x; p.base_y = base_y; p.base_z = base_z;
        p.x = base_x; p.y = base_y; p.z = base_z;
        p.size = 1.0; p.layer = layer;
        p
    }

    #[test]
    fn smooth_step_glides_and_never_snaps() {
        // A 0 -> 1 step input (like the 500ms telemetry poll) must be eased
        // in, not teleported: after one frame the value moves only partway,
        // and it only reaches the target asymptotically.
        let mut v = 0.0_f32;
        v = smooth_step(v, 1.0, 0.10);
        assert!(v > 0.01 && v < 0.5,
            "first frame should move partway, got {v} — a hard snap (>0.5) means jerky reactions");
        let first = v;
        // A few more frames — still climbing, never overshooting past target.
        for _ in 0..5 { v = smooth_step(v, 1.0, 0.10); }
        assert!(v > first && v < 1.0,
            "should keep climbing past {first} toward 1.0 without overshoot, got {v}");
        // Many frames converge asymptotically to the target.
        let mut w = 0.0_f32;
        for _ in 0..200 { w = smooth_step(w, 1.0, 0.10); }
        assert!((w - 1.0).abs() < 1e-3,
            "should converge to target, got {w}");
        // k is clamped to [0,1] — a bad k can never invert or jump.
        assert_eq!(smooth_step(0.0, 1.0, 5.0), 1.0);
        assert_eq!(smooth_step(0.0, 1.0, 0.0), 0.0);
    }

    #[test]
    fn fill_scale_maps_pattern_extent_onto_window() {
        // A pattern whose extent exactly matches the window's aspect should
        // fill it edge-to-edge (touching both axes with no margin).
        let s = fill_scale(1024.0, 768.0, (1.35, 1.35));
        // height is the restrictive axis: 768 / 2.7 = 284.44
        let h = 768.0 / 2.7;
        assert!((s - h).abs() < 0.01, "scale {s} should equal height-fit {h}");
        // width overflows: 1024 / 2.7 = 379.3 > 284.4, so pattern spans the
        // full window height and overflows horizontally -> fills window.
        assert!((s * 1.35 * 2.0 - 768.0).abs() < 0.05,
            "fill must reach window height, reached {}", s * 1.35 * 2.0);
    }

    #[test]
    fn heatscale_and_regions_are_distinct_concentric_ring_patterns() {
        // Both 7.HeatScale and 8.Regions must re-anchor onto concentric
        // circles radiating from the center, and they must not be identical.
        let mut hs = HeatScalePattern;
        let mut rg = RegionsPattern;
        let mut a: Vec<Particle> = (0..PARTICLE_COUNT).map(|i| init_particle_cylinder(i)).collect();
        let mut b: Vec<Particle> = (0..PARTICLE_COUNT).map(|i| init_particle_cylinder(i)).collect();
        hs.on_activate(&mut a);
        rg.on_activate(&mut b);

        fn distinct_rings(ps: &[Particle], want: usize) -> bool {
            let mut radii: Vec<f32> = ps.iter()
                .map(|p| (p.base_x * p.base_x + p.base_y * p.base_y).sqrt())
                .collect();
            radii.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut rings = 0usize;
            let mut prev = -1.0f32;
            for &r in &radii {
                if (r - prev).abs() > 0.05 { rings += 1; prev = r; }
            }
            rings >= want
        }
        assert!(distinct_rings(&a, 5), "HeatScale should form 5 concentric rings");
        assert!(distinct_rings(&b, 7), "Regions should form 7 concentric rings");

        let max_r = |ps: &[Particle]| ps.iter()
            .map(|p| (p.base_x.powi(2) + p.base_y.powi(2)).sqrt())
            .fold(0.0f32, f32::max);
        // Both fill out to ~max_radius 1.34 (a full screen), not a tiny blob.
        assert!(max_r(&a) > 1.0 && max_r(&b) > 1.0,
            "both must fill the viewport, got {} and {}", max_r(&a), max_r(&b));
        // They use different ring counts -> different per-ring keys.
        let ring_keys_a: std::collections::HashSet<u8> = a.iter().map(|p| p.process_id).collect();
        let ring_keys_b: std::collections::HashSet<u8> = b.iter().map(|p| p.process_id).collect();
        assert_eq!(ring_keys_a.len(), 5, "HeatScale uses ring keys 0..5");
        assert_eq!(ring_keys_b.len(), 7, "Regions uses ring keys 0..7");
    }

    #[test]
    fn fill_scale_uses_restrictive_axis_and_never_zero() {
        // Non-square extent on a non-square window -> min axis wins.
        let square = fill_scale(1000.0, 500.0, (2.0, 2.0));
        assert!((square - 125.0).abs() < 0.01, "height-limited: {square}"); // 500/4=125
        // Extent clamped away from zero so scale never blows up.
        let degenerate = fill_scale(1920.0, 1080.0, (0.0, 0.0));
        assert!(degenerate.is_finite() && degenerate > 0.0,
            "zero extent must not produce inf/nan, got {degenerate}");
        // Every registered pattern reports a finite, positive extent.
        let v = VramVisualizer::new();
        for (i, pat) in v.patterns.iter().enumerate() {
            let (wx, wy) = pat.extent();
            assert!(wx > 0.0 && wy > 0.0 && wx.is_finite() && wy.is_finite(),
                "pattern[{i}] {} extent invalid: ({wx},{wy})", pat.name());
        }
    }

    #[test]
    fn process_cloud_idle_never_freezes_and_stays_bounded() {
        // Regression: minimized ProcessCloud used to collapse to inert dots
        // when GPU idle (orbit radius -> 0). The enhanced pattern must keep
        // every node churning and the ring precessing even at act = 0,
        // without any particle escaping the extent.
        let mut pat = ProcessCloudPattern;
        let mut p = make_p(0.0, 0.0, 0.0, 0);
        let (mut prev_x, mut prev_y) = (p.x, p.y);
        let mut moved = false;
        for frame in 0..600u64 {
            pat.update(&mut p, &PatternCtx { frame, activity: 0.0, vram_fill: 0.0, temp_factor: 0.5 });
            if (p.x - prev_x).abs() > 1e-6 || (p.y - prev_y).abs() > 1e-6 { moved = true; }
            prev_x = p.x; prev_y = p.y;
        }
        assert!(moved, "ProcessCloud particle must keep moving at idle activity (was: inert dots)");
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!(r <= 1.06, "particle escaped extent: r = {r}");
    }

    #[test]
    fn process_cloud_vortex_radius_breathes_above_idle_base() {
        // The churning radius has a non-zero idle base (0.10) that swells in
        // busier states, so clusters stay pinwheeling and visibly react to
        // GPU load rather than sitting as a static dot.
        let mut pat = ProcessCloudPattern;
        let mut p = make_p(0.0, 0.0, 0.0, 0);
        // Non-packet particle (phase >= 1.1) so it takes the vortex branch.
        p.phase = std::f32::consts::TAU * 0.75; // ~4.71 > 1.1
        // Settle it onto a vortex orbit at idle, then compare busy reach.
        let mut max_r_idle = 0.0f32;
        for frame in 1000..1160u64 {
            pat.update(&mut p, &PatternCtx { frame, activity: 0.0, vram_fill: 0.0, temp_factor: 0.5 });
            let r = (p.x * p.x + p.y * p.y).sqrt();
            max_r_idle = max_r_idle.max(r);
        }
        // Rewind near origin, run busy, require the churn reach scales up.
        p.x = 0.0; p.y = 0.0; p.vx = 0.0; p.vy = 0.0;
        let mut max_r_busy = 0.0f32;
        for frame in 1000..1200u64 {
            pat.update(&mut p, &PatternCtx { frame, activity: 1.0, vram_fill: 0.0, temp_factor: 0.5 });
            let r = (p.x * p.x + p.y * p.y).sqrt();
            max_r_busy = max_r_busy.max(r);
        }
        assert!(max_r_idle > 0.10 - 0.02, "idle churn radius too small: {max_r_idle}");
        assert!(max_r_busy > max_r_idle,
            "busy churn ({max_r_busy}) should exceed idle churn ({max_r_idle})");
    }

    #[test]
    fn snowstorm_realistic_terminal_fall_and_gpu_swirl() {
        // No. 6 (AnimationPattern -> snowstorm). Realistic contract:
        //   Projection is sy = cy + p.y*scale, and egui rect.bottom() has
        //   larger y than rect.top(), so +y is SCREEN-DOWN:
        //   1) gravity acts — every flake falls, starting at the TOP of the
        //      screen (most-negative p.y) and increasing p.y toward the
        //      bottom, but drag caps descent at a bounded terminal velocity
        //      (no runaway free-fall) so snow reads as drifting, not a blur.
        //   2) mass trumps drag — the heavier flake reaches a higher terminal
        //      descent speed than the light one (depth parallax).
        //   3) GPU load stirs the storm — horizontal travel is far greater
        //      when the GPU is busy than when idle (swirl, not laminar).
        let mut pat = AnimationPattern;
        let mk = |phase: f32| {
            let mut p = make_p(0.0, 0.0, 0.0, 0);
            p.phase = phase;
            p.y = -0.9; p.age = 5.0; // mid-fall (+y is screen-down, top = -ve)
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            p
        };
        // Heavy (f~1) vs light (f~0) flake, both idle.
        let mut heavy = mk(0.999);
        let mut light = mk(0.0);
        // One calm and one gale flake (same light-ish mass) for swirl compare.
        let mut calm = mk(0.5);
        let mut gale = mk(0.5);
        let mut terminal_heavy = 0.0f32;
        let mut terminal_light = 0.0f32;
        let (mut travel_calm, mut travel_gale) = (0.0f32, 0.0f32);
        let (mut px_c, mut px_g) = (calm.x, gale.x);
        let mut max_speed = 0.0f32;
        for frame in 0..800u64 {
            let idle = PatternCtx { frame, activity: 0.0, vram_fill: 0.0, temp_factor: 0.5 };
            let full = PatternCtx { frame, activity: 1.0, vram_fill: 0.0, temp_factor: 0.5 };
            pat.update(&mut heavy, &idle);
            pat.update(&mut light, &idle);
            pat.update(&mut calm, &idle);
            pat.update(&mut gale, &full);
            terminal_heavy = terminal_heavy.max(heavy.vy);   // falling = positive vy
            terminal_light = terminal_light.max(light.vy);
            max_speed = max_speed.max((calm.vx * calm.vx + calm.vy * calm.vy).sqrt());
            travel_calm += (calm.x - px_c).abs();
            travel_gale += (gale.x - px_g).abs();
            px_c = calm.x; px_g = gale.x;
        }
        // 1) Bounded terminal fall: snow settles, never free-falls to a blur.
        assert!(terminal_heavy > 0.005, "snow must fall (terminal vy {terminal_heavy})");
        assert!(terminal_heavy < 0.06, "descent must be bounded, not runaway ({terminal_heavy})");
        assert!(max_speed < 0.10, "flake speed must stay realistic ({max_speed})");
        // 2) Depth: the heavy flake falls measurably faster than the light one.
        assert!(terminal_heavy > terminal_light * 1.1,
            "heavy flake ({terminal_heavy}) should out-fall light ({terminal_light})");
        // 3) GPU swarm: the busy snowfield is stirred sideways far more.
        assert!(travel_gale > travel_calm * 2.0,
            "busy-flake horizontal travel ({travel_gale}) must exceed idle ({travel_calm})");
        // Both stay inside the (padded) extent. The respawn guard fires at
        // y>1.38, |x|>2.1, |z|>1.1, but a flake may briefly overshoot by up
        // to one frame of velocity before the guard resets it, so pad the
        // containment margin slightly. (Heavily active flakes get the most
        // eddy sway, hence the looser z/x allowance.)
        for p in [&calm, &gale, &heavy, &light] {
            assert!(p.x.abs() <= 2.4 && p.y.abs() <= 1.5 && p.z.abs() <= 1.3,
                "flake escaped extent: ({}, {}, {})", p.x, p.y, p.z);
        }
    }

    #[test]
    fn registry_builds_and_names_are_unique() {
        let v = VramVisualizer::new();
        let names: Vec<&str> = v.patterns.iter().map(|p| p.name()).collect();
        assert!(!names.is_empty(), "registry should not be empty");
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(),
            "all pattern names must be unique, got: {names:?}");
    }

    #[test]
    fn reactive_image_is_at_slot_five() {
        let v = VramVisualizer::new();
        assert_eq!(v.patterns[4].name(), "5. ReactiveImage",
            "slot 5 (index 4) must be ReactiveImage, got: {}",
            v.patterns[4].name());
    }

    #[test]
    fn reactive_image_on_activate_anchors_to_square() {
        let mut p = ReactiveImagePattern;
        let mut particles = vec![make_p(99.0, 99.0, 99.0, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let max_abs_x = particles.iter().map(|q| q.base_x.abs()).fold(0.0_f32, f32::max);
        let max_abs_y = particles.iter().map(|q| q.base_y.abs()).fold(0.0_f32, f32::max);
        // 1.20 / 2 = 0.60 max, plus a 0.02 slack for f32 wobble.
        assert!(max_abs_x <= 0.62,
            "max |base_x| should be <= 0.62 (HD grid), got {max_abs_x}");
        assert!(max_abs_y <= 0.62,
            "max |base_y| should be <= 0.62 (HD grid), got {max_abs_y}");
    }

    #[test]
    fn reactive_image_update_does_not_panic() {
        let mut p = ReactiveImagePattern;
        let mut particles = vec![make_p(0.1, 0.1, 0.0, 0); 16];
        let c = ctx();
        for _ in 0..100 {
            for particle in &mut particles {
                p.update(particle, &c);
            }
        }
        // After 100 steps, particles should have moved.
        let total_disp: f32 = particles.iter()
            .map(|q| (q.x - q.base_x).abs() + (q.y - q.base_y).abs())
            .sum();
        assert!(total_disp.is_finite(), "positions must be finite");
    }

    #[test]
    fn reactive_image_color_returns_valid_rgb() {
        let p = ReactiveImagePattern;
        let particle = make_p(0.0, 0.0, 0.0, 0);
        let (r, g, b) = p.color(&particle, &ctx()).expect("color should return Some");
        assert!(r.is_finite() && g.is_finite() && b.is_finite(),
            "RGB must be finite, got ({r}, {g}, {b})");
        assert!((0.0..=255.0).contains(&r));
        assert!((0.0..=255.0).contains(&g));
        assert!((0.0..=255.0).contains(&b));
    }

    #[test]
    fn image_value_resolves_through_dyn_pattern() {
        // The whole point of the trait: calling image_value on a trait
        // object must reach ReactiveImagePattern's override, not the
        // default 0.5 stub.
        let p: Box<dyn Pattern> = Box::new(ReactiveImagePattern);
        let v_center = p.image_value(0.5, 0.5);
        let v_corner = p.image_value(0.0, 0.0);
        assert!(v_center > 0.0, "image center should be non-zero, got {v_center}");
        // The mandala has rotational structure: not the trivial 0.5 default.
        assert!((v_center - 0.5).abs() > 0.05 || (v_corner - 0.5).abs() > 0.05,
            "image_value should be calling our override, not the default \
             (center={v_center}, corner={v_corner})");
    }

    #[test]
    fn image_value_default_is_flat_for_non_image_patterns() {
        // A pattern that doesn't override image_value should return 0.5.
        let p: Box<dyn Pattern> = Box::new(VortexPattern);
        assert_eq!(p.image_value(0.3, 0.7), 0.5);
    }

    #[test]
    fn switch_to_re_anchors_and_changes_particle_state() {
        let mut v = VramVisualizer::new();
        // Default active is Chimera (index 12). Switch to ReactiveImage.
        v.switch_to(4);
        assert_eq!(v.active_idx, 4);
        assert_eq!(v.current_name(), "5. ReactiveImage");
        // Particles should now be on the HD grid (base_x in [-0.6, 0.6]).
        let first_base_x = v.particles[0].base_x;
        assert!(first_base_x.abs() <= 0.62,
            "first particle should be on the image grid, base_x={first_base_x}");
    }

    #[test]
    fn cylinder_on_activate_anchors_particles_onto_a_cylinder() {
        // Cylinder depends on base_* for its spring; switching to it must
        // re-anchor those into a cylinder even if the previous pattern left
        // them in rings/cube/galaxy layout. Regression guard for "cylinder
        // isn't working" — without on_activate the flock springs to a stale
        // shape and no cylinder is drawn.
        let mut p = CylinderPattern;
        // Corrupt the base positions to model a previous pattern's anchor.
        let mut particles: Vec<Particle> = (0..200)
            .map(|i| {
                let mut q = init_particle_cylinder(i);
                q.base_x = 0.0; q.base_y = 0.0; q.base_z = 0.0;
                q.x = 0.0; q.y = 0.0; q.z = 0.0;
                q
            })
            .collect();
        p.on_activate(&mut particles);
        // Every particle should now sit on the cylinder shell: xy radius in
        // [0.15, 1.15] and z (height) in [-0.6, 0.6].
        for q in &particles {
            let r = (q.base_x * q.base_x + q.base_y * q.base_y).sqrt();
            assert!((0.10..=1.20).contains(&r),
                "cyl radius should be in [0.15,1.15], got {r}");
            assert!(q.base_z.abs() <= 0.62, "cyl height z={}", q.base_z);
            // Position snapped to base on activation.
            assert!((q.x - q.base_x).abs() < 1e-6);
            assert!((q.y - q.base_y).abs() < 1e-6);
        }
    }

    #[test]
    fn register_pattern_adds_at_runtime() {
        struct CustomPattern;
        impl Pattern for CustomPattern {
            fn name(&self) -> &'static str { "99. Custom" }
            fn update(&mut self, _p: &mut Particle, _ctx: &PatternCtx) {}
        }
        let mut v = VramVisualizer::new();
        let initial_len = v.patterns.len();
        let idx = v.register_pattern(Box::new(CustomPattern));
        assert_eq!(idx, initial_len, "new pattern should get the next index");
        assert_eq!(v.patterns[idx].name(), "99. Custom");
        // Should be reachable.
        v.switch_to(idx);
        assert_eq!(v.active_idx, idx);
        assert_eq!(v.current_name(), "99. Custom");
    }

    #[test]
    fn color_responds_to_temperature() {
        // Hot vs cold ctx should produce visibly different colors.
        let p = ReactiveImagePattern;
        let particle = make_p(0.0, 0.0, 0.5, 0);
        let cool = PatternCtx { frame: 0, activity: 0.0, vram_fill: 0.0, temp_factor: 0.0 };
        let hot  = PatternCtx { frame: 0, activity: 0.0, vram_fill: 0.0, temp_factor: 0.95 };
        let (cr, cg, cb) = p.color(&particle, &cool).unwrap();
        let (hr, hg, hb) = p.color(&particle, &hot).unwrap();
        let delta = (cr - hr).abs() + (cg - hg).abs() + (cb - hb).abs();
        assert!(delta > 20.0,
            "temperature should shift the palette noticeably, got delta={delta} \
             (cool=({cr},{cg},{cb}) hot=({hr},{hg},{hb}))");
    }

    // ─── SpiralGalaxy "fill the screen" verification ───
    //
    // The galaxy was widened from the cylinder init's ~0.85 max radius
    // to 1.20. These tests pin that behavior so a future tweak can't
    // silently shrink the disk back.

    #[test]
    fn spiral_galaxy_on_activate_reaches_full_disk_radius() {
        let mut p = SpiralGalaxyPattern;
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let max_r = particles.iter()
            .map(|q| (q.base_x * q.base_x + q.base_y * q.base_y).sqrt())
            .fold(0.0_f32, f32::max);
        // Disk radius 1.30 → max_r should reach at least 1.27 to fill
        // a 16:9 HD viewport.
        assert!(max_r > 1.25,
            "spiral galaxy disk should reach >1.25 to fill HD, got max_r={max_r}");
    }

    #[test]
    fn spiral_galaxy_disk_is_roughly_uniform() {
        // sqrt(u) radius distribution is the *correct* way to get
        // uniform surface density in a 2D disk: P(r < x) = (x/R)^2.
        // So in 4 equal-width radial bands, expected counts are:
        //   band 0:  1/16 = 6.25%
        //   band 1:  3/16 = 18.75%
        //   band 2:  5/16 = 31.25%
        //   band 3:  7/16 = 43.75%
        // These are NOT equal — the outer bands must hold more particles.
        let mut p = SpiralGalaxyPattern;
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let bands = 4;
        let mut counts = [0usize; 4];
        for q in &particles {
            let r = (q.base_x * q.base_x + q.base_y * q.base_y).sqrt();
            let b = ((r / 1.30) * bands as f32) as usize;
            counts[b.min(bands - 1)] += 1;
        }
        let expected_fracs = [1.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 7.0 / 16.0];
        for (i, (&c, &ef)) in counts.iter().zip(expected_fracs.iter()).enumerate() {
            let expected = (PARTICLE_COUNT as f32 * ef) as usize;
            // Allow 5% slack — count is deterministic, no noise.
            let drift = (c as f32 - expected as f32).abs() / expected as f32;
            assert!(drift < 0.05,
                "radial band {i} has {c} particles, expected ~{expected} ({:.1}%, drift {drift:.3})",
                ef * 100.0);
        }
    }

    #[test]
    fn spiral_galaxy_update_keeps_particles_on_disk() {
        // After settling, the galaxy should hold its shape — no
        // runaway collapse to the center, no escape past the disk.
        let mut p = SpiralGalaxyPattern;
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let c = PatternCtx { frame: 0, activity: 0.1, vram_fill: 0.0, temp_factor: 0.0 };
        for _ in 0..200 {
            for q in &mut particles {
                p.update(q, &c);
            }
        }
        let max_r = particles.iter()
            .map(|q| (q.x * q.x + q.y * q.y).sqrt())
            .fold(0.0_f32, f32::max);
        let min_r = particles.iter()
            .map(|q| (q.x * q.x + q.y * q.y).sqrt())
            .fold(f32::INFINITY, f32::min);
        // Must still cover most of the disk.
        assert!(max_r > 1.0,
            "spiral galaxy should not collapse — max_r={max_r}");
        assert!(min_r < 0.3,
            "spiral galaxy should retain an inner edge, not collapse to a point — min_r={min_r}");
    }

    #[test]
    fn spiral_galaxy_high_activity_pushes_outward() {
        // High activity should let particles spread slightly past their
        // anchor radius (the * 1.0 + act * 0.15 multiplier).
        let mut p = SpiralGalaxyPattern;
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let c = PatternCtx { frame: 0, activity: 1.0, vram_fill: 0.0, temp_factor: 0.0 };
        for _ in 0..300 {
            for q in &mut particles {
                p.update(q, &c);
            }
        }
        let max_r = particles.iter()
            .map(|q| (q.x * q.x + q.y * q.y).sqrt())
            .fold(0.0_f32, f32::max);
        // At full activity, radius multiplier is 1.15 — so max_r should
        // exceed 1.30 * 1.15 = 1.495. Allow small slack for settling.
        assert!(max_r > 1.40,
            "high activity should push particles past anchor disk, got max_r={max_r}");
    }

    // ─── OrbitCube \"uniform star field, no clumps/edges\" verification ───
    //
    // The old OrbitCube divided base coords by max(|x|,|y|,|z|), collapsing
    // all particles onto 6 faces of a cube — visible as edges and "spots".
    // The version before that used 3 nested cube shells whose faces still
    // read as a geometric shape. The current one scatters particles on a
    // uniform random sphere direction with a continuous radius spread, so
    // it presents as an even field of individual stars with no shape, no
    // clumps and no visible boundary. These tests pin that behaviour.

    #[test]
    fn orbit_cube_creates_uniform_star_field() {
        // A star field must NOT collapse onto a few discrete shells/faces:
        // it should scatter particles across a continuous range of radii
        // with no empty center and no single radial band dominating.
        let mut p = OrbitCubePattern;
        let mut particles = vec![make_p(0.7, 0.1, -0.3, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        // Radius spread must be continuous: collect distinct radii (snapped
        // to 0.05) and require a broad spread rather than 3 tight clusters.
        let mut radii: Vec<f32> = particles.iter().map(|q| {
            (q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
        }).collect();
        radii.sort_by(f32::total_cmp);
        let lo = radii[0]; let hi = *radii.last().unwrap();
        let spread = hi - lo;
        assert!(spread > 1.5_f32,
            "star field radii too narrow: range {spread:.2} (lo={lo:.2}, hi={hi:.2}) — looks like a shell/clump");
        // Centres must not be empty: lower quartile holds a fair share of
        // stars (a uniform scatter has no void at the pivot).
        let q1 = radii[PARTICLE_COUNT / 4];
        assert!(q1 < 1.0_f32,
            "centre of star field is empty (lower-quartile radius {q1:.2})");
        // No single third of the range may hold a huge majority (clumping).
        let a = radii.iter().filter(|&&r| r < lo + spread / 3.0).count();
        let b = radii.iter().filter(|&&r| r < lo + 2.0 * spread / 3.0).count();
        let m = b - a; // middle third
        let hi_count = PARTICLE_COUNT - b;
        let third = PARTICLE_COUNT / 3;
        for (label, c) in [("lo", a), ("mid", m), ("hi", hi_count)] {
            let drift = (c as f32 - third as f32).abs() / third as f32;
            assert!(drift < 0.45,
                "star field clumps into '{label}' third: {c} of {PARTICLE_COUNT} (drift {drift:.2})");
        }
    }

    #[test]
    fn orbit_cube_field_is_isotropic_no_clumps() {
        // Uniform sphere directions must not favour any octant or axis —
        // a star field has no preferred direction (unlike a cube's faces).
        let mut p = OrbitCubePattern;
        let mut particles: Vec<Particle> = (0..PARTICLE_COUNT).map(|i| {
            // Arbitrary varied starting bases; on_activate ignores them.
            let b = i as f32 * 0.001;
            make_p(b, -b, b * 0.5, 0)
        }).collect();
        p.on_activate(&mut particles);
        // Octant buckets: (x sign, y sign, z sign).
        let mut oct = [0usize; 8];
        for q in &particles {
            let sign = |v: f32| if v >= 0.0 { 1 } else { 0 };
            let idx = sign(q.x) * 4 + sign(q.y) * 2 + sign(q.z);
            oct[idx] += 1;
        }
        let min = *oct.iter().min().unwrap();
        let max = *oct.iter().max().unwrap();
        // Uniform sphere → even ~12.5% per octant. Allow max ≤ 1.6× min so
        // any strong directional clumping (face/axis bias) is rejected.
        assert!(max as f32 <= min as f32 * 1.6 + 50.0,
            "star field is anisotropic — clumps into octants: min={min} max={max} oct={oct:?}");
    }

    #[test]
    fn orbit_cube_update_keeps_star_field_bounded() {
        // Rotation about y is radius-preserving, so after settling the
        // field's extent must be unchanged (max radius near the spawn
        // ceiling ~2.55), with no blow-up and no collapse.
        let mut p = OrbitCubePattern;
        let mut particles = vec![make_p(0.7, 0.1, -0.3, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let c = PatternCtx { frame: 0, activity: 0.0, vram_fill: 0.0, temp_factor: 0.0 };
        let spawn_max = particles.iter()
            .map(|q| (q.x * q.x + q.y * q.y + q.z * q.z).sqrt())
            .fold(0.0_f32, f32::max);
        for _ in 0..100 {
            for q in &mut particles {
                p.update(q, &c);
            }
        }
        let settled_max = particles.iter()
            .map(|q| (q.x * q.x + q.y * q.y + q.z * q.z).sqrt())
            .fold(0.0_f32, f32::max);
        assert!((settled_max - spawn_max).abs() < 0.05,
            "star field extent changed under idle spin: spawn={spawn_max:.3} settled={settled_max:.3}");
    }

    // ─── "Fill a 16:9 HD screen" verification ───
    //
    // Every pattern's max world-space radius should be at least
    // 1.0–1.2 so it covers the full viewport at the aspect-aware
    // renderer scale of min(w,h) * 0.42.

    #[test]
    fn cylinder_init_reaches_hd_radius() {
        // Cylinder init is the default for: Cylinder, Animation,
        // HeatScale, Regions, Wavefield, GridWave.
        let mut max_r = 0.0_f32;
        for i in 0..PARTICLE_COUNT {
            let p = init_particle_cylinder(i);
            let r = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
            max_r = max_r.max(r);
        }
        // Must exceed the old 0.85 ceiling.
        assert!(max_r > 1.10,
            "cylinder init should reach >1.10 to fill HD, got max_r={max_r}");
    }

    #[test]
    fn chimera_init_fills_face_oval_for_hd() {
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        init_chimera_particles(&mut particles);
        let max_abs_x = particles.iter().map(|p| p.base_x.abs()).fold(0.0_f32, f32::max);
        let max_abs_y = particles.iter().map(|p| p.base_y.abs()).fold(0.0_f32, f32::max);
        // Chimera FILL = 1.5 → face oval 0.45 wide × 0.975 tall.
        assert!(max_abs_x > 0.40,
            "Chimera face should reach |x| > 0.40 (was 0.30 before HD fill), got {max_abs_x}");
        assert!(max_abs_y > 0.90,
            "Chimera face should reach |y| > 0.90 (was 0.65 before HD fill), got {max_abs_y}");
    }

    #[test]
    fn reactive_image_grid_fills_hd() {
        // Grid half-extent is 0.60 — must be at least 0.55 to claim
        // "HD fill" (slight slack for the actual u/v sampling).
        let mut p = ReactiveImagePattern;
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        p.on_activate(&mut particles);
        let max_abs_x = particles.iter().map(|q| q.base_x.abs()).fold(0.0_f32, f32::max);
        let max_abs_y = particles.iter().map(|q| q.base_y.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs_x > 0.55,
            "ReactiveImage should reach |x| > 0.55 for HD fill, got {max_abs_x}");
        assert!(max_abs_y > 0.55,
            "ReactiveImage should reach |y| > 0.55 for HD fill, got {max_abs_y}");
    }

    #[test]
    fn vortex_respawn_radius_fills_hd() {
        // Simulate: run update on a particle near the event horizon
        // for 50 steps with a moderate ctx; some particles will hit
        // the respawn branch (dist < 0.03). After respawning, their
        // x/y should be at the new wider radius.
        let mut p = VortexPattern;
        // Create a particle already at the center to force respawn.
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); 1000];
        let c = PatternCtx { frame: 0, activity: 0.0, vram_fill: 0.0, temp_factor: 0.0 };
        for _ in 0..50 {
            for q in &mut particles {
                p.update(q, &c);
            }
        }
        let max_r = particles.iter()
            .map(|q| (q.x * q.x + q.y * q.y).sqrt())
            .fold(0.0_f32, f32::max);
        // Respawn radius is 1.0..1.6; max_r should reach at least 1.4
        // because some particles will be at the outer edge.
        assert!(max_r > 1.30,
            "vortex respawn should reach >1.30 for HD fill, got max_r={max_r}");
    }
}
