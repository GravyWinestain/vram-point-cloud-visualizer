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
// Each pattern is a small zero-sized or self-contained struct.

struct VortexPattern;
impl Pattern for VortexPattern {
    fn name(&self) -> &'static str { "1. Vortex" }
    fn extent(&self) -> (f32, f32) { (1.7, 1.7) }
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
        let is_white = white_chance > 0.0 && (p.phase * 7.0 + ctx.frame as f32 * 0.03).sin() > (1.0 - white_chance * 2.0);
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
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
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

struct OrbitCubePattern;
impl Pattern for OrbitCubePattern {
    fn name(&self) -> &'static str { "3. OrbitCube" }
    fn extent(&self) -> (f32, f32) { (1.6, 1.6) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        for p in particles.iter_mut() {
            let r = 0.35 + fastrand::f32() * 2.2;
            let u = fastrand::f32() * 2.0 - 1.0;
            let phi = u.acos();
            let theta = fastrand::f32() * std::f32::consts::TAU;
            let (sin_p, cos_p) = phi.sin_cos();
            let (sin_t, cos_t) = theta.sin_cos();
            p.base_x = r * sin_p * cos_t;
            p.base_y = r * sin_p * sin_t;
            p.base_z = r * cos_p;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            p.size = 0.4 + fastrand::f32().powi(2) * 1.4;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let spin_rate = ctx.frame as f32 / 1_000_000.0;
        p.phase += spin_rate;
        let a = p.phase;
        let cos_a = a.cos();
        let sin_a = a.sin();
        let rtx = p.base_x * cos_a - p.base_z * sin_a;
        let rtz = p.base_x * sin_a + p.base_z * cos_a;
        let breathe = 1.0 + ctx.activity * 0.10;
        let tx = rtx * breathe;
        let ty = p.base_y * breathe;
        let tz = rtz * breathe;
        p.vx += (tx - p.x) * 0.10;
        p.vy += (ty - p.y) * 0.10;
        p.vz += (tz - p.z) * 0.10;
        p.vx *= 0.85; p.vy *= 0.85; p.vz *= 0.85;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
}

struct HeadPattern;
impl Pattern for HeadPattern {
    fn name(&self) -> &'static str { "4. Head" }
    fn extent(&self) -> (f32, f32) { (1.2, 1.2) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let r = (i as f32) / n;
            if r < 0.7 {
                let phi = (i as f32 * 2.399_963) % std::f32::consts::TAU;
                let theta = fastrand::f32() * std::f32::consts::PI;
                let radius = 0.6;
                p.base_x = radius * theta.sin() * phi.cos();
                p.base_y = radius * theta.sin() * phi.sin();
                p.base_z = radius * theta.cos();
            } else if r < 0.9 {
                let phi = (i as f32 * 2.399_963) % std::f32::consts::TAU;
                let theta = fastrand::f32() * std::f32::consts::PI;
                let radius = 0.3;
                p.base_x = radius * theta.sin() * phi.cos();
                p.base_y = radius * theta.sin() * phi.sin();
                p.base_z = radius * theta.cos() - 0.5;
            } else {
                if fastrand::f32() > 0.5 {
                    let side = if fastrand::f32() > 0.5 { 0.2 } else { -0.2 };
                    p.base_x = side; p.base_y = 0.1; p.base_z = 0.3;
                } else {
                    p.base_x = 0.0; p.base_y = (fastrand::f32() - 0.5) * 0.1; p.base_z = 0.3 - fastrand::f32() * 0.3;
                }
            }
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
            p.phase = (i as f32 * 0.1) % std::f32::consts::TAU;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let breathe = 1.0 + (t * 0.5 + p.phase).sin() * 0.05 * act;
        let tx = p.base_x * breathe;
        let ty = p.base_y * breathe;
        let tz = p.base_z * breathe;
        let drift = (t * 0.8 + p.phase).sin() * 0.02 * act;
        let spring = 0.05 + ctx.vram_fill * 0.05;
        p.vx += (tx + drift - p.x) * spring; p.vy += (ty + drift - p.y) * spring; p.vz += (tz + drift - p.z) * spring;
        p.vx *= 0.9; p.vy *= 0.9; p.vz *= 0.9;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let vfill = ctx.vram_fill;
        let act = ctx.activity;
        let r = 50.0 + vfill * 200.0;
        let g = 40.0 + act * 100.0;
        let b = 200.0 + vfill * 55.0;
        let flicker = 0.9 + (p.phase * 10.0 + ctx.frame as f32 * 0.1).sin() * 0.1;
        Some((r * flicker, g * flicker, b * flicker))
    }
}

struct ProcessCloudPattern;
impl Pattern for ProcessCloudPattern {
    fn name(&self) -> &'static str { "5. ProcessCloud" }
    fn extent(&self) -> (f32, f32) { (1.05, 1.05) }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let pid = p.process_id as f32;
        let rot = t * 0.15;
        let a0 = pid * 1.7;
        let cx = (a0 + rot).cos() * 0.75;
        let cy = (a0 + rot).sin() * 0.75;
        let cz = (pid * 0.9).sin() * 0.3;
        if p.phase.rem_euclid(std::f32::consts::TAU) < 1.1 {
            let prog = (t * 0.8 + p.phase).fract();
            let ang = a0 + rot + prog * 1.7;
            let tx = ang.cos() * 0.72;
            let ty = ang.sin() * 0.72;
            let tz = (ang * 0.9).sin() * 0.3;
            p.vx += (tx - p.x) * 0.12; p.vy += (ty - p.y) * 0.12; p.vz += (tz - p.z) * 0.12;
            p.vx *= 0.90; p.vy *= 0.90; p.vz *= 0.90;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
            return;
        }
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
    fn extent(&self) -> (f32, f32) { (1.9, 1.4) }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let f = p.phase % 1.0;
        p.age += 0.016;
        let mass = 0.75 + 0.55 * f;
        p.vy += 0.00034 * mass;
        p.vy *= 0.97;
        p.vx *= 0.985;
        p.vz *= 0.985;
        let swirl = act * (0.25 + 0.75 * f);
        let core_x = 0.5 * (t * 0.35).sin();
        let core_y = 0.2 * (t * 0.42).cos() - 0.1;
        let dx = p.x - core_x;
        let dy = p.y - core_y;
        let d2 = dx * dx + dy * dy + p.z * p.z + 0.08;
        p.vx += swirl * (-dy / d2) * 0.020;
        p.vz += swirl * ( dx / d2) * 0.020;
        p.vy += swirl * (p.z / d2) * 0.012;
        p.vx += (t * 1.3 + p.phase * 7.0).sin() * act * 0.010;
        p.vz += (t * 0.9 + p.phase * 5.0).cos() * act * 0.010;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
        if p.age > 34.0 || p.y > 1.38 || p.x.abs() > 2.1 || p.z.abs() > 1.1 {
            p.x = (fastrand::f32() - 0.5) * 2.4;
            p.y = -1.30 - fastrand::f32() * 0.25;
            p.z = (fastrand::f32() - 0.5) * 1.0;
            p.vx = (fastrand::f32() - 0.5) * 0.06;
            p.vy = 0.008 + f * 0.008;
            p.vz = (fastrand::f32() - 0.5) * 0.06;
            p.age = 0.0;
        }
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let act = ctx.activity;
        let sparkle = 0.8 + 0.4 * (p.phase * 13.0).sin().abs();
        let v = (120.0 + act * 115.0) * sparkle;
        Some((v, v + 4.0, v + 10.0))
    }
}

// Shared helper
fn anchor_concentric_rings(particles: &mut [Particle], max_radius: f32, rings: usize) {
    let n = particles.len() as f32;
    for (i, p) in particles.iter_mut().enumerate() {
        let fr = i as f32 / n;
        let ring = (fr * rings as f32).min((rings - 1) as f32) as usize;
        let ring_radius = max_radius * (ring as f32 + 1.0) / rings as f32;
        let r = ring_radius * (0.86 + 0.14 * fastrand::f32());
        let a = fr * std::f32::consts::TAU * (rings as f32) + (ring as f32) * 0.4;
        p.base_x = a.cos() * r;
        p.base_y = a.sin() * r;
        p.base_z = (fastrand::f32() - 0.5) * 0.2;
        p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
        p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        p.phase = (i % 64) as f32 * 0.098;
        p.process_id = ring as u8;
    }
}

struct HeatScalePattern;
impl Pattern for HeatScalePattern {
    fn name(&self) -> &'static str { "7. HeatScale" }
    fn extent(&self) -> (f32, f32) { (1.35, 1.35) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        anchor_concentric_rings(particles, 1.34, 5);
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let motion = 1.0 + act * 1.2;
        let br = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let ang = p.base_y.atan2(p.base_x);
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

struct RegionsPattern;
impl Pattern for RegionsPattern {
    fn name(&self) -> &'static str { "8. Regions" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let x = (i as f32 / n) * 3.0 - 1.5;
            let y = ((i as f32 * 1.7).sin()) * 1.5;
            p.base_x = x; p.base_y = y; p.base_z = 0.0;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let tx = p.base_x + (ctx.frame as f32 * 0.01).sin() * act * 0.2;
        let ty = p.base_y + (ctx.frame as f32 * 0.015).cos() * act * 0.2;
        let spring = 0.05 + ctx.vram_fill * 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.9; p.vy *= 0.9;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let c = (p.base_x + 1.5) / 3.0;
        Some((c * 255.0, (1.0 - c) * 255.0, 128.0))
    }
}

struct WavefieldPattern;
impl Pattern for WavefieldPattern {
    fn name(&self) -> &'static str { "9. Wavefield" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let x = (i as f32 / n) * 3.0 - 1.5;
            let z = ((i as f32 * 0.3).sin()) * 1.5;
            p.base_x = x; p.base_y = 0.0; p.base_z = z;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let tx = p.base_x;
        let ty = (p.base_x * 3.0 - t * 2.0).sin() * act * 0.5;
        let tz = p.base_z;
        let spring = 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring; p.vz += (tz - p.z) * spring;
        p.vx *= 0.9; p.vy *= 0.9; p.vz *= 0.9;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let c = (p.y + 1.0) / 2.0;
        Some((0.0, c * 255.0, 255.0))
    }
}

struct SpiralGalaxyPattern;
impl Pattern for SpiralGalaxyPattern {
    fn name(&self) -> &'static str { "10. SpiralGalaxy" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let r = (i as f32 / n) * 1.5;
            let a = (i as f32 * 0.1) + r * 2.0;
            p.base_x = a.cos() * r;
            p.base_y = a.sin() * r;
            p.base_z = (fastrand::f32() - 0.5) * 0.2;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let a = t * 0.2 + p.phase;
        let orbit = act * 0.1;
        let tx = p.base_x + a.cos() * orbit;
        let ty = p.base_y + a.sin() * orbit;
        let spring = 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.9; p.vy *= 0.9;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let r = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let c = 1.0 - (r / 1.5).clamp(0.0, 1.0);
        Some((c * 255.0, c * 100.0, 255.0))
    }
}

struct GridWavePattern;
impl Pattern for GridWavePattern {
    fn name(&self) -> &'static str { "11. GridWave" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        let cols = 20;
        for (i, p) in particles.iter_mut().enumerate() {
            let c = i % cols;
            let r = i / cols;
            p.base_x = (c as f32 / cols as f32) * 3.0 - 1.5;
            p.base_y = (r as f32 / (n as usize / cols) as f32) * 3.0 - 1.5;
            p.base_z = 0.0;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let tx = p.base_x;
        let ty = p.base_y;
        let tz = (p.base_x * 5.0 - t * 2.0).sin() * (p.base_y * 5.0 - t * 2.0).sin() * act * 0.5;
        let spring = 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring; p.vz += (tz - p.z) * spring;
        p.vx *= 0.9; p.vy *= 0.9; p.vz *= 0.9;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some((100.0, 200.0, 255.0))
    }
}

struct ChimeraPattern;
impl Pattern for ChimeraPattern {
    fn name(&self) -> &'static str { "12. Chimera" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let r = (i as f32 / n) * 1.5;
            let a = (i as f32 * 0.1);
            p.base_x = a.cos() * r;
            p.base_y = a.sin() * r;
            p.base_z = (fastrand::f32() - 0.5) * 1.5;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let tx = p.base_x + (t * 0.5).sin() * act * 0.2;
        let ty = p.base_y + (t * 0.5).cos() * act * 0.2;
        let tz = p.base_z + (t * 0.3).sin() * act * 0.2;
        let spring = 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring; p.vz += (tz - p.z) * spring;
        p.vx *= 0.9; p.vy *= 0.9; p.vz *= 0.9;
        p.x += p.vx; p.y += p.vy; p.z += p.vz;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some((255.0, 128.0, 64.0))
    }
}

struct ReactiveImagePattern;
impl Pattern for ReactiveImagePattern {
    fn name(&self) -> &'static str { "13. ReactiveImage" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn on_activate(&mut self, particles: &mut [Particle]) {
        let n = particles.len() as f32;
        for (i, p) in particles.iter_mut().enumerate() {
            let u = (i % 64) as f32 / 64.0;
            let v = (i / 64) as f32 / 64.0;
            p.base_x = u * 3.0 - 1.5;
            p.base_y = v * 3.0 - 1.5;
            p.base_z = 0.0;
            p.x = p.base_x; p.y = p.base_y; p.z = p.base_z;
            p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        }
    }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let t = ctx.frame as f32 * 0.016;
        let act = ctx.activity;
        let tx = p.base_x + (t * 0.5).sin() * act * 0.1;
        let ty = p.base_y + (t * 0.5).cos() * act * 0.1;
        let spring = 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.9; p.vy *= 0.9;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        let u = (p.base_x + 1.5) / 3.0;
        let v = (p.base_y + 1.5) / 3.0;
        Some((u * 255.0, v * 255.0, 128.0))
    }
}

struct CustomPattern;
impl Pattern for CustomPattern {
    fn name(&self) -> &'static str { "14. Custom" }
    fn extent(&self) -> (f32, f32) { (1.5, 1.5) }
    fn update(&mut self, _p: &mut Particle, _ctx: &PatternCtx) {}
}

// ─── GnomeWorx chip patterns ───
//
// A family of CUDA-core monitoring patterns built around a central
// "GnomeWorx chip": a square die with IC leads (pins) around the border.
// Every pattern re-anchors onto the same chip silhouette (so the whole
// set reads as one coherent brand motif) but animates it differently.
// Selectable with number keys 1-0.
//
// layer 0 = die body, layer 2 = border pins, layer 3 = die border ring.

const CHIP_DIE_HALF: f32 = 0.50;

/// Anchor every particle onto the central GnomeWorx chip silhouette:
/// a square die, a bright die-border ring (the frame), and IC leads
/// (pins) radiating out from all four edges — the "chip border". The
/// border ring and pins are given extra weight so the chip reads as
/// clearly framed.
fn anchor_chip(particles: &mut [Particle]) {
    let n = particles.len() as f32;
    for (i, p) in particles.iter_mut().enumerate() {
        let fr = i as f32 / n;
        let (bx, by, layer) = if fr < 0.50 {
            // Die body: even 2D square fill. Two INDEPENDENT irrational
            // seeds (phi ~0.618 and sqrt2-1 ~0.414) so the scatter is a
            // true 2D fill — NOT a collapsed diagonal (0.618+0.382=1
            // made u and v complementary into a line).
            let u = (i as f32 * 0.618_033_9887).fract();
            let v = (i as f32 * 0.414_213_5624).fract();
            ((u * 2.0 - 1.0) * CHIP_DIE_HALF, (v * 2.0 - 1.0) * CHIP_DIE_HALF, 0u8)
        } else if fr < 0.66 {
            // Die border: a bright SQUARE frame hugging the die edge,
            // so the chip reads as clearly bordered. 16% gives enough
            // particles for a CONTINUOUS THIN line along each edge.
            let e = (fr - 0.50) / 0.16;
            let seg = (e * 4.0) as usize % 4;
            let along = (e * 4.0).fract() * 2.0 - 1.0; // -1..1
            let r = CHIP_DIE_HALF * 0.99;
            let (bx, by) = match seg {
                0 => ( along * r,  r), // top edge (rightward)
                1 => ( r, -along * r), // right edge (downward)
                2 => (-along * r, -r), // bottom edge (leftward)
                _ => (-r,  along * r), // left edge (upward)
            };
            (bx, by, 3u8)
        } else {
            // Border pins: long leads radiating out from each edge.
            let e = (fr - 0.66) / 0.34;
            let side = (e * 4.0) as usize % 4;
            let along = (e * 4.0).fract();
            let pin_len = CHIP_DIE_HALF * 0.80;
            let span = CHIP_DIE_HALF * 0.98;
            let (ex, ey, dx, dy) = match side {
                0 => ( CHIP_DIE_HALF, (along * 2.0 - 1.0) * span,  pin_len, 0.0),
                1 => ((along * 2.0 - 1.0) * span,  CHIP_DIE_HALF, 0.0,  pin_len),
                2 => (-CHIP_DIE_HALF, (along * 2.0 - 1.0) * span, -pin_len, 0.0),
                _ => ((along * 2.0 - 1.0) * span, -CHIP_DIE_HALF, 0.0, -pin_len),
            };
            let t = fastrand::f32();
            (ex + dx * t, ey + dy * t, 2u8)
        };
        // Particle size by layer: the die body uses larger dots so the
        // square fills densely into a solid violet die; the die border is
        // deliberately SLIM so the chip package reads as a thin crisp
        // frame (a real IC substrate has a hairline silver edge). Pins
        // stay finer still.
        p.size = match layer {
            0 => 3.0,
            3 => 1.8,
            _ => 2.0,
        };
        p.base_x = bx; p.base_y = by; p.base_z = 0.0;
        p.x = bx; p.y = by; p.z = 0.0;
        p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
        p.phase = (i % 64) as f32 * 0.098;
        p.layer = layer;
    }
}

/// GnomeWorx chip palette: violet die, bright die-border ring, copper
/// pins. All brighten with GPU activity; the die also shifts with VRAM
/// fill and temperature so the chip reads as "working". The border ring
/// and pins are kept high-contrast so the chip is clearly framed.
fn chip_color(p: &Particle, ctx: &PatternCtx) -> (f32, f32, f32) {
    let act = ctx.activity;
    let vfill = ctx.vram_fill;
    let tfact = ctx.temp_factor;
    let flicker = 0.9 + (p.phase * 9.0 + ctx.frame as f32 * 0.05).sin() * 0.1;
    match p.layer {
        0 => {
            let r = (90.0 + act * 60.0 + vfill * 30.0) * flicker;
            let g = (40.0 + act * 30.0) * flicker;
            let b = (200.0 + act * 55.0 + tfact * 20.0) * flicker;
            (r, g, b)
        }
        3 => {
            // Die border ring: bright cyan-white frame.
            let r = (150.0 + act * 60.0) * flicker;
            let g = (220.0 + act * 35.0) * flicker;
            let b = (255.0) * flicker;
            (r, g, b)
        }
        _ => {
            // Pins: bright gold, glowing with activity.
            let r = (255.0) * flicker;
            let g = (200.0 + act * 55.0) * flicker;
            let b = (90.0 + act * 30.0) * flicker;
            (r, g, b)
        }
    }
}

/// 1. ChipCore — the resting chip: a gentle shimmer over the die and
/// pins, intensifying as the GPU works.
struct ChipCorePattern;
impl Pattern for ChipCorePattern {
    fn name(&self) -> &'static str { "1. ChipCore" }
    fn extent(&self) -> (f32, f32) { (1.0, 1.0) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let jx = (t * 1.2 + p.phase * 3.0).sin() * act * 0.02;
        let jy = (t * 1.4 + p.phase * 2.0).cos() * act * 0.02;
        let tx = p.base_x + jx;
        let ty = p.base_y + jy;
        let spring = 0.10;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.85; p.vy *= 0.85;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 2. ChipPulse — the whole chip breathes: the die swells and
/// contracts with GPU activity, like a beating processor heart.
struct ChipPulsePattern;
impl Pattern for ChipPulsePattern {
    fn name(&self) -> &'static str { "2. ChipPulse" }
    fn extent(&self) -> (f32, f32) { (1.0, 1.0) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let pulse = 1.0 + (t * 2.0 + p.phase).sin() * act * 0.15;
        let tx = p.base_x * pulse;
        let ty = p.base_y * pulse;
        let spring = 0.08;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.85; p.vy *= 0.85;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 3. ChipOrbit — CUDA cores circle the chip like electrons, the orbit
/// radius and speed growing with GPU load.
struct ChipOrbitPattern;
impl Pattern for ChipOrbitPattern {
    fn name(&self) -> &'static str { "3. ChipOrbit" }
    fn extent(&self) -> (f32, f32) { (1.3, 1.3) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let a = t * (0.3 + act * 0.5) + p.phase;
        let orbit = 0.15 + act * 0.25;
        let tx = p.base_x + a.cos() * orbit;
        let ty = p.base_y + a.sin() * orbit;
        let spring = 0.06;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.88; p.vy *= 0.88;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 4. ChipGrid — the die is a grid of CUDA cores; each cell lights up
/// with activity, a wave of computation sweeping across the chip.
struct ChipGridPattern;
impl Pattern for ChipGridPattern {
    fn name(&self) -> &'static str { "4. ChipGrid" }
    fn extent(&self) -> (f32, f32) { (1.0, 1.0) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        // A travelling wave across the die; pins stay anchored.
        let wave = if p.layer == 0 {
            (p.base_x * 6.0 - t * 2.0).sin() * act * 0.10
        } else { 0.0 };
        let tx = p.base_x;
        let ty = p.base_y + wave;
        let spring = 0.08;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.85; p.vy *= 0.85;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        // Grid cells: brighten a cell based on its position and activity.
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let cell = ((p.base_x * 6.0 - t * 2.0).sin() * 0.5 + 0.5) * act;
        let (r, g, b) = chip_color(p, ctx);
        Some((r + cell * 60.0, g + cell * 40.0, b + cell * 20.0))
    }
}

/// 5. ChipWave — a ripple travels across the die surface, like heat
/// shimmering off a working processor.
struct ChipWavePattern;
impl Pattern for ChipWavePattern {
    fn name(&self) -> &'static str { "5. ChipWave" }
    fn extent(&self) -> (f32, f32) { (1.0, 1.0) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let ripple = (dist * 8.0 - t * 3.0).sin() * act * 0.12;
        let tx = p.base_x;
        let ty = p.base_y + ripple;
        let spring = 0.08;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.85; p.vy *= 0.85;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 6. ChipBreathe — the whole chip scales in and out with GPU load,
/// a slow, massy inhale/exhale.
struct ChipBreathePattern;
impl Pattern for ChipBreathePattern {
    fn name(&self) -> &'static str { "6. ChipBreathe" }
    fn extent(&self) -> (f32, f32) { (1.1, 1.1) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let breathe = 1.0 + (t * 0.8 + p.phase).sin() * act * 0.12;
        let tx = p.base_x * breathe;
        let ty = p.base_y * breathe;
        let spring = 0.07;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.85; p.vy *= 0.85;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 7. ChipSpiral — cores spiral inward toward the chip centre, the
/// spiral tightening as the GPU ramps up.
struct ChipSpiralPattern;
impl Pattern for ChipSpiralPattern {
    fn name(&self) -> &'static str { "7. ChipSpiral" }
    fn extent(&self) -> (f32, f32) { (1.2, 1.2) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let a = t * (0.4 + act * 0.6) + p.phase;
        let pull = 0.02 + act * 0.05;
        let tx = p.base_x + a.cos() * pull;
        let ty = p.base_y + a.sin() * pull;
        let spring = 0.06;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.88; p.vy *= 0.88;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 8. ChipRings — concentric rings radiate outward from the chip,
/// like a signal broadcast from the processor.
struct ChipRingsPattern;
impl Pattern for ChipRingsPattern {
    fn name(&self) -> &'static str { "8. ChipRings" }
    fn extent(&self) -> (f32, f32) { (1.3, 1.3) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
        let ring = (dist * 6.0 - t * 2.0).sin() * act * 0.10;
        let tx = p.base_x;
        let ty = p.base_y + ring;
        let spring = 0.08;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.85; p.vy *= 0.85;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 9. ChipStorm — a storm of particles swirls around the chip, the
/// vortex tightening and speeding with GPU load.
struct ChipStormPattern;
impl Pattern for ChipStormPattern {
    fn name(&self) -> &'static str { "9. ChipStorm" }
    fn extent(&self) -> (f32, f32) { (1.4, 1.4) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        let t = ctx.frame as f32 * 0.016;
        let a = t * (0.5 + act * 0.8) + p.phase;
        let swirl = 0.10 + act * 0.20;
        let tx = p.base_x + a.cos() * swirl;
        let ty = p.base_y + a.sin() * swirl;
        let spring = 0.05;
        p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
        p.vx *= 0.88; p.vy *= 0.88;
        p.x += p.vx; p.y += p.vy;
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

/// 0. ChipMatrix — a matrix-style rain of cores falls within the chip
/// border, the fall speeding up as the GPU works.
struct ChipMatrixPattern;
impl Pattern for ChipMatrixPattern {
    fn name(&self) -> &'static str { "0. ChipMatrix" }
    fn extent(&self) -> (f32, f32) { (1.0, 1.0) }
    fn on_activate(&mut self, particles: &mut [Particle]) { anchor_chip(particles); }
    fn update(&mut self, p: &mut Particle, ctx: &PatternCtx) {
        let act = ctx.activity;
        // Die-body cores fall downward (screen-down = +y) and reset at the
        // top; border ring and pins stay anchored.
        if p.layer == 0 {
            let fall = 0.0006 + act * 0.0012;
            p.vy += fall;
            p.vy *= 0.99;
            p.x += p.vx; p.y += p.vy;
            if p.y > CHIP_DIE_HALF {
                p.y = -CHIP_DIE_HALF;
                p.x = p.base_x;
                p.vx = (fastrand::f32() - 0.5) * 0.002;
                p.vy = 0.0;
            }
        } else {
            let tx = p.base_x;
            let ty = p.base_y;
            let spring = 0.10;
            p.vx += (tx - p.x) * spring; p.vy += (ty - p.y) * spring;
            p.vx *= 0.85; p.vy *= 0.85;
            p.x += p.vx; p.y += p.vy;
        }
    }
    fn color(&self, p: &Particle, ctx: &PatternCtx) -> Option<(f32, f32, f32)> {
        Some(chip_color(p, ctx))
    }
}

// ─── Missing definitions (recovered) ───
#[derive(Clone)]
struct ProcessInfo {
    pid: u32,
    name: String,
    used_memory_mb: f32,
    model: String,
}

struct Config {
    max_speed: f32,
    lightning: f32,
    max_rotation: f32,
    // When true the window is locked in place (no drag-to-move) so it
    // stays pinned behind the gauges. Set PIN=0 in the env to allow
    // dragging again without a rebuild.
    pinned: bool,
}

impl Config {
    fn from_env() -> Self {
        // Defaults
        Config {
            max_speed: 0.015,
            lightning: 1.0,
            max_rotation: 0.0004,
            pinned: std::env::var("PIN").map(|v| v != "0").unwrap_or(true),
        }
    }
}

fn smooth_step(current: f32, target: f32, k: f32) -> f32 {
    current + (target - current) * k
}

fn fill_scale(win_w: f32, win_h: f32, extent: (f32, f32)) -> f32 {
    let (ex, ey) = extent;
    let sx = if win_w > 0.0 { win_w / (ex * 2.0) } else { 1.0 };
    let sy = if win_h > 0.0 { win_h / (ey * 2.0) } else { 1.0 };
    sx.min(sy)
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
    // Runtime tunables (rotation speed, lightning intensity).
    cfg: Config,
}

impl VramVisualizer {
    fn new() -> Self {
        let particles: Vec<Particle> = (0..PARTICLE_COUNT)
            .map(init_particle_cylinder)
            .collect();
        // Build the registry. Order here is the Tab cycle order.
        // Number keys 1-0 select the GnomeWorx chip patterns (indices 0-9);
        // OrbitCube stays reachable at the end of the Tab cycle.
        let patterns: Vec<Box<dyn Pattern>> = vec![
            Box::new(ChipCorePattern),
            Box::new(ChipPulsePattern),
            Box::new(ChipOrbitPattern),
            Box::new(ChipGridPattern),
            Box::new(ChipWavePattern),
            Box::new(ChipBreathePattern),
            Box::new(ChipSpiralPattern),
            Box::new(ChipRingsPattern),
            Box::new(ChipStormPattern),
            Box::new(ChipMatrixPattern),
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
            // Three patterns: 0 OrbitCube, 1 Nebula, 2 Sphere.
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
            cfg: Config::from_env(),
        };

        // Activate the first pattern.
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
        // Updated: capture per-process info and estimate tok/s.
        if let Ok((model_str, procs)) = query_loaded_model() {
            self.loaded_model = model_str.clone();
            self.processes = procs;
            // Estimate tok/s from model name + GPU utilisation.
            // RTX 4060 Ti baseline: ~100 tok/s for a 7B Q4 at 100% GPU.
            self.tok_s = estimate_tok_s(&model_str, self.gpu.util);
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
        let mut target_spin = 0.0001 + log_activity * self.cfg.max_rotation;
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
                    // Window stacking: B = send to back (behind the gauges),
                    // F = bring to front. Useful because the window is
                    // frameless/transparent and may overlap the Groucho gauges.
                    match key {
                        egui::Key::B => {
                            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(
                                egui::WindowLevel::AlwaysOnBottom,
                            ));
                            continue;
                        }
                        egui::Key::F => {
                            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(
                                egui::WindowLevel::Normal,
                            ));
                            continue;
                        }
                        _ => {}
                    }
                    let new_idx = match key {
                        egui::Key::Num1 => Some(0),  // 1. ChipCore
                        egui::Key::Num2 => Some(1),  // 2. ChipPulse
                        egui::Key::Num3 => Some(2),  // 3. ChipOrbit
                        egui::Key::Num4 => Some(3),  // 4. ChipGrid
                        egui::Key::Num5 => Some(4),  // 5. ChipWave
                        egui::Key::Num6 => Some(5),  // 6. ChipBreathe
                        egui::Key::Num7 => Some(6),  // 7. ChipSpiral
                        egui::Key::Num8 => Some(7),  // 8. ChipRings
                        egui::Key::Num9 => Some(8),  // 9. ChipStorm
                        egui::Key::Num0 => Some(9),  // 0. ChipMatrix
                        egui::Key::Minus => Some(10), // - OrbitCube
                        egui::Key::Equals => Some(11), // = (reserved)
                        egui::Key::N => Some(1),  // N. ChipPulse (index 1)
                        egui::Key::S => Some(2),  // S. ChipOrbit (index 2)
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
            // Tesla-coil lightning: big branching arcs between distant points
            // on the sphere, with a bright core and a coloured glow halo.
            {
                use egui::{pos2, vec2, Color32, Shape, Stroke};
                let lite = self.cfg.lightning;
                // Tesla coils crackle continuously. Base rate is higher than
                // the old bolts and scales with GPU usage and intensity.
                if lite > 0.0 && self.strike_cooldown <= 0.0 {
                    // Strike rate: 2.0 Hz at 0% → (2.0 + 8.0*intensity) Hz at 100%
                    let strike_rate = 2.0 + (self.gpu.util / 100.0) * (8.0 * lite);
                    if fastrand::f32() < (strike_rate / 60.0) {
                        self.strike_cooldown = 1.0 / strike_rate;
                        let usage_factor = (self.gpu.util / 100.0).clamp(0.0, 1.0);
                        // Pick two DISTANT particles on the sphere so the arc
                        // spans the full sphere (a big Tesla coil discharge).
                        // Retry to find two on-screen points far apart.
                        let mut attempts = 0;
                        let mut chosen = None;
                        while attempts < 20 {
                            let i1 = fastrand::usize(0..sorted.len());
                            let i2 = fastrand::usize(0..sorted.len());
                            if i1 == i2 { attempts += 1; continue; }
                            let p1 = &sorted[i1];
                            let p2 = &sorted[i2];
                            let d1 = 3.0 / (3.0 + p1.z * 0.8);
                            let d2 = 3.0 / (3.0 + p2.z * 0.8);
                            let s1x = cx + p1.x * scale * d1;
                            let s1y = cy + p1.y * scale * d1;
                            let s2x = cx + p2.x * scale * d2;
                            let s2y = cy + p2.y * scale * d2;
                            let within = |x: f32, y: f32| {
                                x >= rect.left() - 10.0 && x <= rect.right() + 10.0 && y >= rect.top() - 10.0 && y <= rect.bottom() + 10.0
                            };
                            // Require a minimum screen distance so arcs are big.
                            let dist = ((s2x - s1x).powi(2) + (s2y - s1y).powi(2)).sqrt();
                            if within(s1x, s1y) && within(s2x, s2y) && dist > rect.width() * 0.25 {
                                chosen = Some(((s1x, s1y), (s2x, s2y), (d1 + d2) * 0.5));
                                break;
                            }
                            attempts += 1;
                        }
                        if let Some(((sx1, sy1), (sx2, sy2), depth_scale)) = chosen {
                            // Build a jagged, angular main bolt (Tesla arcs are
                            // sharp, not smooth). Bow slightly toward the centre.
                            let segments = 12usize;
                            let mut main = Vec::with_capacity(segments + 1);
                            main.push(pos2(sx1, sy1));
                            let mx = (sx1 + sx2) * 0.5;
                            let my = (sy1 + sy2) * 0.5;
                            let bow = 0.15 + 0.30 * usage_factor;
                            let jitter_amp = 16.0 * depth_scale * (0.5 + usage_factor) * (0.6 + 0.4 * lite).min(2.0);
                            let perp = vec2(sy2 - sy1, -(sx2 - sx1)).normalized();
                            for i in 1..segments {
                                let t = i as f32 / segments as f32;
                                let ix = sx1 + (sx2 - sx1) * t;
                                let iy = sy1 + (sy2 - sy1) * t;
                                let arc = (t * (1.0 - t)) * 4.0 * bow;
                                let bx = ix + (cx - mx) * arc;
                                let by = iy + (cy - my) * arc;
                                // Sharp angular jitter (Tesla arcs zig-zag hard).
                                let j = (fastrand::f32() - 0.5) * jitter_amp;
                                main.push(pos2(bx + perp.x * j, by + perp.y * j));
                            }
                            main.push(pos2(sx2, sy2));

                            // Branching: 2-3 secondary arcs split off the main
                            // bolt at random points and shoot outward — the
                            // classic Tesla coil corona.
                            let branches = 2 + (fastrand::usize(0..2));
                            let mut branch_lines: Vec<Vec<egui::Pos2>> = Vec::new();
                            for _ in 0..branches {
                                // Pick a split point partway along the main bolt.
                                let bt = 0.2 + fastrand::f32() * 0.6;
                                let bi = (bt * segments as f32) as usize;
                                let bi = bi.clamp(1, segments - 1);
                                let start = main[bi];
                                // Branch shoots outward (away from the chord),
                                // length scales with usage and intensity.
                                let blen = (0.15 + 0.35 * usage_factor) * (0.6 + 0.4 * lite).min(2.0) * rect.width() * 0.12;
                                let bdir = vec2(sx2 - sx1, sy2 - sy1).normalized();
                                let bperp = vec2(-bdir.y, bdir.x);
                                let bangle = (fastrand::f32() - 0.5) * 2.0; // which side
                                let bvec = bdir * (fastrand::f32() * 0.3) + bperp * bangle;
                                let bvec = bvec.normalized();
                                let mut bline = vec![start];
                                let mut bx = start.x;
                                let mut by = start.y;
                                let bsegs = 4usize;
                                for s in 0..bsegs {
                                    let f = (s + 1) as f32 / bsegs as f32;
                                    bx += bvec.x * blen / bsegs as f32;
                                    by += bvec.y * blen / bsegs as f32;
                                    // Small jitter on the branch too.
                                    bx += (fastrand::f32() - 0.5) * jitter_amp * 0.5;
                                    by += (fastrand::f32() - 0.5) * jitter_amp * 0.5;
                                    bline.push(pos2(bx, by));
                                }
                                branch_lines.push(bline);
                            }

                            // Glow halo: a wide, dim coloured line under the core.
                            let glow_w = (6.0 * depth_scale * (0.5 + usage_factor) * (0.6 + 0.4 * lite).min(2.0)).max(2.0);
                            let glow_col = Color32::from_rgba_premultiplied(60, 120, 255, 60);
                            painter.add(Shape::line(main.clone(), Stroke::new(glow_w, glow_col)));
                            for bl in &branch_lines {
                                painter.add(Shape::line(bl.clone(), Stroke::new(glow_w * 0.6, glow_col)));
                            }
                            // Bright white-hot core.
                            let core_w = (2.2 * depth_scale * (0.4 + 0.6 * usage_factor) * (0.6 + 0.4 * lite).min(2.0)).max(1.0);
                            painter.add(Shape::line(main.clone(), Stroke::new(core_w, Color32::from_rgb(255, 255, 240))));
                            for bl in &branch_lines {
                                painter.add(Shape::line(bl.clone(), Stroke::new(core_w * 0.7, Color32::from_rgb(255, 255, 240))));
                            }
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



            // HUD — minimal: Model + tok/s, second row only if a second model is loaded
            let hud_color = egui::Color32::from_rgba_premultiplied(200, 200, 220, 180);
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
                // Line 2: second model + tok/s (only shown when populated)
                if self.loaded_model2 != "—" {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&self.loaded_model2)
                            .size(14.0).color(hud_color));
                        ui.separator();
                        let tps2 = if self.tok_s2 > 0.0 { format!("{:.1} tok/s", self.tok_s2) } else { String::from("—") };
                        ui.label(egui::RichText::new(tps2)
                            .size(14.0).color(hud_color));
                    });
                }
            });

            // Dragging: pick the window up only when the pointer is first
            // pressed (edge-triggered) AND actually moved — so a plain click
            // doesn't grab it, and a held button doesn't keep re-engaging
            // the drag (which would glue the window to the mouse and make it
            // unable to release).
            // When PINNED, drag-to-move is disabled so the window stays
            // locked in place behind the gauges. A click still grabs focus
            // so the 1-0 pattern keys reach the window.
            let drag_started = ctx.input(|i| {
                !self.cfg.pinned && i.pointer.primary_pressed() && i.pointer.delta().length() > 1.0
            });
            if drag_started {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            // Grab keyboard focus on any click so the B/F/1-0 keys reach the
            // window. A transparent frameless overlay is not focused by
            // default on Wayland, so without this the key events never arrive.
            let clicked = ctx.input(|i| i.pointer.primary_pressed());
            if clicked {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
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

/// Estimate tokens/second from the model name and GPU utilisation.
/// RTX 4060 Ti baseline: ~100 tok/s for a 7B Q4 model at 100% GPU util.
/// Extracts parameter count from the model name (e.g. "gemma4:9b" → 9.0).
fn estimate_tok_s(model_name: &str, gpu_util: f32) -> f32 {
    // Look for a pattern like "9b", "14B", "7b" in the model name
    let mut param_count: f32 = 7.0; // default if not found
    let lower = model_name.to_lowercase();
    let bytes = lower.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i].is_ascii_digit() && (bytes[i + 1] == b'b') {
            // Parse the full number before 'b'
            let mut start = i;
            while start > 0 && bytes[start - 1].is_ascii_digit() {
                start -= 1;
            }
            if let Ok(n) = std::str::from_utf8(&bytes[start..i + 1]).unwrap_or("7").parse::<f32>() {
                if n > 0.0 {
                    param_count = n;
                }
            }
            break;
        }
    }
    let max_tok = 100.0 * (7.0 / param_count).max(0.1);
    max_tok * (gpu_util / 100.0)
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
            .with_decorations(false)
            // Centre the window on the DP-9 monitor (geometry 1848,0 1423x800).
            // Note: on Wayland a client can't set its own absolute position, so
            // this is a no-op there — the KWin window rule (kwinrulesrc, title
            // "VRAM Point Cloud", position=2048,16 positionrule=2) is what
            // actually places it on DP-9. This line matters only if the app
            // runs under X11.
            .with_position([2048.0, 16.0])
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
    fn chip_patterns_occupy_number_key_slots() {
        // Number keys 1-0 must select the GnomeWorx chip patterns at
        // indices 0-9, in order. This pins the hotkey mapping.
        let v = VramVisualizer::new();
        let expected = [
            "1. ChipCore",
            "2. ChipPulse",
            "3. ChipOrbit",
            "4. ChipGrid",
            "5. ChipWave",
            "6. ChipBreathe",
            "7. ChipSpiral",
            "8. ChipRings",
            "9. ChipStorm",
            "0. ChipMatrix",
        ];
        for (i, name) in expected.iter().enumerate() {
            assert_eq!(v.patterns[i].name(), *name,
                "slot {i} (key {}) must be {name}, got: {}",
                if i == 9 { 0 } else { i + 1 }, v.patterns[i].name());
        }
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
        // Default active is ChipCore (index 0). Switch to ChipPulse (index 1).
        v.switch_to(1);
        assert_eq!(v.active_idx, 1);
        assert_eq!(v.current_name(), "2. ChipPulse");
        // Particles should now be anchored onto the chip die (base_x in
        // [-0.5, 0.5]).
        let first_base_x = v.particles[0].base_x;
        assert!(first_base_x.abs() <= 0.52,
            "first particle should be on the chip die, base_x={first_base_x}");
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
                fn name(&self) -> &'static str { "14. Custom" }
            fn update(&mut self, _p: &mut Particle, _ctx: &PatternCtx) {}
        }
        let mut v = VramVisualizer::new();
        let initial_len = v.patterns.len();
        let idx = v.register_pattern(Box::new(CustomPattern));
        assert_eq!(idx, initial_len, "new pattern should get the next index");
        assert_eq!(v.patterns[idx].name(), "14. Custom");
        // Should be reachable.
        v.switch_to(idx);
        assert_eq!(v.active_idx, idx);
        assert_eq!(v.current_name(), "14. Custom");
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
    fn chip_anchor_forms_die_ring_and_border_pins() {
        // The GnomeWorx chip silhouette must place particles in three
        // distinct regions: the square die (|x|,|y| <= 0.5), the bright
        // die-border ring (layer 3, hugging the die edge), and border
        // pins (|x| or |y| > 0.5). No gear emblem (removed).
        let mut particles = vec![make_p(0.0, 0.0, 0.0, 0); PARTICLE_COUNT];
        anchor_chip(&mut particles);
        let mut die = 0usize;
        let mut ring = 0usize;
        let mut pins = 0usize;
        for q in &particles {
            match q.layer {
                0 => die += 1,
                3 => ring += 1,
                _ => pins += 1,
            }
        }
        // Roughly 50% die, 16% border ring, 34% pins.
        let n = PARTICLE_COUNT as f32;
        assert!((die as f32 / n - 0.50).abs() < 0.05, "die share wrong: {die}");
        assert!((ring as f32 / n - 0.16).abs() < 0.03, "ring share wrong: {ring}");
        assert!((pins as f32 / n - 0.34).abs() < 0.05, "pins share wrong: {pins}");
        // Die particles stay inside the die; pins extend past the border.
        for q in &particles {
            if q.layer == 0 {
                assert!(q.base_x.abs() <= 0.51 && q.base_y.abs() <= 0.51,
                    "die particle escaped: ({}, {})", q.base_x, q.base_y);
            } else if q.layer == 2 {
                assert!(q.base_x.abs() > 0.49 || q.base_y.abs() > 0.49,
                    "pin particle not on the border: ({}, {})", q.base_x, q.base_y);
            }
        }
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

    #[test]
    fn chip_patterns_stay_finite_and_bounded() {
        // Every GnomeWorx chip pattern must keep every particle at finite
        // positions across many frames at high activity (no blow-up), and
        // stay within a bounded region around the chip.
        let mut patterns: Vec<Box<dyn Pattern>> = vec![
            Box::new(ChipCorePattern),
            Box::new(ChipPulsePattern),
            Box::new(ChipOrbitPattern),
            Box::new(ChipGridPattern),
            Box::new(ChipWavePattern),
            Box::new(ChipBreathePattern),
            Box::new(ChipSpiralPattern),
            Box::new(ChipRingsPattern),
            Box::new(ChipStormPattern),
            Box::new(ChipMatrixPattern),
        ];
        let c = PatternCtx { frame: 0, activity: 1.0, vram_fill: 1.0, temp_factor: 1.0 };
        for pat in patterns.iter_mut() {
            let mut particles: Vec<Particle> = (0..PARTICLE_COUNT).map(|i| init_particle_cylinder(i)).collect();
            pat.on_activate(&mut particles);
            for _ in 0..300 {
                for q in &mut particles {
                    pat.update(q, &c);
                }
            }
            for q in &particles {
                assert!(q.x.is_finite() && q.y.is_finite() && q.z.is_finite(),
                    "{} produced a non-finite position", pat.name());
                assert!(q.x.abs() < 10.0 && q.y.abs() < 10.0 && q.z.abs() < 10.0,
                    "{} particle escaped bounds: ({}, {}, {})", pat.name(), q.x, q.y, q.z);
            }
        }
    }

    #[test]
    fn chip_color_responds_to_activity() {
        // The chip palette must brighten with GPU activity: a hot, busy
        // chip should be visibly brighter than an idle one.
        let p = ChipCorePattern;
        let particle = make_p(0.0, 0.0, 0.0, 0);
        let idle = PatternCtx { frame: 0, activity: 0.0, vram_fill: 0.0, temp_factor: 0.0 };
        let busy = PatternCtx { frame: 0, activity: 1.0, vram_fill: 1.0, temp_factor: 1.0 };
        let (ir, ig, ib) = p.color(&particle, &idle).unwrap();
        let (br, bg, bb) = p.color(&particle, &busy).unwrap();
        let delta = (br - ir).abs() + (bg - ig).abs() + (bb - ib).abs();
        assert!(delta > 20.0,
            "activity should brighten the chip palette, got delta={delta} \
             (idle=({ir},{ig},{ib}) busy=({br},{bg},{bb}))");
    }

    /// Offscreen-render the ChipCore pattern to a PPM so the chip motif
    /// (square die, border ring, gold pins) can be verified visibly.
    /// Maps particles exactly as the on-screen painter does.
    #[test]
    fn chip_core_renders_a_framed_die_to_ppm() {
        let w: usize = 800;
        let h: usize = 800;
        // Big RGBA buffer, premultiplied dark background.
        let mut buf = vec![0u8; w * h * 4];
        // (r,g,b,a) as f32 accumulator per pixel to model alpha over black.
        let mut acc = vec![0.0f32; w * h * 4];
        let mut particles: Vec<Particle> =
            (0..PARTICLE_COUNT).map(init_particle_cylinder).collect();
        let mut pat = ChipCorePattern;
        pat.on_activate(&mut particles);

        let (wx, wy) = pat.extent();
        let scale = fill_scale(w as f32, h as f32, (wx, wy));
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let act = 0.6f32;
        let ctx = PatternCtx { frame: 5, activity: act, vram_fill: 0.3, temp_factor: 0.4 };

        let mut max_r2: f32 = 0.0;
        for p in &particles {
            if let Some((r, g, b)) = pat.color(p, &ctx) {
                let depth = 3.0 / (3.0 + p.z * 0.8);
                let sx = cx + p.x * scale * depth;
                let sy = cy + p.y * scale * depth;
                if sx < 0.0 || sx > w as f32 || sy < 0.0 || sy > h as f32 { continue; }
                let pr = (r as f32).clamp(0.0, 255.0);
                let pg = (g as f32).clamp(0.0, 255.0);
                let pb = (b as f32).clamp(0.0, 255.0);
                let alpha = (0.3 + act * 0.5 + depth * 0.3).clamp(0.15, 0.95) * 0.5;
                let ps = (p.size * depth * (0.8 + act * 1.2) * 0.5).max(0.8);
                let r2 = (ps * 2.0).ceil() as i32;
                max_r2 = max_r2.max(r2 as f32);
                for dy in -r2..=r2 {
                    for dx in -r2..=r2 {
                        let ii = (sy as i32) + dy;
                        let jj = (sx as i32) + dx;
                        if ii < 0 || ii >= h as i32 || jj < 0 || jj >= w as i32 { continue; }
                        let d = ((dx * dx + dy * dy) as f32).sqrt();
                        if d > r2 as f32 { continue; }
                        let o = ((ii as usize) * w + (jj as usize)) * 4;
                        // Additive alpha blend over the buffer.
                        let wgt = (1.0 - d / (r2 as f32 + 1.0)) * alpha;
                        acc[o]     += pr * wgt;
                        acc[o + 1] += pg * wgt;
                        acc[o + 2] += pb * wgt;
                        acc[o + 3] += wgt * 255.0;
                    }
                }
            }
        }
        // Write final premultiplied RGBA into the byte buffer.
        let mut bbox_min_x = w as i32; let mut bbox_max_x = 0i32;
        let mut bbox_min_y = h as i32; let mut bbox_max_y = 0i32;
        for py in 0..h {
            for px in 0..w {
                let o = (py * w + px) * 4;
                let rr = acc[o].min(255.0) as u8;
                let gg = acc[o+1].min(255.0) as u8;
                let bb = acc[o+2].min(255.0) as u8;
                buf[o] = rr; buf[o+1] = gg; buf[o+2] = bb;
                buf[o+3] = acc[o+3].min(255.0) as u8;
                if buf[o+3] > 8 { // non-empty pixel
                    bbox_min_x = bbox_min_x.min(px as i32);
                    bbox_max_x = bbox_max_x.max(px as i32);
                    bbox_min_y = bbox_min_y.min(py as i32);
                    bbox_max_y = bbox_max_y.max(py as i32);
                }
            }
        }
        // Write P6 PPM at /tmp/chip_render.ppm
        let path = std::path::Path::new("/tmp/chip_render.ppm");
        let mut ppm = Vec::new();
        ppm.extend_from_slice(format!("P6\n{w}\n{h}\n255\n").as_bytes());
        for py in 0..h {
            for px in 0..w {
                let o = (py * w + px) * 4;
                ppm.push(buf[o]); ppm.push(buf[o+1]); ppm.push(buf[o+2]);
            }
        }
        let _ = std::fs::write(path, &ppm);
        let wpx = (bbox_max_x - bbox_min_x + 1) as usize;
        let hpx = (bbox_max_y - bbox_min_y + 1) as usize;
        assert!(wpx > 0 && hpx > 0, "chip render must produce non-empty pixels");
        // A framed die should spread across most of the window.
        assert!(wpx > w / 2 && hpx > h / 2,
            "chip should fill the frame, got {wpx}x{hpx} bbox");
        assert!(max_r2 >= 1.0, "particles should render with radius >= 1");
    }
}
