use eframe::egui;
use std::process::Command;
use std::time::Instant;

// ─── pattern enum ───
#[derive(Clone, Copy, PartialEq)]
enum Pattern {
    Vortex,      // 0: black hole swirl
    Cylinder,    // 1: original orbiting cylinder
    Scatter3D,   // 2: X=time, Y=util, Z=other metrics
    ProcessCloud,// 3: colored by synthetic process zones
    Animation,   // 4: streaming points appearing over time
    HeatScale,   // 5: brightness-only color scaling
    OrbitCube,   // 6: 3D rotating cube shell
    Regions,     // 7: height-mapped regions
    Wavefield,   // 8: sine-wave displacement field
    SpiralGalaxy,// 9: galaxy arms rotating
    GridWave,    // 10: grid with wave propagation
}

impl Pattern {
    fn next(self) -> Self {
        match self {
            Pattern::Vortex => Pattern::Cylinder,
            Pattern::Cylinder => Pattern::Scatter3D,
            Pattern::Scatter3D => Pattern::ProcessCloud,
            Pattern::ProcessCloud => Pattern::Animation,
            Pattern::Animation => Pattern::HeatScale,
            Pattern::HeatScale => Pattern::OrbitCube,
            Pattern::OrbitCube => Pattern::Regions,
            Pattern::Regions => Pattern::Wavefield,
            Pattern::Wavefield => Pattern::SpiralGalaxy,
            Pattern::SpiralGalaxy => Pattern::GridWave,
            Pattern::GridWave => Pattern::Vortex,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Pattern::Vortex => "1. Vortex",
            Pattern::Cylinder => "2. Cylinder",
            Pattern::Scatter3D => "3. Scatter3D",
            Pattern::ProcessCloud => "4. ProcessCloud",
            Pattern::Animation => "5. Animation",
            Pattern::HeatScale => "6. HeatScale",
            Pattern::OrbitCube => "7. OrbitCube",
            Pattern::Regions => "8. Regions",
            Pattern::Wavefield => "9. Wavefield",
            Pattern::SpiralGalaxy => "10. SpiralGalaxy",
            Pattern::GridWave => "11. GridWave",
        }
    }
}

// ─── types ───
#[derive(Clone, Copy)]
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
    age: f32,          // for animation pattern
    process_id: u8,    // 0-7 synthetic process zones
}

struct VramVisualizer {
    gpu: GpuData,
    error: String,
    particles: Vec<Particle>,
    last_poll: Instant,
    frame: u64,
    pattern: Pattern,
    history: Vec<GpuData>, // rolling history for scatter/animation
}

const PARTICLE_COUNT: usize = 3000;
const HISTORY_MAX: usize = 240;

impl VramVisualizer {
    fn new() -> Self {
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        for i in 0..PARTICLE_COUNT {
            particles.push(Self::init_particle_cylinder(i));
        }
        Self {
            gpu: GpuData { util: 0.0, vram_used_mb: 0.0, vram_total_mb: 0.0, temp_c: 0.0, power_w: 0.0 },
            error: String::new(),
            particles,
            last_poll: Instant::now(),
            frame: 0,
            pattern: Pattern::Vortex,
            history: Vec::with_capacity(HISTORY_MAX),
        }
    }

    fn init_particle_cylinder(i: usize) -> Particle {
        let n = PARTICLE_COUNT as f32;
        let angle = (i as f32 * 2.5) % std::f32::consts::TAU;
        let radius = 0.15 + (i as f32 / n) * 0.70;
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
        }
    }

    fn respawn_particle(p: &mut Particle) {
        let i = fastrand::u32(0..PARTICLE_COUNT as u32) as usize;
        *p = Self::init_particle_cylinder(i);
    }

    fn poll(&mut self) {
        if self.last_poll.elapsed().as_millis() < 500 {
            return;
        }
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
    }

    fn activity(&self) -> f32 { (self.gpu.util / 100.0).clamp(0.0, 1.0) }
    fn vram_fill(&self) -> f32 {
        if self.gpu.vram_total_mb > 0.0 {
            (self.gpu.vram_used_mb / self.gpu.vram_total_mb).clamp(0.0, 1.0)
        } else { 0.0 }
    }
    fn temp_factor(&self) -> f32 { ((self.gpu.temp_c - 25.0) / 60.0).clamp(0.0, 1.0) }

    fn update_particles(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        match self.pattern {
            Pattern::Vortex => self.update_vortex(),
            Pattern::Cylinder => self.update_cylinder(),
            Pattern::Scatter3D => self.update_scatter3d(),
            Pattern::ProcessCloud => self.update_process_cloud(),
            Pattern::Animation => self.update_animation(),
            Pattern::HeatScale => self.update_cylinder(),
            Pattern::OrbitCube => self.update_orbit_cube(),
            Pattern::Regions => self.update_regions(),
            Pattern::Wavefield => self.update_wavefield(),
            Pattern::SpiralGalaxy => self.update_spiral_galaxy(),
            Pattern::GridWave => self.update_gridwave(),
        }
    }

    // ─── pattern physics ───

    fn update_vortex(&mut self) {
        let _t = self.frame as f32 * 0.016;
        let act = self.activity();
        let _vfill = self.vram_fill();

        for p in &mut self.particles {
            // Apply velocity first
            p.x += p.vx;
            p.y += p.vy;
            p.z += p.vz;

            // Vector to center
            let dx = -p.x;
            let dy = -p.y;
            let dist = (dx * dx + dy * dy).sqrt().max(0.001);

            // Tangential swirl direction (perpendicular to radial)
            // Swirl speed varies with distance: faster at edges, slower near center
            let swirl_speed = 0.0004 * (1.0 + act) * dist;
            p.vx += dy * swirl_speed;
            p.vy += -dx * swirl_speed;

            // Gravity pull — gentle, increases slightly with activity
            let gravity = 0.00015 + act * 0.0002;
            let pull = gravity / (dist + 0.05);
            p.vx += dx * pull;
            p.vy += dy * pull;

            // Gentle pull toward z=0 plane
            p.vz += -p.z * 0.005;

            // Light damping
            p.vx *= 0.993;
            p.vy *= 0.993;
            p.vz *= 0.97;

            // Respawn at edge when inside the event horizon
            if dist < 0.03 {
                let a = fastrand::f32() * std::f32::consts::TAU;
                let r = 0.7 + fastrand::f32() * 0.3;
                p.x = a.cos() * r;
                p.y = a.sin() * r;
                p.z = (fastrand::f32() - 0.5) * 1.0;
                // Give it initial tangential velocity so it doesn't just fall straight in
                let init_v = 0.001 + fastrand::f32() * 0.003;
                p.vx = -a.sin() * init_v;
                p.vy = a.cos() * init_v;
                p.vz = (fastrand::f32() - 0.5) * 0.001;
            }
        }
    }

    fn update_cylinder(&mut self) {
        let t = self.frame as f32 * 0.016;
        let act = self.activity();
        let vfill = self.vram_fill();

        for p in &mut self.particles {
            let a = t * 0.3 + p.phase;
            let orbit = act * 0.15;
            let tx = p.base_x + a.cos() * orbit;
            let ty = p.base_y + a.sin() * orbit;
            let tz = p.base_z + (t * 0.5 + p.phase).sin() * act * 0.2;

            let spring = 0.04 + vfill * 0.06;
            p.vx += (tx - p.x) * spring;
            p.vy += (ty - p.y) * spring;
            p.vz += (tz - p.z) * spring;

            let damp = 0.92 - act * 0.05;
            p.vx *= damp; p.vy *= damp; p.vz *= damp;

            let jit = act * 0.002;
            p.vx += p.phase.sin() * jit;
            p.vy += p.phase.cos() * jit;
            p.vz += (p.phase * 1.3).sin() * jit;

            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_scatter3d(&mut self) {
        // X=time index, Y=util, Z=temp/power
        let act = self.activity();
        let vfill = self.vram_fill();
        let n = self.history.len().max(1) as f32;

        for (i, p) in self.particles.iter_mut().enumerate() {
            let hi = (i as f32 / PARTICLE_COUNT as f32 * (n - 1.0)) as usize;
            let hist = self.history.get(hi).copied().unwrap_or(self.gpu);

            let tx = (hi as f32 / n.max(1.0) - 0.5) * 1.8;
            let ty = (hist.util / 100.0 - 0.5) * 1.6;
            let tz = ((hist.temp_c - 25.0) / 60.0 - 0.5) * 1.2;

            p.vx += (tx - p.x) * 0.06;
            p.vy += (ty - p.y) * 0.06;
            p.vz += (tz - p.z) * 0.06;
            p.vx *= 0.90; p.vy *= 0.90; p.vz *= 0.90;

            let jit = act * 0.001;
            p.vx += p.phase.sin() * jit;
            p.vy += p.phase.cos() * jit;

            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_process_cloud(&mut self) {
        // Process-colored zones — particles cluster by process_id
        let t = self.frame as f32 * 0.016;
        let act = self.activity();

        for p in &mut self.particles {
            let pid = p.process_id as f32;
            // Each process gets a different zone center
            let cx = ((pid * 1.7).cos() * 0.5) as f32;
            let cy = ((pid * 1.7).sin() * 0.5) as f32;
            let cz = ((pid * 0.9).sin() * 0.3) as f32;

            let orbit = act * 0.08;
            let tx = cx + (t * 0.4 + pid).cos() * orbit;
            let ty = cy + (t * 0.4 + pid).sin() * orbit;
            let tz = cz + (t * 0.3).sin() * 0.1;

            p.vx += (tx - p.x) * 0.05;
            p.vy += (ty - p.y) * 0.05;
            p.vz += (tz - p.z) * 0.05;
            p.vx *= 0.91; p.vy *= 0.91; p.vz *= 0.91;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_animation(&mut self) {
        // Particles have age — they appear, drift, then respawn
        let act = self.activity();

        for p in &mut self.particles {
            p.age += 0.016;
            // Drift upward and fade
            p.vy += act * 0.0003;
            p.vx += p.phase.sin() * 0.0002;
            p.vx *= 0.98; p.vy *= 0.98; p.vz *= 0.98;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;

            // Respawn when too old or drifted too far
            if p.age > 8.0 || p.y > 0.9 || p.y < -0.9 {
                p.x = (fastrand::f32() - 0.5) * 0.8;
                p.y = -0.85;
                p.z = (fastrand::f32() - 0.5) * 0.6;
                p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
                p.age = 0.0;
            }
        }
    }

    fn update_orbit_cube(&mut self) {
        // Particles orbit on the surface of a rotating cube
        let t = self.frame as f32 * 0.016;
        for p in &mut self.particles {
            let a = t * 0.2 + p.phase;
            let cos_a = a.cos();
            let sin_a = a.sin();

            // Rotate base position
            let rx = p.base_x * cos_a - p.base_z * sin_a;
            let rz = p.base_x * sin_a + p.base_z * cos_a;

            // Project base positions onto a cube surface
            let abs_x = p.base_x.abs();
            let abs_y = p.base_y.abs();
            let abs_z = p.base_z.abs();
            let max_dim = abs_x.max(abs_y).max(abs_z).max(0.01);

            let tx = p.base_x / max_dim * 0.8;
            let ty = p.base_y / max_dim * 0.8;
            let tz = p.base_z / max_dim * 0.8;

            let rtx = tx * cos_a - tz * sin_a;
            let rtz = tx * sin_a + tz * cos_a;

            p.vx += (rtx - p.x) * 0.06;
            p.vy += (ty - p.y) * 0.06;
            p.vz += (rtz - p.z) * 0.06;
            p.vx *= 0.88; p.vy *= 0.88; p.vz *= 0.88;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_regions(&mut self) {
        // Height-mapped regions based on particle position
        let t = self.frame as f32 * 0.016;
        let act = self.activity();
        for p in &mut self.particles {
            // Height = sin(x*3) * cos(y*3) scaled by activity
            let h = (p.base_x * 4.0).sin() * (p.base_y * 4.0).cos() * act * 0.5;
            let tx = p.base_x;
            let ty = p.base_y;
            let tz = h;
            p.vx += (tx - p.x) * 0.05;
            p.vy += (ty - p.y) * 0.05;
            p.vz += (tz - p.z) * 0.05;
            p.vx *= 0.90; p.vy *= 0.90; p.vz *= 0.90;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_wavefield(&mut self) {
        // Sine-wave displacement field
        let t = self.frame as f32 * 0.016;
        let act = self.activity();
        for p in &mut self.particles {
            let wave = (p.base_x * 3.0 + t).sin() * (p.base_y * 3.0 + t * 1.3).cos() * act * 0.4;
            let tx = p.base_x + (p.base_y * 2.0 + t).cos() * act * 0.15;
            let ty = p.base_y + (p.base_x * 2.0 + t).sin() * act * 0.15;
            let tz = wave;
            p.vx += (tx - p.x) * 0.06;
            p.vy += (ty - p.y) * 0.06;
            p.vz += (tz - p.z) * 0.06;
            p.vx *= 0.90; p.vy *= 0.90; p.vz *= 0.90;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_spiral_galaxy(&mut self) {
        // Multiple spiral arms rotating
        let t = self.frame as f32 * 0.016;
        let act = self.activity();
        for p in &mut self.particles {
            let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt().max(0.01);
            let base_angle = p.base_y.atan2(p.base_x);
            let spiral_offset = dist * 3.0;
            let rot_speed = 0.3 / (dist + 0.2) * (1.0 + act);
            let angle = base_angle + t * rot_speed - spiral_offset;
            let r = dist * (0.9 + act * 0.15);
            let tx = angle.cos() * r;
            let ty = angle.sin() * r;
            let tz = p.base_z + (dist * 4.0 + t).sin() * act * 0.1;

            p.vx += (tx - p.x) * 0.07;
            p.vy += (ty - p.y) * 0.07;
            p.vz += (tz - p.z) * 0.07;
            p.vx *= 0.89; p.vy *= 0.89; p.vz *= 0.89;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }

    fn update_gridwave(&mut self) {
        // Grid layout with wave propagation from center
        let t = self.frame as f32 * 0.016;
        let act = self.activity();
        for p in &mut self.particles {
            let dist = (p.base_x * p.base_x + p.base_y * p.base_y).sqrt();
            let wave = (dist * 6.0 - t * 2.0 * (1.0 + act)).sin() * act * 0.5;
            let tx = p.base_x;
            let ty = p.base_y;
            let tz = wave;
            p.vx += (tx - p.x) * 0.06;
            p.vy += (ty - p.y) * 0.06;
            p.vz += (tz - p.z) * 0.06;
            p.vx *= 0.89; p.vy *= 0.89; p.vz *= 0.89;
            p.x += p.vx; p.y += p.vy; p.z += p.vz;
        }
    }
}

// ─── rendering ───
impl eframe::App for VramVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        self.update_particles();

        // Keyboard pattern switching
        ctx.input(|i| {
            for ev in &i.events {
                if let egui::Event::Key { key: egui::Key::Tab, pressed: true, .. } = ev {
                    self.pattern = self.pattern.next();
                }
                if let egui::Event::Key { key, pressed: true, .. } = ev {
                    match key {
                        egui::Key::Num1 => self.pattern = Pattern::Vortex,
                        egui::Key::Num2 => self.pattern = Pattern::Cylinder,
                        egui::Key::Num3 => self.pattern = Pattern::Scatter3D,
                        egui::Key::Num4 => self.pattern = Pattern::ProcessCloud,
                        egui::Key::Num5 => self.pattern = Pattern::Animation,
                        egui::Key::Num6 => self.pattern = Pattern::HeatScale,
                        egui::Key::Num7 => self.pattern = Pattern::OrbitCube,
                        egui::Key::Num8 => self.pattern = Pattern::Regions,
                        egui::Key::Num9 => self.pattern = Pattern::Wavefield,
                        egui::Key::Num0 => self.pattern = Pattern::SpiralGalaxy,
                        egui::Key::Minus => self.pattern = Pattern::GridWave,
                        _ => {}
                    }
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let rect = ui.max_rect();
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgba_premultiplied(0, 0, 0, 20));

            let cx = rect.center().x;
            let cy = rect.center().y;
            let scale = rect.height() * 0.42;

            let act = self.activity();
            let tfact = self.temp_factor();
            let vfill = self.vram_fill();

            // Sort by z for depth
            let mut sorted: Vec<(f32, f32, f32, f32, f32, u8, f32)> = self.particles.iter().map(|p| {
                (p.x, p.y, p.z, p.size, p.phase, p.process_id, p.age)
            }).collect();
            sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            let t = self.frame as f32 * 0.016;

            for (px, py, pz, size, phase, proc_id, age) in &sorted {
                let depth_scale = 1.0 / (1.0 + *pz * 0.8);
                let sx = cx + px * scale * depth_scale;
                let sy = cy + py * scale * depth_scale;

                if sx < rect.left() - 10.0 || sx > rect.right() + 10.0
                    || sy < rect.top() - 10.0 || sy > rect.bottom() + 10.0 {
                    continue;
                }

                // Color based on pattern
                let (r, g, b) = match self.pattern {
                    Pattern::Vortex => {
                        let dist = (px * px + py * py).sqrt();
                        let depth_near = 1.0 - (dist / 1.0).clamp(0.0, 1.0);

                        // Fire simulation: temperature drives color progression
                        // Cool (idle): dark red/amber base
                        // Hot (>50°C): orange appears
                        // Very hot (>70°C): rare white-hot embers
                        let temp = tfact; // 0..1 mapped from 25-85°C

                        // Base fire: red→orange→yellow progression by depth + temp
                        let base_r = 180.0 + (1.0 - temp) * 75.0;  // 180-255
                        let base_g = 30.0 + temp * 120.0 + depth_near * 80.0; // 30→200 as temp rises
                        let base_b = 5.0 + depth_near * 30.0; // mostly zero, slight blue at depth

                        // Orange bloom: particles near center get orange boost as temp rises
                        let orange_chance = (temp - 0.25).clamp(0.0, 1.0) * 0.4; // 0-40% at max temp
                        let is_orange = (*phase + self.frame as f32 * 0.01).sin() > (1.0 - orange_chance * 2.0);

                        // White ember: rare at >70°C, very bright
                        let white_chance = ((temp - 0.7) / 0.3).clamp(0.0, 1.0) * 0.08; // 0-8%
                        let is_white = white_chance > 0.0
                            && (*phase * 7.0 + self.frame as f32 * 0.03).sin() > (1.0 - white_chance * 2.0);

                        let (r, g, b) = if is_white {
                            (240.0 + temp * 15.0, 200.0 + temp * 55.0, 160.0 + temp * 95.0)
                        } else if is_orange {
                            (220.0 + temp * 35.0, 80.0 + temp * 100.0, 5.0 + temp * 15.0)
                        } else {
                            (base_r, base_g, base_b)
                        };

                        // Flicker: slight random variation
                        let flicker = ((*phase * 13.0 + self.frame as f32 * 0.05).sin() * 0.1 + 0.95) as f32;
                        (r * flicker, g * flicker, b * flicker)
                    }
                    Pattern::ProcessCloud => {
                        let pid = *proc_id as f32;
                        let r = ((pid * 2.2).cos() * 0.5 + 0.5) * 200.0 + 55.0;
                        let g = ((pid * 2.2 + 2.1).cos() * 0.5 + 0.5) * 200.0 + 55.0;
                        let b = ((pid * 2.2 + 4.2).cos() * 0.5 + 0.5) * 200.0 + 55.0;
                        (r, g, b)
                    }
                    Pattern::Animation => {
                        let fade = (*age / 8.0).clamp(0.0, 1.0);
                        let r = 80.0 + act * 180.0;
                        let g = 200.0 * (1.0 - fade);
                        let b = 120.0 * fade + act * 100.0;
                        (r, g, b)
                    }
                    Pattern::HeatScale => {
                        let bright = (vfill * 0.7 + act * 0.3).clamp(0.0, 1.0);
                        let r = 60.0 + bright * 200.0;
                        let g = 40.0 + bright * 80.0;
                        let b = 20.0 + bright * 40.0;
                        (r, g, b)
                    }
                    Pattern::SpiralGalaxy => {
                        let dist = (px * px + py * py).sqrt();
                        let arm_phase = (dist * 3.0 + t).sin() * 0.5 + 0.5;
                        let r = 100.0 + arm_phase * 120.0 + act * 60.0;
                        let g = 60.0 + arm_phase * 140.0;
                        let b = 180.0 + arm_phase * 50.0;
                        (r, g, b)
                    }
                    _ => {
                        let shimmer = (*phase + t * 2.0).sin() * 0.2 + 0.8;
                        let pulse = shimmer * (0.7 + act * 0.5);
                        let r = (80.0 + act * 160.0 + tfact * 80.0) * pulse;
                        let g = (120.0 * (1.0 - act * 0.7) + vfill * 60.0) * pulse;
                        let b = (200.0 * (1.0 - act * 0.8) - tfact * 80.0) * pulse;
                        (r, g, b)
                    }
                };

                let shimmer = (*phase + t * 2.0).sin() * 0.2 + 0.8;
                let pulse = shimmer * (0.7 + act * 0.5);
                let alpha = ((0.3 + act * 0.5 + depth_scale * 0.3).clamp(0.15, 0.95)) * 0.5;

                let color = egui::Color32::from_rgba_premultiplied(
                    (r as u8).clamp(0, 255),
                    (g as u8).clamp(0, 255),
                    (b as u8).clamp(0, 255),
                    (alpha * 255.0) as u8,
                );

                let point_size = *size * depth_scale * (0.8 + act * 1.2) * 0.5;

                // Glow halo
                painter.circle_filled(
                    egui::pos2(sx, sy),
                    point_size * 2.0,
                    egui::Color32::from_rgba_premultiplied(0, 0, 0, (alpha * 0.15 * 255.0) as u8),
                );
                // Core
                painter.circle_filled(egui::pos2(sx, sy), point_size, color);
            }

            // HUD
            let hud_color = egui::Color32::from_rgba_premultiplied(200, 200, 220, 180);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("VRAM  {:.0} / {:.0} MB", self.gpu.vram_used_mb, self.gpu.vram_total_mb))
                    .size(16.0).color(hud_color));
                ui.separator();
                ui.label(egui::RichText::new(format!("GPU  {:.0}%", self.gpu.util))
                    .size(16.0).color(hud_color));
                ui.separator();
                ui.label(egui::RichText::new(format!("{:.0}°C", self.gpu.temp_c))
                    .size(16.0).color(hud_color));
                ui.separator();
                ui.label(egui::RichText::new(format!("{:.0} W", self.gpu.power_w))
                    .size(16.0).color(hud_color));
            });
            // Pattern indicator
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(self.pattern.name())
                    .size(12.0).color(egui::Color32::from_rgba_premultiplied(150, 150, 170, 180)));
                ui.label(egui::RichText::new(" | Tab/1-0/- to switch")
                    .size(12.0).color(egui::Color32::from_rgba_premultiplied(100, 100, 120, 160)));
            });
            if !self.error.is_empty() {
                ui.colored_label(egui::Color32::from_rgb(220, 90, 90), format!("⚠ {}", self.error));
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
