use eframe::egui;
use std::process::Command;
use std::time::Instant;

struct GpuData {
    util: f32,
    vram_used_mb: f32,
    vram_total_mb: f32,
    temp_c: f32,
    power_w: f32,
}

#[derive(Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    phase: f32,
    size: f32,
}

struct VramVisualizer {
    gpu: GpuData,
    error: String,
    particles: Vec<Particle>,
    last_poll: Instant,
    frame: u64,
    width: f32,
    height: f32,
}

impl Default for VramVisualizer {
    fn default() -> Self {
        let count = 3000;
        let mut particles = Vec::with_capacity(count);
        for i in 0..count {
            // Distribute in a cylindrical volume — like a data cloud
            let angle = (i as f32 * 2.5) % std::f32::consts::TAU;
            let radius = 0.15 + (i as f32 / count as f32) * 0.70;
            let height_z = ((i as f32 * 1.7).sin() * 0.6) as f32;
            let x = angle.cos() * radius;
            let y = angle.sin() * radius;
            let z = height_z;
            particles.push(Particle {
                x, y, z,
                vx: 0.0, vy: 0.0, vz: 0.0,
                base_x: x, base_y: y, base_z: z,
                phase: (i as f32 * 0.1) % std::f32::consts::TAU,
                size: 1.5 + (i % 3) as f32 * 0.8,
            });
        }
        Self {
            gpu: GpuData { util: 0.0, vram_used_mb: 0.0, vram_total_mb: 0.0, temp_c: 0.0, power_w: 0.0 },
            error: String::new(),
            particles,
            last_poll: Instant::now(),
            frame: 0,
            width: 800.0,
            height: 600.0,
        }
    }
}

impl VramVisualizer {
    fn poll(&mut self) {
        if self.last_poll.elapsed().as_millis() < 500 {
            return;
        }
        self.last_poll = Instant::now();
        match query_nvidia_smi() {
            Ok(d) => { self.gpu = d; self.error.clear(); }
            Err(e) => self.error = e,
        }
    }

    fn update_particles(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        let t = self.frame as f32 * 0.016;
        let activity = (self.gpu.util / 100.0).clamp(0.0, 1.0);
        let vram_fill = if self.gpu.vram_total_mb > 0.0 {
            (self.gpu.vram_used_mb / self.gpu.vram_total_mb).clamp(0.0, 1.0)
        } else { 0.0 };

        for p in &mut self.particles {
            // Spiral drift
            let a = t * 0.3 + p.phase;
            let orbit_radius = activity * 0.15;
            let target_x = p.base_x + a.cos() * orbit_radius;
            let target_y = p.base_y + a.sin() * orbit_radius;
            let target_z = p.base_z + (t * 0.5 + p.phase).sin() * activity * 0.2;

            // Spring toward target
            let spring = 0.04 + vram_fill * 0.06;
            p.vx += (target_x - p.x) * spring;
            p.vy += (target_y - p.y) * spring;
            p.vz += (target_z - p.z) * spring;

            // Damping
            let damp = 0.92 - activity * 0.05;
            p.vx *= damp;
            p.vy *= damp;
            p.vz *= damp;

            // Random jitter based on activity
            let jitter = activity * 0.002;
            p.vx += (p.phase.sin() * jitter);
            p.vy += (p.phase.cos() * jitter);
            p.vz += ((p.phase * 1.3).sin() * jitter);

            p.x += p.vx;
            p.y += p.vy;
            p.z += p.vz;
        }
    }
}

impl eframe::App for VramVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        self.update_particles();

        let screen = ctx.screen_rect();
        self.width = screen.width();
        self.height = screen.height();

        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let rect = ui.max_rect();

            // Transparent background — desktop shows through behind particles
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgba_premultiplied(0, 0, 0, 20));

            let cx = rect.center().x;
            let cy = rect.center().y;
            let scale = rect.height() * 0.42;

            let activity = (self.gpu.util / 100.0).clamp(0.0, 1.0);
            let temp_factor = ((self.gpu.temp_c - 25.0) / 60.0).clamp(0.0, 1.0);
            let vram_fill = if self.gpu.vram_total_mb > 0.0 {
                (self.gpu.vram_used_mb / self.gpu.vram_total_mb).clamp(0.0, 1.0)
            } else { 0.0 };

            // Sort particles by z for depth
            let mut sorted: Vec<(f32, f32, f32, f32, f32)> = self.particles.iter().map(|p| {
                (p.x, p.y, p.z, p.size, p.phase)
            }).collect();
            sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            let t = self.frame as f32 * 0.016;

            for (px, py, pz, size, phase) in &sorted {
                // Perspective projection
                let depth_scale = 1.0 / (1.0 + *pz * 0.8);
                let sx = cx + px * scale * depth_scale;
                let sy = cy + py * scale * depth_scale;

                if sx < rect.left() - 10.0 || sx > rect.right() + 10.0
                    || sy < rect.top() - 10.0 || sy > rect.bottom() + 10.0 {
                    continue;
                }

                // Color: cool blue at rest, shifting to hot orange/red with activity
                let shimmer = (*phase + t * 2.0).sin() * 0.2 + 0.8;
                let pulse = shimmer * (0.7 + activity * 0.5);

                let r = (80.0 + activity * 160.0 + temp_factor * 80.0) * pulse;
                let g = (120.0 * (1.0 - activity * 0.7) + vram_fill * 60.0) * pulse;
                let b = (200.0 * (1.0 - activity * 0.8) - temp_factor * 80.0) * pulse;

                let alpha = (0.3 + activity * 0.5 + depth_scale * 0.3).clamp(0.15, 0.95);
                let color = egui::Color32::from_rgba_premultiplied(
                    (r as u8).clamp(0, 255),
                    (g as u8).clamp(0, 255),
                    (b as u8).clamp(0, 255),
                    (alpha * 255.0) as u8,
                );

                let point_size = *size * depth_scale * (0.8 + activity * 1.2);

                // Glow halo
                painter.circle_filled(
                    egui::pos2(sx, sy),
                    point_size * 2.5,
                    egui::Color32::from_rgba_premultiplied(
                        (r as u8).saturating_mul(0) / 1, 0, 0,
                        (alpha * 0.15 * 255.0) as u8,
                    ),
                );
                // Core
                painter.circle_filled(egui::pos2(sx, sy), point_size, color);
            }

            // HUD overlay — top-left
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
            Box::new(VramVisualizer::default())
        }),
    )
}
