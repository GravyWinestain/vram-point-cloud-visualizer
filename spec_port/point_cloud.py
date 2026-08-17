"""
point_cloud.py — Python harness for the standalone GLSL compute shader
described in compute.glsl.

This is a *separate* project from the Rust/eframe cuda_monitor binary
in this same repo. The Rust binary is the live visual app (12 patterns,
CPU physics, 12k particles); this harness is the spec-faithful port
(Python + PyOpenGL + GLFW, 131k particles, GPU compute).

Run:
    .venv/bin/python3 point_cloud.py

Controls:
    Space      next pattern
    1..4       jump to pattern N
    Esc        quit
"""

import math
import sys
import time

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from telemetry import TelemetrySource
from ollama_driver import OllamaDriver

# ─── Constants from the spec ───
NUM_PARTICLES = 131_072
WORKGROUP_SIZE = 256
# Origin radius for the per-token shockwave in the compute shader.
# Sits inside the 8-unit visualisation bubble but outside the dense
# attractor core (~0.5..3.0), so the radial push is clearly visible.
_BURST_ORIGIN_RADIUS = 1.5
SSBO_STRIDE = 64  # bytes per particle (4 * vec4: pos, vel, col, anchor)
# GridWave anchor spacing: square grid of √N particles per side, scaled
# so the full grid spans roughly ±1.6 on each axis (matches the
# viewport at default aspect). The Rust port uses anchor extent 1.20
# but its visualisation was 1.6..2.4× further away — so the same
# extent here looks the same.

# ─── GLSL sources ───
# The compute shader lives in compute.glsl so it can be read directly
# from disk. Render shaders are tiny — just enough to draw the SSBO as
# a point cloud with perspective-correct point size.

VERTEX_SHADER_SRC = """\
#version 430 core
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_velocity;
layout(location = 2) in vec4 a_color;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_tokens_per_sec;
out vec4 v_color;
void main() {
    v_color = a_color;
    vec4 worldPos = u_model * vec4(a_position.xyz, 1.0);
    vec4 viewPos = u_view * worldPos;
    gl_Position = u_projection * viewPos;
    float speed = length(a_velocity.xyz);
    gl_PointSize = (3.0 + speed * 0.8 + u_tokens_per_sec * 4.0) * (1.0 / -viewPos.z);
}
"""

FRAGMENT_SHADER_SRC = """\
#version 430 core
in vec4 v_color;
out vec4 fragColor;
void main() {
    vec2 circ = gl_PointCoord - vec2(0.5);
    float distSq = dot(circ, circ);
    if (distSq > 0.25) discard;
    float alpha = v_color.a * (1.0 - distSq * 4.0);
    fragColor = vec4(v_color.rgb, alpha);
}
"""


def _read_compute_shader(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _check_gl(label: str) -> None:
    """Surface any pending GL error with a clear label."""
    err = glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"GL error after {label}: 0x{err:04x}")


# ─── Minimal matrix helpers (no glm dep) ───

def _perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


# ─── Engine ───

class PointCloudEngine:
    def __init__(self, num_particles=NUM_PARTICLES):
        self.num_particles = num_particles
        self.window = None
        self.compute_program = None
        self.render_program = None
        self.ssbo = None
        self.vao = None
        self.pattern_mode = 0
        # When True (default), the engine's pattern_mode is driven by
        # the OllamaDriver phase ("idle"→0, "prefill"→1,
        # "generating"→2). Pressing 1/2/3/4 or Space flips this off and
        # locks the mode; press M (or 0) to return to auto.
        self.auto_mode = True
        # Per-token visual reaction state. The driver pushes
        # (mono_time, hash) events; _update_telemetry() consumes them
        # to set token_burst_origin/intensity, which feed into the
        # compute shader's u_token_burst uniform.
        self.token_burst_origin = (0.0, 0.0, 0.0)
        self.token_burst_intensity = 0.0
        # Rolling response-text hash → hue rotation uniform.
        self.response_hue = 0.0
        self.response_text = ""
        # Used by the per-frame log to print the prompt/response only
        # when they actually change (so logs stay readable).
        self._last_prompt_log = None
        self.mode_names = [
            "0: Aizawa Attractor (Idle/Breathing)",
            "1: Context Prefill (Inward Swirl)",
            "2: Token Generation (Harmonic Grid)",
            "3: Thermal Turbulence (Chaos)",
            "4: GridWave (Concentric Ripple)",  # Replicates Rust pattern 11 (-)
        ]
        # Live telemetry from nvidia-smi (passive) — see telemetry.py.
        self.telemetry = TelemetrySource()
        # Live Ollama heartbeat (active) — see ollama_driver.py. This
        # drives the visualizer's pattern_mode when auto_mode is True,
        # and feeds real eval tokens/sec into the compute shader.
        self.ollama = OllamaDriver()

    # ─── OpenGL setup ───
    def init_gl(self):
        if not glfw.init():
            sys.exit("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # Forward-compatible so the 4.3 core profile is strict.
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.window = glfw.create_window(
            1280, 720, "Interactive Point Cloud Visualizer", None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self._key_callback)
        # Track the current framebuffer size; the resize callback keeps
        # this in sync. Used by glViewport() and the projection aspect.
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.width = int(fb_w)
        self.height = int(fb_h)
        # Bind the callback as an attribute so its reference is kept
        # alive for the duration of the program.
        self._resize_callback = self._on_resize
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # additive
        glEnable(GL_PROGRAM_POINT_SIZE)
        _check_gl("GL state setup")

        cs_src = _read_compute_shader("compute.glsl")
        cs = compileShader(cs_src, GL_COMPUTE_SHADER)
        self.compute_program = compileProgram(cs)
        _check_gl("compute program compile")

        vs = compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fs = compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
        self.render_program = compileProgram(vs, fs)
        _check_gl("render program compile")

        self._init_particles()
        _check_gl("particle init")

    def _init_particles(self):
        """Pre-fill the SSBO with random positions on a 3D shell.

        Matches the spec's reference Python: r in [0.5, 3.0], uniform
        on a sphere. The shader will overwrite these on the first
        dispatch; they're just a starting point so the first frame
        isn't a single dot at the origin.
        """
        # 48 bytes = 3 * vec4. Pre-fill xyz, leave w=1 and vy/vz=0.
        rng = np.random.default_rng(seed=42)
        r = rng.uniform(0.5, 3.0, size=self.num_particles).astype(np.float32)
        theta = rng.uniform(0, 2 * math.pi, size=self.num_particles).astype(np.float32)
        phi = rng.uniform(0, math.pi, size=self.num_particles).astype(np.float32)
        pos = np.zeros((self.num_particles, 4), dtype=np.float32)
        pos[:, 0] = r * np.sin(phi) * np.cos(theta)
        pos[:, 1] = r * np.sin(phi) * np.sin(theta)
        pos[:, 2] = r * np.cos(phi)
        pos[:, 3] = 1.0  # phase/life
        vel = np.zeros((self.num_particles, 4), dtype=np.float32)
        col = np.zeros((self.num_particles, 4), dtype=np.float32)
        col[:, 0] = 0.2
        col[:, 1] = 0.6
        col[:, 2] = 1.0
        col[:, 3] = 0.8
        # Grid anchor: square grid of √N particles per side, scaled so
        # the full grid covers roughly ±1.6 on each axis (mirrors the
        # Rust GridWave extent of 1.20 in its own coordinate system).
        # w-channel stays 0 — reserved for future per-particle
        # parameters (e.g. individual phase offsets).
        anchor = np.zeros((self.num_particles, 4), dtype=np.float32)
        side = int(round(math.sqrt(self.num_particles)))
        if side * side < self.num_particles:
            side += 1
        # Cover ±extent on each axis, with a tiny inset so edge
        # particles aren't pinned at the visual border.
        extent = 1.55
        step = (2.0 * extent) / max(side - 1, 1)
        # Index each particle to a grid cell in raster order. We use
        # np.arange to get a deterministic layout — no random
        # shuffling, so the grid is dense and orderly like the Rust
        # version's Particle::base_x/base_y seeding.
        cell = np.arange(self.num_particles, dtype=np.int32)
        cx = cell % side
        cy = cell // side
        anchor[:, 0] = -extent + cx * step
        anchor[:, 1] = -extent + cy * step
        anchor[:, 2] = 0.0
        # Interleave to a single buffer in the spec's order:
        # pos, vel, col, anchor — 16 floats per particle, 64 bytes.
        interleaved = np.empty((self.num_particles, 16), dtype=np.float32)
        interleaved[:, 0:4] = pos
        interleaved[:, 4:8] = vel
        interleaved[:, 8:12] = col
        interleaved[:, 12:16] = anchor

        self.ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            interleaved.nbytes,
            interleaved,
            GL_DYNAMIC_DRAW,
        )
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo)
        _check_gl("SSBO upload")

        # VAO binds the same SSBO four times: pos@0, vel@16, col@32,
        # anchor@48. The render shader only reads pos/col; the anchor
        # is for the compute shader alone, but binding it as a vertex
        # attribute doesn't cost anything and keeps the layout in one
        # place.
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.ssbo)
        stride = SSBO_STRIDE
        for loc, off in [(0, 0), (1, 16), (2, 32), (3, 48)]:
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(off))
        glBindVertexArray(0)
        _check_gl("VAO setup")

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
        elif key == glfw.KEY_SPACE:
            self.auto_mode = False
            self.pattern_mode = (self.pattern_mode + 1) % len(self.mode_names)
            print(f"[Manual] {self.mode_names[self.pattern_mode]}")
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5):
            self.auto_mode = False
            self.pattern_mode = key - glfw.KEY_1
            print(f"[Manual] {self.mode_names[self.pattern_mode]}")
        elif key in (glfw.KEY_0, glfw.KEY_M):
            # 0 and M both return to auto mode (driven by Ollama phase).
            if not self.auto_mode:
                self.auto_mode = True
                print("[Auto] pattern_mode follows Ollama phase")
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            # Replicates Rust's `-` key (pattern 11, GridWave).
            self.auto_mode = False
            self.pattern_mode = 4
            print(f"[Manual] {self.mode_names[self.pattern_mode]}")

    def _on_resize(self, _window, width, height):
        """GLFW framebuffer-size callback. Update the viewport and the
        cached dimensions used for the projection aspect."""
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        glViewport(0, 0, self.width, self.height)

    # ─── Telemetry ───
    def _update_telemetry(self, _t):
        # GPU stats are polled by telemetry.py's background threads.
        self.gpu_usage, self.gpu_temp, self.vram_usage, _old_tps = \
            self.telemetry.snapshot()
        # Ollama stats come from the streaming driver. When in auto
        # mode we drive pattern_mode from the lifecycle phase; when
        # manual we leave it alone. The live eval tokens/sec replaces
        # whatever the old telemetry.py one-shot probe was reporting.
        # recent_tokens is a snapshot of (mono_time, hash) pairs the
        # driver filled in mid-stream; we use them to fire GPU
        # shockwaves whenever a new token arrived since the last
        # frame, and to derive a per-beat colour tint.
        phase, eval_tps, _prefill_tps, _eval_count, _pe_count, \
            recent_tokens, response_hue, response_text = \
            self.ollama.snapshot()
        self.tokens_per_sec = eval_tps
        self.response_hue = response_hue
        self.response_text = response_text
        # New tokens since last frame? Pop the freshest one within the
        # last 250ms (older bursts are stale and would distort the
        # visual). The render thread treats it as a single shockwave
        # per frame — coarser than per-token but plenty fast for the
        # eye at 60fps.
        self.token_burst_intensity = 0.0
        self.token_burst_origin = (0.0, 0.0, 0.0)
        now = time.monotonic()
        for ev_time, ev_hash in reversed(recent_tokens):
            if now - ev_time > 0.25:
                continue
            # Map hash to a point on a unit sphere: azimuth and
            # elevation come from different slices of the hash.
            az = (ev_hash & 0xFFFF) / 65535.0 * 6.2831853
            el = ((ev_hash >> 16) & 0xFFFF) / 65535.0 * 3.1415927 - 1.5707963
            r = _BURST_ORIGIN_RADIUS
            self.token_burst_origin = (
                r * math.cos(el) * math.sin(az),
                r * math.sin(el),
                r * math.cos(el) * math.cos(az),
            )
            # Intensity peaks at 1.0 when tokens arrive fast (>1/50ms),
            # decays with age of the burst.
            age_s = max(0.0, now - ev_time)
            self.token_burst_intensity = max(
                self.token_burst_intensity,
                math.exp(-age_s * 8.0),
            )
        if self.auto_mode:
            # Phase mapping: idle→Aizawa, prefill→Context Prefill,
            # generating→Harmonic Orbits, done→Harmonic Orbits (the
            # eval stats we just recorded are still relevant), error
            # falls through to Aizawa.
            new_mode = {
                "idle":       0,
                "prefill":    1,
                "generating": 2,
                "done":       2,
                "error":      0,
            }.get(phase, 0)
            if new_mode != self.pattern_mode:
                self.pattern_mode = new_mode
                # Only print on transitions to avoid log spam.
                print(f"[Auto: {phase}] {self.mode_names[self.pattern_mode]}")

    # ─── Main loop ───
    def run(self):
        self.init_gl()
        self.telemetry.start()  # spawn background polling threads
        self.ollama.start()     # spawn streaming Ollama heartbeat
        last_time = time.time()
        glClearColor(0.02, 0.02, 0.05, 1.0)
        frame_count = 0
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            dt = min(current_time - last_time, 0.033)  # clamp per spec
            last_time = current_time
            t = float(current_time)

            self._update_telemetry(t)

            # ── Compute pass ──
            glUseProgram(self.compute_program)
            for name, value in [
                ("u_time", t),
                ("u_delta_time", dt),
                ("u_pattern_mode", float(self.pattern_mode)),
                ("u_gpu_usage", self.gpu_usage),
                ("u_gpu_temp", self.gpu_temp),
                ("u_tokens_per_sec", self.tokens_per_sec),
                ("u_vram_usage", self.vram_usage),
                # Per-token shockwave: position on the unit sphere is
                # token-hash-derived, intensity decays with age. When
                # intensity == 0 the shader's contribution is a no-op.
                ("u_token_burst",
                 (self.token_burst_origin[0],
                  self.token_burst_origin[1],
                  self.token_burst_origin[2],
                  self.token_burst_intensity)),
                # 0..1 hue derived from rolling hash of the response
                # text. The shader uses this to rotate the per-mode
                # base palette so each beat looks distinct.
                ("u_response_hue", self.response_hue),
            ]:
                loc = glGetUniformLocation(self.compute_program, name)
                if name == "u_pattern_mode":
                    glUniform1i(loc, int(value))
                elif name == "u_token_burst":
                    glUniform4f(loc, *value)
                else:
                    glUniform1f(loc, value)
            groups = (self.num_particles + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
            glDispatchCompute(groups, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            _check_gl(f"compute dispatch frame {frame_count}")

            # ── Render pass ──
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.render_program)
            cam_x = 6.0 * math.cos(t * 0.2)
            cam_z = 6.0 * math.sin(t * 0.2)
            view = _look_at(
                np.array([cam_x, 2.5, cam_z], dtype=np.float32),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )
            proj = _perspective(45.0, self.width / max(1, self.height), 0.1, 100.0)
            model = np.identity(4, dtype=np.float32)
            for name, mat in [("u_model", model), ("u_view", view), ("u_projection", proj)]:
                loc = glGetUniformLocation(self.render_program, name)
                glUniformMatrix4fv(loc, 1, GL_TRUE, mat)
            loc = glGetUniformLocation(self.render_program, "u_tokens_per_sec")
            glUniform1f(loc, self.tokens_per_sec)
            glBindVertexArray(self.vao)
            glDrawArrays(GL_POINTS, 0, self.num_particles)
            _check_gl(f"render frame {frame_count}")

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            frame_count += 1
            if frame_count % 60 == 0:
                # On the first frame, also report whether telemetry
                # sources are live (helps debugging if nvidia-smi or
                # Ollama is unreachable).
                src = ""
                phase = self.ollama.snapshot()[0]
                # Log the prompt + a preview of the response whenever
                # it changes, so the user can confirm the rotation
                # works and *see* what the model is actually saying.
                current_prompt = self.ollama.prompt
                response_preview = self.response_text.replace("\n", " ")[:60]
                if (current_prompt, response_preview) != self._last_prompt_log:
                    self._last_prompt_log = (current_prompt, response_preview)
                    print(f"[prompt] {current_prompt}")
                    if response_preview:
                        print(f"[response] {response_preview}…")
                if frame_count == 60:
                    parts = []
                    if self.telemetry._has_nvidia:
                        parts.append("nvidia-smi")
                    if self.ollama.request_count > 0:
                        parts.append("ollama-stream")
                    src = f" [sources: {' + '.join(parts) or 'sine-wave stand-ins'}]"
                auto = "AUTO" if self.auto_mode else "MANUAL"
                print(f"[frame {frame_count}] {auto} phase={phase} "
                      f"mode={self.pattern_mode} "
                      f"hue={self.response_hue:.2f} "
                      f"burst=({self.token_burst_origin[0]:+.2f},"
                      f"{self.token_burst_origin[1]:+.2f},"
                      f"{self.token_burst_origin[2]:+.2f},"
                      f"{self.token_burst_intensity:.2f}) "
                      f"gpu={self.gpu_usage:.2f} temp={self.gpu_temp:.2f} "
                      f"tps={self.tokens_per_sec:.2f} "
                      f"vram={self.vram_usage:.2f}{src}")

        glfw.terminate()
        self.telemetry.stop()
        self.ollama.stop()
        print(f"[Exited cleanly after {frame_count} frames]")

if __name__ == "__main__":
    PointCloudEngine(num_particles=NUM_PARTICLES).run()
