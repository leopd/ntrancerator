# Real-Time GPU Audio Spectrogram Visualizer — Technical Specification

**Status:** Draft for implementation
**Audience:** Coding agent implementing Phase 1; human reviewers
**Language:** Rust
**Scope:** Phase 1 is a standalone real-time spectrogram. There is **no GAN and no ML** in this phase. Forward-compatibility notes exist only to avoid architectural dead-ends; they require **zero implementation work** now.

---

## 1. Purpose

Capture audio — from a decoded file (`.mp3`/`.wav`) or a live input device — transform it via a Short-Time Fourier Transform (STFT), and render it as a scrolling, color-mapped spectrogram full-screen-capable on the GPU using `wgpu`.

---

## 2. Target Environment

The app supports **two deployment targets**. Both run as **native Wayland clients**
via `winit` (no X11/XWayland involved); both render via `wgpu`'s Vulkan backend;
both reach PipeWire for audio through `cpal`'s ALSA backend.

### Target A — NVIDIA DGX Spark, headless over SSH (primary dev/run box)

| Property | Value |
|---|---|
| Hardware | NVIDIA DGX Spark (Grace Blackwell), NVIDIA GPU |
| CPU architecture | `aarch64` (ARMv8) |
| OS | DGX OS, based on Ubuntu 24.04 |
| Session | **Headless** — no desktop. App runs under **`cage`** (a single-app kiosk Wayland compositor) launched from an SSH session **as root** |
| Window mode | **Always fullscreen** — `cage` gives the app one fullscreen surface with no window manager, so "start windowed" and the `F` toggle are inert here |
| HDMI output | May enumerate under KMS as a generic connector name like `Unknown-1` rather than `HDMI-A-1` — **do not hardcode connector names** |

### Target B — Acer Nitro V15 laptop, desktop use

| Property | Value |
|---|---|
| Hardware | Intel iGPU + NVIDIA RTX dGPU, **no MUX switch** |
| Display wiring | Internal panel driven by the **Intel iGPU** (login + desktop); the **HDMI port is wired directly to the NVIDIA dGPU** and is where the app runs fullscreen |
| Session | Normal **GNOME Wayland**, extended desktop, internal panel set as primary |
| GPU selection | App must render on the **dGPU** (the GPU wired to HDMI) — request the high-performance adapter (see §9) |

### NVIDIA prerequisite (BOTH targets)

Kernel modesetting must be enabled: **`nvidia-drm.modeset=1`** on the kernel
command line via GRUB. The `/etc/modprobe.d` route may silently fail to apply, so
the **GRUB cmdline is the supported method**. Without it, no KMS connectors appear
and nothing renders. See the README for the exact steps and verification.

Implications:
- Target `aarch64` on Target A. `rustfft` provides NEON SIMD on this arch — no x86-only assumptions.
- `wgpu` selects the Vulkan backend; `winit` opens a native Wayland surface on both targets.
- `cpal`'s default ALSA backend reaches PipeWire via the ALSA compatibility layer. PipeWire also handles output-device sample-rate conversion, so explicit resampling is usually unnecessary.

### Deployment & Launch

Exact per-target invocations live in the **README** ("Deployment" section); the
summary:
- **Target A (DGX, headless):** stop `gdm3`, then `sudo LIBSEAT_BACKEND=builtin cage -- ./spectro …`. Root is required because a headless SSH session has no logind seat; `cage` runs the one app fullscreen and returns to the console on exit.
- **Target B (Nitro laptop):** run inside the GNOME Wayland session targeting the HDMI output, e.g. `./spectro --input live --monitor <hdmi-output-name> --fullscreen`. The app auto-selects the high-performance (dGPU) adapter.

---

## 3. Goals & Non-Goals

### Phase 1 goals
- Ingest audio from **either** a decoded file (mp3/wav) **or** a live input device (e.g., a USB sampler), selected via CLI.
- For file input, **play the audio** to the default output device while visualizing (toggleable).
- Compute a continuous STFT with configurable parameters, driven by the source's native sample rate.
- Render a smooth, scrolling, **log-frequency**, **inferno**-colored spectrogram, windowed with a fullscreen toggle.
- Clean module boundaries that don't preclude later work.

### Non-Goals (Phase 1)
- **No GAN, no ML inference, no image generation of any kind.**
- No recording/encoding to disk.
- No GUI chrome (menus, dialogs). Minimal keyboard controls only (§10).
- No network features.

---

## 4. High-Level Architecture

```
  SOURCE (one of):
   ┌─ File:  symphonia decode (mp3/wav) ──► f32 frames @ file rate
   │           │
   │           ├──► cpal OUTPUT stream (playback)   [unless --no-audio-out]
   │           └──► (tee) downmix to mono ──┐
   │                                         │
   └─ Live:  cpal INPUT stream (device) ──► downmix to mono ──┐
                                                              ▼
                                          rtrb SPSC ring buffer (lock-free, no alloc in callback)
                                                              │
                 (main / render thread, winit event loop)
                                                              ▼
   drain samples ──► sliding analysis buffer ──► STFT column producer
                                                      │  (window → FFT → magnitude → dB)
                                                      ▼
                                            spectrogram history texture (GPU, R32Float)
                                                      │
                                              wgpu render pass (log-freq map + inferno colormap)
                                                      ▼
                                              surface (windowed; F toggles fullscreen)
```

Both source types converge on a single contract: **push mono `f32` samples into the analysis ring buffer at real-time rate.** Everything downstream is source-agnostic.

Threads (Phase 1):
1. **Audio/callback thread(s)** — owned by `cpal`. Minimal work only: format-convert, downmix, push to ring (and, for file playback, also emit samples to the output device). No allocation, no locks, no logging.
2. **Main/render thread** — owns the `winit` event loop, `wgpu` device/surface, and the DSP. Each redraw: drain ring → produce any new STFT columns → upload → render.

> **Sync detail (file mode):** when playback is enabled, the **output stream's callback is the clock** — it pulls decoded frames to play and tees the same frames (downmixed) into the analysis ring, so audio and visuals stay aligned for free. With `--no-audio-out`, a paced feeder thread pushes samples into the ring at the file's real-time rate instead.

---

## 5. Data Flow & Timing

- Render is decoupled from audio. Surface uses **Fifo** present mode (vsync); rendering is capped to refresh rate.
- Each frame the producer emits however many STFT columns became available since the last frame (often 0–1). The history texture advances by that many columns.
- Sample rate is **source-driven** and may be anything from 8 kHz to 192 kHz+. Nothing hardcodes a rate. Ring-buffer capacity scales with rate (e.g., `max(fft_size * 4, sample_rate / 4)` samples ≈ ≥250 ms of headroom).

---

## 6. Audio Sources

**Module:** `audio` with a `Source` abstraction and two implementations plus optional playback.

### 6.1 Common contract
- Expose the **native sample rate** and channel count of the source.
- Deliver mono `f32` (downmix by averaging channels) into the analysis ring buffer.
- On error / device loss / EOF, signal the main thread cleanly (channel or shared flag); never panic in a callback.

### 6.2 `FileSource` (mp3/wav)
- Decode with **`symphonia`** (enable `mp3` and `wav`/`pcm` features; mp3 patents are expired). Convert decoded samples to `f32` at the file's native rate.
- Playback (default ON): open a `cpal` **output** stream; its callback pulls decoded frames (stereo preserved for listening) and tees a mono downmix into the analysis ring.
- `--no-audio-out`: skip the output stream; a paced feeder thread pushes mono samples into the ring at real-time rate.
- At end of file: stop feeding; leave the final display in place. (A `--loop` flag may be added later; out of scope now.)
- Rely on PipeWire for output device rate conversion; only add `rubato` resampling if a target rate is rejected (treat as optional/contingency).

### 6.3 `LiveSource` (device / USB)
- `cpal` **input** stream from the default device, or a named device via `--device`. A USB sampler is just a selectable input device — no special handling.
- Request `f32`; if unavailable, convert in the callback. Downmix to mono and push to the ring. Analysis only (no playback of live input).
- `--list-devices` prints available input devices and exits.

---

## 7. DSP / STFT Column Producer

**Module:** `dsp`

Stateful producer: ingests mono `f32`, emits completed spectrogram columns.

Pipeline per frame:
1. **Framing** — sliding buffer; emit a frame of `fft_size` samples advancing by `hop_size`.
2. **Windowing** — multiply by a precomputed window (default **Hann**). Coefficients computed once.
3. **FFT** — real-input FFT via `realfft` (over `rustfft`); plan built once and reused. Output: `fft_size/2 + 1` complex bins.
4. **Magnitude** — `sqrt(re² + im²)` per bin.
5. **dB** — `20 * log10(magnitude + epsilon)`.
6. **Emit** a column: reused `f32` buffer of length `fft_size/2 + 1`.

Normalization, dB floor/ceiling clamping, log-frequency mapping, and colormapping are done **in the shader** via uniforms so they're tunable live.

**Configurable parameters (Phase-1 defaults):**

| Parameter | Default | Notes / CLI |
|---|---|---|
| `fft_size` | 2048 | `--fft-size`; power of two. |
| `hop_size` | `fft_size/2` | `--hop`; smaller = smoother scroll. |
| `window` | Hann | `--window hann\|hamming\|blackman-harris`. |
| `sample_rate` | source-native | Not hardcoded; from the source. |
| `db_floor` | -100.0 | `--db-floor`; shader uniform. |
| `db_ceiling` | 0.0 | `--db-ceiling`; shader uniform. |

All buffers preallocated; no per-frame heap allocation in steady state.

---

## 8. Spectrogram History & GPU Upload

**Module:** `render::spectrogram`

- GPU texture: `width = history_columns` × `height = num_bins` (`num_bins = fft_size/2 + 1`), format **`R32Float`** (dB values).
- **Ring/cursor** upload: keep a write cursor `x`; each new column is written to texture column `x` via `queue.write_texture` (1-column region), then `x = (x + 1) % width`. No full-texture shifting.
- Fragment shader receives `x` as a uniform and applies a UV offset so **the newest column is at the right edge and history scrolls left** (horizontal time axis).
- `history_columns` default tied to surface width (one column per horizontal pixel) or a fixed value (e.g., 2048); configurable.

---

## 9. Rendering (`wgpu` + `winit`)

**Module:** `render`

- `winit` ≥ 0.30 `ApplicationHandler` event loop. Pin `wgpu` and `winit` to mutually compatible versions (they interlock; verify at build time).
- Init: instance → surface → adapter (Vulkan) → device + queue → surface config.
- **Adapter selection:** request **`PowerPreference::HighPerformance`** so the discrete NVIDIA GPU is chosen on hybrid laptops (Target B) and generally. On Target B this also avoids a cross-GPU copy, since the HDMI port is on the dGPU.
- **Window mode:**
  - **Target B (desktop):** start **windowed**; `F` toggles **borderless fullscreen** (trivial in `winit`; exclusive fullscreen remains a later option). Handle resize / surface-lost.
  - **Target A (`cage`):** the app is **always fullscreen** on a single surface. Default to fullscreen there and **do not assume a window manager exists** — "start windowed" and the `F` toggle are inert.
  - **Output selection:** use `--monitor <name|index>` (§10) to pick which monitor receives the borderless-fullscreen surface; default to the external/HDMI output when present, else primary. `winit` exposes monitor names/positions to match against.
- Present mode: **Fifo** (vsync) default; constant to switch to Mailbox/Immediate for low-latency experiments.

### Render pipeline
- One full-screen triangle/quad.
- Fragment shader (WGSL):
  1. Map fragment UV → (time, frequency). **Time axis:** horizontal, with the cursor UV offset (newest at right). **Frequency axis:** vertical, **logarithmic** between `freq_min` and `freq_max`. For each output row: `freq = freq_min * (freq_max/freq_min)^row_norm`; `bin = freq * fft_size / sample_rate`; sample the texture at `bin` with linear interpolation.
  2. Sample `R32Float` history → dB.
  3. Normalize: `t = clamp((db - db_floor) / (db_ceiling - db_floor), 0, 1)`.
  4. Colormap `t` → RGB (default **inferno**).
- **Frequency display range:** `--freq-min` (default 20 Hz), `--freq-max` (default Nyquist = `sample_rate/2`). Clamp `freq_min` ≥ small positive value to keep the log well-defined.
- **Colormap:** implement as a 256×1 LUT texture (easy swapping) or an analytic WGSL approximation. Provide inferno (default) plus magma, viridis, grayscale via `--colormap`.

### Layer structure (good hygiene, not GAN work)
Render through an ordered list of layers; Phase 1 ships exactly **one** layer (the spectrogram). This is just clean structure and adds no extra Phase-1 work. Do **not** build compositing/blending machinery beyond what one layer needs.

---

## 10. Configuration, Controls, Errors

### CLI (use `clap`)
```
spectro [OPTIONS]

Input:
  --input <live|file>     Source type (default: live)
  --file <PATH>           .mp3/.wav path (required if --input file)
  --device <NAME>         Capture device for live input (default: system default)
  --list-devices          List input devices and exit
  --no-audio-out          File mode: visualize without playing audio

DSP:
  --fft-size <N>          (default 2048)
  --hop <N>               (default fft-size/2)
  --window <hann|hamming|blackman-harris>   (default hann)
  --db-floor <DB>         (default -100)
  --db-ceiling <DB>       (default 0)

Display:
  --freq-min <HZ>         (default 20)
  --freq-max <HZ>         (default Nyquist)
  --colormap <inferno|magma|viridis|gray>   (default inferno)
  --fullscreen            Start fullscreen (default: windowed; always on under cage)
  --monitor <NAME|INDEX>  Output to target for fullscreen
                          (default: external/HDMI output if present, else primary)
```

`--monitor` makes the target output deterministic for borderless fullscreen: the
app matches the value against the monitor names/indices `winit` enumerates. This
matters on Target B (the app must land on the HDMI output wired to the dGPU) and
is harmless on Target A (`cage` only ever exposes the one connected output).

### Keyboard (minimal)
`Esc`/`Q` quit · `F` toggle fullscreen (Target B only; inert under `cage`) · `[` / `]` adjust dB floor · `C` cycle colormap.

### Errors
`anyhow` at the top level (+ `thiserror` for typed lib errors if useful). Audio device/file errors and GPU surface-lost must degrade gracefully (log + recover/exit), never panic in a callback.

---

## 11. Milestones

- **M1 — Sources:** `--input live` and `--input file` both deliver mono samples into the ring; file mode plays audio out and stays in sync. Verify via RMS / dropped-sample counters. No graphics.
- **M2 — DSP:** STFT producer with unit tests on synthetic signals (known sine → energy in the expected bin; verify at multiple sample rates incl. 44.1k/48k/192k).
- **M3 — Render MVP (Phase 1 acceptance):** windowed full-screen-toggle window; scrolling log-frequency inferno spectrogram from both source types; `F` fullscreen; minimal controls.
- **M4 — Polish:** window-function and colormap switching, freq-range controls, exclusive-fullscreen option.

---

## 12. Forward Compatibility (NO Phase-1 work)

We intend to add audio-driven generative imagery later. To avoid dead-ends, only two cheap habits apply now, both of which are just good architecture:
1. Keep the **DSP producer self-contained** so a future feature tap (band energies, onset, etc.) can be added without restructuring.
2. Keep **rendering organized as layers** (one layer today) so additional visual layers can be added later.

No GAN, ML, interop, or feature-extraction code is to be written in Phase 1.

---

## 13. Suggested Project Structure

```
src/
  main.rs                 // wiring, event loop entry, CLI parse
  config.rs               // Config struct + defaults + clap
  audio/mod.rs            // Source trait, ring producer, downmix
  audio/file.rs           // symphonia decode + playback tee / paced feeder
  audio/live.rs           // cpal input device capture
  dsp/mod.rs              // STFT column producer
  dsp/window.rs           // window functions
  render/mod.rs           // wgpu device/surface, frame loop, layer list
  render/spectrogram.rs   // history texture, ring upload, spectrogram layer
  render/shaders/spectrogram.wgsl
```

## 14. Dependencies (verify current compatible versions at build time)

- `cpal` — audio input + output
- `symphonia` — mp3/wav decoding (features: `mp3`, `wav`/`pcm`)
- `rtrb` — lock-free SPSC ring buffer
- `rustfft` + `realfft` — real-input FFT (NEON on aarch64)
- `wgpu` — GPU rendering (Vulkan)
- `winit` (≥ 0.30 `ApplicationHandler`) — windowing/fullscreen, version-matched to `wgpu`
- `bytemuck` — POD casts for GPU buffers
- `clap` — CLI
- `anyhow` (+ optional `thiserror`) — errors
- `rubato` — **optional/contingency** resampler for playback only if PipeWire can't match a rate

---

## 15. Acceptance Criteria (Phase 1 / M3)

- `--input file --file song.mp3` plays the file and shows a synchronized scrolling spectrogram; `--no-audio-out` shows it silently.
- `--input live --device <USB sampler>` shows a live spectrogram; `--list-devices` works.
- A pure tone produces a correctly-placed horizontal band on the **log** frequency axis that tracks pitch; **inferno** coloring.
- Smooth scrolling at refresh rate; no visible per-frame texture-shift cost. On Target B, `F` toggles fullscreen and `--monitor <name|index>` lands the fullscreen surface on the chosen (HDMI/dGPU) output; on Target A the app comes up fullscreen under `cage`.
- Works at 44.1 kHz, 48 kHz, and 192 kHz sources.
- ≥10 min run: stable memory, zero steady-state per-frame allocations (spot-check), dropped-sample counter at zero under normal load.
