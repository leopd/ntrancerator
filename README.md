# N-Trancerator

A DJ visualization tool. **Phase 1** is a real-time GPU audio **spectrogram
visualizer**: it captures audio (from a decoded `.mp3`/`.wav` file or a live
input device), runs a Short-Time Fourier Transform, and renders a scrolling,
log-frequency, inferno-colored spectrogram on the GPU via `wgpu`.

See [`specs/spectrogram-visualizer-spec.md`](specs/spectrogram-visualizer-spec.md)
for the full specification.

## Prerequisites

- [Rust](https://rustup.rs/) stable (`rustup default stable`).
- To run the full application (default build) you also need:
  - **ALSA dev headers** for audio I/O: `sudo apt install libasound2-dev`
    (runtime uses PipeWire's ALSA compatibility layer).
  - A **Vulkan-capable GPU** and driver, plus a display server (Wayland or X11).

The library core (DSP, decoding, config, render math) builds with no system
dependencies and is fully covered by the test suite.

## Build & run

```sh
# Build everything (audio I/O + GPU rendering are on by default).
cargo build --release          # binary: target/release/spectro

# Live input (default) — visualize the default capture device:
cargo run --release

# List input devices and exit:
cargo run --release -- --list-devices

# File input — play an mp3/wav and show a synchronized spectrogram:
cargo run --release -- --input file --file song.mp3

# File input without audio playback (silent visualization):
cargo run --release -- --input file --file song.wav --no-audio-out

# Start fullscreen on a specific output (e.g. the HDMI monitor) with a colormap:
cargo run --release -- --fullscreen --monitor HDMI-A-1 --colormap viridis
```

Run `cargo run -- --help` for the complete option list (FFT size, hop, window,
dB floor/ceiling, frequency range, `--monitor`, etc.).

### Keyboard controls

`Esc`/`Q` quit · `F` toggle fullscreen (desktop only — inert under `cage`) ·
`[` / `]` lower/raise the dB floor · `C` cycle colormap.

## Deployment

The app runs as a **native Wayland client** and renders via Vulkan. It targets
two hardware setups; both share the one-time NVIDIA setup below. System tools
(`cage`, `kmscube`) are installed via `apt` — they are **not** Cargo dependencies.

### One-time NVIDIA setup (both targets)

Kernel modesetting must be enabled or **no KMS connectors appear and nothing
renders**. The `/etc/modprobe.d` route can silently fail to apply, so set it on
the kernel command line via GRUB. Edit `/etc/default/grub` and append
`nvidia-drm.modeset=1` **inside the quotes** of `GRUB_CMDLINE_LINUX_DEFAULT`, then:

```sh
sudo update-grub
sudo reboot
```

Verify after reboot:

```sh
cat /sys/module/nvidia_drm/parameters/modeset            # expect: Y
for p in /sys/class/drm/*/status; do echo "$p -> $(cat $p)"; done
```

### Target A — DGX Spark, headless over SSH

No desktop session. A kiosk Wayland compositor (`cage`) grabs the display and
runs the app fullscreen on the single connected output:

```sh
sudo apt-get install -y cage
sudo systemctl stop gdm3
sudo LIBSEAT_BACKEND=builtin cage -- ./spectro --input file --file song.mp3
```

- **Root is required** because a headless SSH session has no logind seat.
- `LIBSEAT_BACKEND=builtin` lets `cage` grab the display directly (no seat manager).
- `cage` runs the one app fullscreen and returns to the console on exit.
- The HDMI output may enumerate under KMS as a generic name like `Unknown-1`
  rather than `HDMI-A-1`; `--monitor` is optional here since `cage` only exposes
  the one output.

### Target B — Acer Nitro V15 laptop

Hybrid graphics with **no MUX switch**: the internal panel is driven by the Intel
iGPU, while the **HDMI port is wired to the NVIDIA dGPU** — so the app must run
fullscreen on the HDMI output and on the dGPU.

- Install the NVIDIA proprietary driver and apply the modeset step above.
- In GNOME display settings: **extend** displays and set the **internal panel as
  primary**.
- Run the app targeting the HDMI output:

```sh
./spectro --input live --monitor <hdmi-output-name> --fullscreen
```

The app requests the **high-performance adapter**, so it selects the dGPU
automatically (which is also the GPU wired to HDMI, avoiding a cross-GPU copy).

### Optional smoke test (both targets)

Confirm the KMS/GPU display path independently of the app before launching it:

```sh
sudo systemctl stop gdm3        # headless (Target A) only
sudo apt-get install -y kmscube
kmscube                          # a spinning cube means the KMS/GPU path works
```

## Tests

The DSP, file decoding, configuration, and render math are exercised by unit
tests plus an **end-to-end audio test** that synthesizes a `.wav`, decodes it
through `symphonia`, runs the STFT, and asserts the tone lands in the expected
frequency bin.

```sh
# Fast, dependency-free core suite (no ALSA/GPU needed) — recommended:
cargo test --no-default-features

# Full suite (also compiles the GPU/audio layers):
cargo test
```

The e2e test lives in [`tests/e2e_audio.rs`](tests/e2e_audio.rs).

## Lint & format

```sh
cargo clippy --all-targets
cargo fmt
```

## Architecture

The crate is a library (`ntrancerator`) plus a thin binary (`spectro`):

| Module | Responsibility |
|---|---|
| `config` | CLI parsing (`clap`), defaults, validation |
| `dsp` | STFT column producer + window functions (`realfft`) |
| `audio` | `Source` trait, mono downmix, ring buffer, `symphonia` decode |
| `audio::live` / `audio::playback` | `cpal` capture / file playback (feature `playback`) |
| `render::{mapping,colormap,history}` | pure, testable shader-mirror math |
| `render::gpu` | `wgpu`/`winit` driver + WGSL shader (feature `gui`) |

Cargo features `playback` and `gui` (both on by default) gate the platform
layers, so the testable core compiles and runs headlessly.

> **Note:** GPU rendering and live/playback audio require a Wayland surface and
> audio/Vulkan devices, so they can't run in a CI environment with no display at
> all; the application degrades gracefully (logs an error, no panic) when they're
> absent. ("Headless over SSH" in [Deployment](#deployment) is not this case —
> `cage` provides the Wayland surface the app renders into.)
