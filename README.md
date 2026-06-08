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

# Start fullscreen with a different colormap:
cargo run --release -- --fullscreen --colormap viridis
```

Run `cargo run -- --help` for the complete option list (FFT size, hop, window,
dB floor/ceiling, frequency range, etc.).

### Keyboard controls

`Esc`/`Q` quit · `F` toggle fullscreen · `[` / `]` lower/raise the dB floor ·
`C` cycle colormap.

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

> **Note:** GPU rendering and live/playback audio require a display server and
> audio/Vulkan devices, so they can't run in a headless CI environment; the
> application degrades gracefully (logs an error, no panic) when they're absent.
