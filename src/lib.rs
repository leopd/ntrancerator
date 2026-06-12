//! N-Trancerator: a real-time GPU audio spectrogram visualizer (Phase 1).
//!
//! See `specs/spectrogram-visualizer-spec.md`. The crate is split into a pure,
//! always-compiled core (configuration, DSP, file decoding, and the render
//! math) and feature-gated platform layers:
//!
//! - `playback` — `cpal` live capture and file audio output.
//! - `gui` — `wgpu`/`winit` GPU rendering and windowing.
//!
//! Keeping the core free of device/GPU dependencies means the bulk of the logic
//! is exercised by fast, deterministic unit and e2e tests that run headlessly.

pub mod audio;
pub mod config;
pub mod dsp;
pub mod gan;
pub mod render;

pub use config::Config;
