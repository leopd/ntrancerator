//! Rendering: history texture, layers, and the wgpu/winit driver (spec §9).
//!
//! The math-only submodules ([`colormap`], [`mapping`], [`history`]) are pure
//! and always compiled so they can be unit tested without a GPU. The actual
//! `wgpu`/`winit` machinery lives in the feature-gated [`gpu`] submodule.

pub mod colormap;
pub mod history;
pub mod mapping;
pub mod monitor;

#[cfg(feature = "gui")]
pub mod gpu;
