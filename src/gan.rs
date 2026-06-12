//! GAN inference client — spawns a Python subprocess and communicates over
//! stdin/stdout using a simple binary protocol.
//!
//! The Python server (`pygan/gan_server.py`) reads z-vectors and writes back
//! raw RGB pixel buffers.  This module handles the subprocess lifecycle,
//! serialization, and exposes a blocking `generate` method.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Metadata sent by the Python server at startup.
#[derive(Debug, Clone, Copy)]
pub struct GanInfo {
    pub z_dim: u32,
    pub img_size: u32,
    pub img_channels: u32,
}

impl GanInfo {
    /// Total bytes of one output image (RGB, row-major).
    pub fn image_bytes(&self) -> usize {
        self.img_size as usize * self.img_size as usize * self.img_channels as usize
    }

    /// Total bytes of one input z-vector (f32).
    pub fn z_bytes(&self) -> usize {
        self.z_dim as usize * 4
    }
}

/// A running GAN inference server.
pub struct GanClient {
    child: Child,
    info: GanInfo,
}

impl GanClient {
    /// Spawn the Python GAN server.
    ///
    /// `pygan_dir` — path to the `pygan/` directory (must contain `gan_server.py`).
    /// `model`     — path to the `.pkl` model file (relative to pygan_dir or absolute).
    /// `trunc`     — truncation psi (0.0 = mean, 1.0 = full variety).
    pub fn spawn(pygan_dir: &Path, model: &Path, trunc: f32) -> Result<Self> {
        let venv_python = pygan_dir.join(".venv/bin/python");
        let script = pygan_dir.join("gan_server.py");

        if !venv_python.exists() {
            bail!(
                "Python venv not found at {}. Run `uv sync` in pygan/ first.",
                venv_python.display()
            );
        }
        if !script.exists() {
            bail!("gan_server.py not found at {}", script.display());
        }

        let model_path: PathBuf = if model.is_absolute() {
            model.to_path_buf()
        } else {
            pygan_dir.join(model)
        };

        let mut child = Command::new(&venv_python)
            .arg(&script)
            .arg("--network")
            .arg(&model_path)
            .arg("--trunc")
            .arg(format!("{trunc}"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Python logs go to terminal
            .spawn()
            .with_context(|| format!("failed to spawn {}", venv_python.display()))?;

        // Read 12-byte header: [z_dim: u32, img_size: u32, img_channels: u32]
        let stdout = child.stdout.as_mut().context("no stdout")?;
        let mut hdr = [0u8; 12];
        stdout
            .read_exact(&mut hdr)
            .context("failed to read GAN server header")?;

        let z_dim = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let img_size = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let img_channels = u32::from_le_bytes(hdr[8..12].try_into().unwrap());

        let info = GanInfo {
            z_dim,
            img_size,
            img_channels,
        };

        Ok(Self { child, info })
    }

    /// Model metadata (dimensions, resolution).
    pub fn info(&self) -> GanInfo {
        self.info
    }

    /// Send a z-vector and receive the generated image as raw RGB bytes.
    ///
    /// `z` must have exactly `info.z_dim` elements.
    /// Returns a buffer of `info.image_bytes()` bytes (HWC, row-major, u8).
    pub fn generate(&mut self, z: &[f32]) -> Result<Vec<u8>> {
        let info = self.info;
        assert_eq!(
            z.len(),
            info.z_dim as usize,
            "z-vector length mismatch: expected {}, got {}",
            info.z_dim,
            z.len()
        );

        let stdin = self.child.stdin.as_mut().context("no stdin")?;
        let z_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(z.as_ptr() as *const u8, z.len() * 4) };
        stdin
            .write_all(z_bytes)
            .context("failed to write z-vector")?;
        stdin.flush().context("failed to flush stdin")?;

        let stdout = self.child.stdout.as_mut().context("no stdout")?;
        let mut img = vec![0u8; info.image_bytes()];
        stdout
            .read_exact(&mut img)
            .context("failed to read image from GAN server")?;

        Ok(img)
    }

    /// Send a z-vector and write the generated image directly into the provided buffer.
    ///
    /// `buf` must be at least `info.image_bytes()` bytes long.
    pub fn generate_into(&mut self, z: &[f32], buf: &mut [u8]) -> Result<()> {
        let info = self.info;
        assert_eq!(z.len(), info.z_dim as usize);
        assert!(buf.len() >= info.image_bytes());

        let stdin = self.child.stdin.as_mut().context("no stdin")?;
        let z_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(z.as_ptr() as *const u8, z.len() * 4) };
        stdin.write_all(z_bytes)?;
        stdin.flush()?;

        let stdout = self.child.stdout.as_mut().context("no stdout")?;
        stdout.read_exact(&mut buf[..info.image_bytes()])?;

        Ok(())
    }
}

impl Drop for GanClient {
    fn drop(&mut self) {
        // Close stdin to signal the server to exit.
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

/// Random projection from a low-dimensional slider space to the GAN's z-space.
///
/// Each column of the projection matrix maps one slider axis to a direction in
/// z-space.  Sliders are normalized from MIDI [0, 127] to [-1, 1].
pub struct SliderProjection {
    /// Shape: [z_dim, num_sliders] stored in column-major order
    /// (each contiguous z_dim-length slice is one slider's direction).
    matrix: Vec<f32>,
    z_dim: usize,
    num_sliders: usize,
    /// Base z-vector (center of exploration).
    base_z: Vec<f32>,
}

impl SliderProjection {
    /// Create a new random projection.
    ///
    /// `z_dim` — dimensionality of the GAN's z-space (e.g. 512).
    /// `num_sliders` — number of slider axes (e.g. 8).
    /// `seed` — RNG seed for reproducibility.
    pub fn new(z_dim: usize, num_sliders: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let matrix: Vec<f32> = (0..z_dim * num_sliders)
            .map(|_| rng.normal())
            .collect();
        let base_z: Vec<f32> = (0..z_dim).map(|_| rng.normal()).collect();
        Self {
            matrix,
            z_dim,
            num_sliders,
            base_z,
        }
    }

    /// Re-randomize the projection direction for a single slider.
    pub fn rerandomize_slider(&mut self, slider_idx: usize, seed: u64) {
        assert!(slider_idx < self.num_sliders);
        let mut rng = SimpleRng::new(seed);
        let offset = slider_idx * self.z_dim;
        for i in 0..self.z_dim {
            self.matrix[offset + i] = rng.normal();
        }
    }

    /// Project slider values (MIDI 0..127) to a z-vector.
    ///
    /// Returns a z-vector of length `z_dim`.
    pub fn project(&self, sliders: &[u8]) -> Vec<f32> {
        assert!(sliders.len() >= self.num_sliders);
        let mut z = self.base_z.clone();

        for (s_idx, &val) in sliders.iter().take(self.num_sliders).enumerate() {
            // Normalize MIDI 0..127 → -1..1
            let t = (val as f32 / 127.0) * 2.0 - 1.0;
            let offset = s_idx * self.z_dim;
            for (zi, &mi) in z.iter_mut().zip(&self.matrix[offset..offset + self.z_dim]) {
                *zi += t * mi;
            }
        }

        z
    }

    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    pub fn num_sliders(&self) -> usize {
        self.num_sliders
    }
}

/// Minimal xorshift-based RNG with Box-Muller normal generation.
/// Avoids pulling in the `rand` crate for this simple use case.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1), // avoid 0
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slider_projection_dimensions() {
        let proj = SliderProjection::new(512, 8, 42);
        assert_eq!(proj.z_dim(), 512);
        assert_eq!(proj.num_sliders(), 8);
        assert_eq!(proj.matrix.len(), 512 * 8);
        assert_eq!(proj.base_z.len(), 512);
    }

    #[test]
    fn project_midpoint_returns_base() {
        let proj = SliderProjection::new(4, 2, 42);
        // Sliders at midpoint (63/64) map to t ≈ 0, so z ≈ base_z
        let sliders = [64u8, 64];
        let z = proj.project(&sliders);
        // t = (64/127)*2 - 1 = 0.00787..., nearly 0
        for (i, &zi) in z.iter().enumerate() {
            let diff = (zi - proj.base_z[i]).abs();
            assert!(diff < 0.1, "dim {i}: diff {diff} too large");
        }
    }

    #[test]
    fn project_extremes_differ() {
        let proj = SliderProjection::new(4, 2, 42);
        let z_low = proj.project(&[0, 0]);
        let z_high = proj.project(&[127, 127]);
        // Should be different
        let diff: f32 = z_low
            .iter()
            .zip(z_high.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "extremes should differ substantially");
    }

    #[test]
    fn rerandomize_changes_direction() {
        let mut proj = SliderProjection::new(4, 2, 42);
        let before: Vec<f32> = proj.matrix[0..4].to_vec();
        proj.rerandomize_slider(0, 999);
        let after: Vec<f32> = proj.matrix[0..4].to_vec();
        assert_ne!(before, after);
        // Slider 1 should be unchanged
        let s1_before: Vec<f32> = proj.matrix[4..8].to_vec();
        let proj2 = SliderProjection::new(4, 2, 42);
        assert_eq!(s1_before, proj2.matrix[4..8].to_vec());
    }

    #[test]
    fn rng_produces_varied_output() {
        let mut rng = SimpleRng::new(42);
        let vals: Vec<f32> = (0..100).map(|_| rng.normal()).collect();
        // Check it's not all zeros or identical
        let unique: std::collections::HashSet<u32> =
            vals.iter().map(|v| v.to_bits()).collect();
        assert!(unique.len() > 50, "RNG should produce varied output");
        // Check mean is roughly 0
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        assert!(
            mean.abs() < 0.5,
            "mean of 100 normals should be near 0, got {mean}"
        );
    }

    #[test]
    fn gan_info_byte_counts() {
        let info = GanInfo {
            z_dim: 512,
            img_size: 1024,
            img_channels: 3,
        };
        assert_eq!(info.z_bytes(), 512 * 4);
        assert_eq!(info.image_bytes(), 1024 * 1024 * 3);
    }
}
