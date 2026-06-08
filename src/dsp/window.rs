//! Analysis window functions (spec §7).
//!
//! Coefficients are computed once into a `Vec<f32>` and then reused for every
//! frame, so this is pure setup-time work with no per-frame allocation.

use crate::config::WindowKind;
use std::f32::consts::PI;

/// Compute window coefficients of length `n` for the given [`WindowKind`].
///
/// We use the "periodic" (DFT-even) form of each window — i.e. denominator `n`
/// rather than `n - 1` — which is the correct choice for spectral analysis
/// because it avoids a discontinuity when frames are laid end to end.
pub fn coefficients(kind: WindowKind, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f32;
    (0..n)
        .map(|i| {
            let x = i as f32 / nf; // in [0, 1)
            match kind {
                WindowKind::Hann => 0.5 - 0.5 * (2.0 * PI * x).cos(),
                WindowKind::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
                WindowKind::BlackmanHarris => {
                    const A0: f32 = 0.35875;
                    const A1: f32 = 0.48829;
                    const A2: f32 = 0.14128;
                    const A3: f32 = 0.01168;
                    A0 - A1 * (2.0 * PI * x).cos() + A2 * (4.0 * PI * x).cos()
                        - A3 * (6.0 * PI * x).cos()
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_singleton() {
        assert!(coefficients(WindowKind::Hann, 0).is_empty());
        assert_eq!(coefficients(WindowKind::Hann, 1), vec![1.0]);
    }

    #[test]
    fn hann_endpoints_and_center() {
        let w = coefficients(WindowKind::Hann, 1024);
        // Periodic Hann starts at 0 and peaks at 1.0 in the middle.
        assert!(w[0].abs() < 1e-6);
        assert!((w[512] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hamming_endpoints() {
        let w = coefficients(WindowKind::Hamming, 1024);
        // Hamming has a non-zero pedestal of 0.08 at the edge.
        assert!((w[0] - 0.08).abs() < 1e-6);
        assert!((w[512] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blackman_harris_is_bounded() {
        let w = coefficients(WindowKind::BlackmanHarris, 2048);
        for &c in &w {
            assert!((-0.01..=1.01).contains(&c), "coeff out of range: {c}");
        }
        // Peak near the middle is close to 1.0.
        assert!((w[1024] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn all_windows_have_requested_length() {
        for kind in [
            WindowKind::Hann,
            WindowKind::Hamming,
            WindowKind::BlackmanHarris,
        ] {
            assert_eq!(coefficients(kind, 777).len(), 777);
        }
    }
}
