//! Frequency-axis and dB normalization math (spec §9).
//!
//! These functions are the CPU-side mirror of the arithmetic performed in the
//! fragment shader. Keeping them here makes the mapping unit-testable even
//! though the shader is what actually runs on the GPU at draw time.

/// Normalize a dB value to `[0, 1]` against the floor/ceiling window (spec §9).
///
/// `t = clamp((db - floor) / (ceiling - floor), 0, 1)`.
pub fn normalize_db(db: f32, db_floor: f32, db_ceiling: f32) -> f32 {
    let span = db_ceiling - db_floor;
    if span <= 0.0 {
        return 0.0;
    }
    ((db - db_floor) / span).clamp(0.0, 1.0)
}

/// Frequency (Hz) displayed at a normalized vertical position `row_norm`
/// (`0.0` = bottom = `freq_min`, `1.0` = top = `freq_max`) on the log axis:
/// `freq = freq_min * (freq_max / freq_min)^row_norm` (spec §9).
pub fn log_freq_for_row(row_norm: f32, freq_min: f32, freq_max: f32) -> f32 {
    freq_min * (freq_max / freq_min).powf(row_norm.clamp(0.0, 1.0))
}

/// Fractional FFT bin index for a frequency (spec §9):
/// `bin = freq * fft_size / sample_rate`.
pub fn freq_to_bin(freq: f32, fft_size: usize, sample_rate: u32) -> f32 {
    freq * fft_size as f32 / sample_rate as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_endpoints_and_clamping() {
        assert_eq!(normalize_db(-100.0, -100.0, 0.0), 0.0);
        assert_eq!(normalize_db(0.0, -100.0, 0.0), 1.0);
        assert_eq!(normalize_db(-50.0, -100.0, 0.0), 0.5);
        // Out-of-range values clamp.
        assert_eq!(normalize_db(-200.0, -100.0, 0.0), 0.0);
        assert_eq!(normalize_db(20.0, -100.0, 0.0), 1.0);
        // Degenerate window does not divide by zero.
        assert_eq!(normalize_db(5.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn log_freq_hits_endpoints() {
        assert!((log_freq_for_row(0.0, 20.0, 20_000.0) - 20.0).abs() < 1e-3);
        assert!((log_freq_for_row(1.0, 20.0, 20_000.0) - 20_000.0).abs() < 1e-1);
    }

    #[test]
    fn log_freq_midpoint_is_geometric_mean() {
        // On a log axis the centre row is the geometric mean of the endpoints.
        let f = log_freq_for_row(0.5, 20.0, 20_000.0);
        let geo = (20.0f32 * 20_000.0).sqrt();
        assert!((f - geo).abs() < 1e-1, "{f} vs {geo}");
    }

    #[test]
    fn log_freq_is_monotonic_increasing() {
        let mut prev = 0.0;
        for i in 0..=100 {
            let f = log_freq_for_row(i as f32 / 100.0, 20.0, 20_000.0);
            assert!(f > prev, "not increasing at {i}: {f} <= {prev}");
            prev = f;
        }
    }

    #[test]
    fn freq_to_bin_matches_dsp_inverse() {
        // bin spacing at 48k/2048 is 23.4375 Hz; bin 64 -> 1500 Hz.
        assert!((freq_to_bin(1500.0, 2048, 48_000) - 64.0).abs() < 1e-3);
        assert!((freq_to_bin(0.0, 2048, 48_000)).abs() < 1e-6);
    }
}
