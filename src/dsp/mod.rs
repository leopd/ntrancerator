//! STFT column producer (spec §7).
//!
//! A stateful, allocation-free-in-steady-state pipeline that ingests mono `f32`
//! samples and emits completed spectrogram columns of dB magnitudes. The FFT
//! plan, window coefficients, and every scratch buffer are built once at
//! construction; [`StftProducer::process`] performs no heap allocation.

pub mod window;

use crate::config::WindowKind;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::VecDeque;
use std::sync::Arc;

/// Added inside the `log10` to keep dB finite for silent (zero-magnitude) bins.
const DB_EPSILON: f32 = 1e-10;

/// Converts a stream of mono samples into a stream of dB-magnitude columns.
pub struct StftProducer {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,

    /// Samples received but not yet consumed by a completed frame.
    pending: VecDeque<f32>,

    // --- reused scratch (no per-frame allocation) ---
    frame: Vec<f32>,          // length fft_size: windowed real input
    spectrum: Vec<Complex32>, // length num_bins: complex output
    scratch: Vec<Complex32>,  // realfft internal scratch
    column: Vec<f32>,         // length num_bins: reused dB output
}

impl StftProducer {
    /// Build a producer for the given frame/hop/window and source sample rate.
    ///
    /// # Panics
    /// Panics if `fft_size < 2`, `hop_size` is not in `1..=fft_size`, which the
    /// CLI layer rejects up front via [`crate::config::Config::validate`].
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        window_kind: WindowKind,
        sample_rate: u32,
    ) -> Self {
        assert!(fft_size >= 2, "fft_size must be >= 2");
        assert!(
            (1..=fft_size).contains(&hop_size),
            "hop_size must be in 1..=fft_size"
        );

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let frame = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        let num_bins = spectrum.len();
        debug_assert_eq!(num_bins, fft_size / 2 + 1);

        Self {
            window: window::coefficients(window_kind, fft_size),
            fft,
            fft_size,
            hop_size,
            sample_rate,
            pending: VecDeque::with_capacity(fft_size * 2),
            frame,
            spectrum,
            scratch,
            column: vec![0.0; num_bins],
        }
    }

    /// Number of frequency bins in each emitted column (`fft_size / 2 + 1`).
    pub fn num_bins(&self) -> usize {
        self.column.len()
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Center frequency (Hz) of FFT bin `bin`.
    pub fn bin_to_freq(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate as f32 / self.fft_size as f32
    }

    /// Push `samples` and invoke `emit` once per newly completed column.
    ///
    /// `emit` receives a borrowed slice of length [`num_bins`](Self::num_bins)
    /// holding dB magnitudes; the slice is reused across calls, so the callback
    /// must copy out anything it needs to retain. Returns the number of columns
    /// emitted. Performs no heap allocation once warmed up.
    pub fn process(&mut self, samples: &[f32], mut emit: impl FnMut(&[f32])) -> usize {
        self.pending.extend(samples.iter().copied());

        let mut emitted = 0;
        while self.pending.len() >= self.fft_size {
            self.compute_column();
            emit(&self.column);
            emitted += 1;
            // Advance the sliding window by one hop.
            self.pending.drain(..self.hop_size);
        }
        emitted
    }

    /// Window the oldest `fft_size` pending samples, FFT, and fill `self.column`
    /// with `20*log10(|X| + eps)`.
    fn compute_column(&mut self) {
        for i in 0..self.fft_size {
            // VecDeque indexing is O(1); no allocation.
            self.frame[i] = self.pending[i] * self.window[i];
        }
        self.fft
            .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
            .expect("realfft input/output lengths are fixed at construction");

        for (out, c) in self.column.iter_mut().zip(self.spectrum.iter()) {
            let mag = c.norm();
            *out = 20.0 * (mag + DB_EPSILON).log10();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate `n` samples of a unit-amplitude sine at `freq` Hz.
    fn sine(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    /// Index of the maximum-magnitude bin in a column.
    fn peak_bin(column: &[f32]) -> usize {
        column
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    #[test]
    fn column_length_is_num_bins() {
        let mut p = StftProducer::new(2048, 1024, WindowKind::Hann, 48_000);
        assert_eq!(p.num_bins(), 1025);
        let mut len = 0;
        p.process(&sine(1000.0, 48_000, 4096), |col| len = col.len());
        assert_eq!(len, 1025);
    }

    #[test]
    fn pure_tone_lands_in_expected_bin_48k() {
        let sr = 48_000;
        let fft = 2048;
        // Choose a frequency that sits exactly on a bin center to avoid leakage:
        // bin spacing = sr/fft = 23.4375 Hz; bin 64 -> 1500 Hz.
        let target_bin = 64;
        let freq = target_bin as f32 * sr as f32 / fft as f32;
        let mut p = StftProducer::new(fft, fft / 2, WindowKind::Hann, sr);

        let mut last = vec![];
        p.process(&sine(freq, sr, fft * 4), |col| last = col.to_vec());
        assert_eq!(peak_bin(&last), target_bin);
        assert!((p.bin_to_freq(target_bin) - freq).abs() < 1e-3);
    }

    #[test]
    fn pure_tone_at_multiple_sample_rates() {
        // Spec M2: verify at 44.1k, 48k, and 192k.
        for sr in [44_100u32, 48_000, 192_000] {
            let fft = 4096;
            let target_bin = 100;
            let freq = target_bin as f32 * sr as f32 / fft as f32;
            let mut p = StftProducer::new(fft, fft / 2, WindowKind::Hann, sr);
            let mut last = vec![];
            p.process(&sine(freq, sr, fft * 4), |col| last = col.to_vec());
            let pk = peak_bin(&last);
            // Allow the immediate neighbours to absorb any tiny spectral leakage.
            assert!(
                (pk as i32 - target_bin).abs() <= 1,
                "sr={sr}: peak bin {pk} != target {target_bin}"
            );
        }
    }

    #[test]
    fn emits_expected_number_of_columns() {
        let fft = 1024;
        let hop = 256;
        let mut p = StftProducer::new(fft, hop, WindowKind::Hann, 44_100);
        // With N samples, columns = floor((N - fft) / hop) + 1.
        let n = 4096;
        let cols = p.process(&sine(440.0, 44_100, n), |_| {});
        assert_eq!(cols, (n - fft) / hop + 1);
    }

    #[test]
    fn streaming_in_chunks_matches_one_shot() {
        // Feeding the same signal split across calls must yield identical columns.
        let fft = 512;
        let hop = 128;
        let sig = sine(880.0, 44_100, 4096);

        let mut whole = StftProducer::new(fft, hop, WindowKind::Hann, 44_100);
        let mut cols_whole: Vec<Vec<f32>> = vec![];
        whole.process(&sig, |c| cols_whole.push(c.to_vec()));

        let mut chunked = StftProducer::new(fft, hop, WindowKind::Hann, 44_100);
        let mut cols_chunked: Vec<Vec<f32>> = vec![];
        for chunk in sig.chunks(37) {
            chunked.process(chunk, |c| cols_chunked.push(c.to_vec()));
        }

        assert_eq!(cols_whole.len(), cols_chunked.len());
        for (a, b) in cols_whole.iter().zip(&cols_chunked) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn silence_is_near_db_floor() {
        let mut p = StftProducer::new(1024, 512, WindowKind::Hann, 48_000);
        let mut last = vec![];
        p.process(&vec![0.0f32; 4096], |c| last = c.to_vec());
        // 20*log10(eps) for eps=1e-10 is -200 dB; everything should be very low.
        for &db in &last {
            assert!(db < -150.0, "silent bin not near floor: {db}");
        }
    }
}
