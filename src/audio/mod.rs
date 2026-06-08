//! Audio sources and the shared mono contract (spec §6).
//!
//! Every source — file or live — converges on the same contract: push mono
//! `f32` samples into the analysis ring buffer at real-time rate. This module
//! holds the source-agnostic pieces (format description, channel downmix, ring
//! sizing) plus the pure-Rust file decoder. The `cpal`-backed live capture and
//! file playback live in feature-gated submodules.

pub mod decode;

#[cfg(feature = "playback")]
pub mod live;
#[cfg(feature = "playback")]
pub mod playback;

use rtrb::{Consumer, Producer, RingBuffer};

/// Native format reported by a source.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

/// Common interface exposed by every audio source (spec §6.1).
///
/// The actual sample delivery is push-based (into the ring buffer) and differs
/// between sources, so the trait only carries the metadata that downstream DSP
/// needs to interpret those samples.
pub trait Source {
    /// Native sample rate of the source, in Hz.
    fn sample_rate(&self) -> u32;
    /// Native channel count of the source before downmix.
    fn channels(&self) -> u16;

    fn format(&self) -> AudioFormat {
        AudioFormat {
            sample_rate: self.sample_rate(),
            channels: self.channels(),
        }
    }
}

/// A [`Source`] that also exposes the consumer end of its analysis ring, so the
/// render loop can drain mono samples regardless of which source produced them.
pub trait RingSource: Source {
    fn consumer(&mut self) -> &mut Consumer<f32>;
}

/// Downmix interleaved multi-channel `f32` to mono by averaging channels,
/// appending to `out` (spec §6.1).
///
/// A trailing partial frame (fewer than `channels` samples) is ignored. With
/// `channels <= 1` the input is copied through unchanged.
pub fn downmix_to_mono(interleaved: &[f32], channels: u16, out: &mut Vec<f32>) {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let inv = 1.0 / ch as f32;
    for frame in interleaved.chunks_exact(ch) {
        let sum: f32 = frame.iter().sum();
        out.push(sum * inv);
    }
}

/// Recommended analysis ring-buffer capacity in samples (spec §5):
/// `max(fft_size * 4, sample_rate / 4)` ≈ at least 250 ms of headroom.
pub fn ring_capacity(fft_size: usize, sample_rate: u32) -> usize {
    (fft_size * 4).max((sample_rate / 4) as usize)
}

/// Create the lock-free SPSC analysis ring sized via [`ring_capacity`].
pub fn analysis_ring(fft_size: usize, sample_rate: u32) -> (Producer<f32>, Consumer<f32>) {
    RingBuffer::<f32>::new(ring_capacity(fft_size, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_averages_channels() {
        let mut out = Vec::new();
        // L/R interleaved: (1,-1) -> 0, (0.5,0.5) -> 0.5.
        downmix_to_mono(&[1.0, -1.0, 0.5, 0.5], 2, &mut out);
        assert_eq!(out, vec![0.0, 0.5]);
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let mut out = Vec::new();
        downmix_to_mono(&[0.1, 0.2, 0.3], 1, &mut out);
        assert_eq!(out, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn downmix_ignores_partial_trailing_frame() {
        let mut out = Vec::new();
        // 5 samples, 2 channels: last lone sample is dropped.
        downmix_to_mono(&[1.0, 1.0, 2.0, 2.0, 9.0], 2, &mut out);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn downmix_appends_to_existing() {
        let mut out = vec![42.0];
        downmix_to_mono(&[1.0, 3.0], 2, &mut out);
        assert_eq!(out, vec![42.0, 2.0]);
    }

    #[test]
    fn ring_capacity_takes_the_larger_term() {
        // fft term dominates at low rates...
        assert_eq!(ring_capacity(2048, 8_000), 2048 * 4);
        // ...rate term dominates at high rates.
        assert_eq!(ring_capacity(2048, 192_000), 48_000);
    }

    #[test]
    fn analysis_ring_has_expected_slots() {
        let (prod, _cons) = analysis_ring(2048, 48_000);
        assert_eq!(prod.slots(), ring_capacity(2048, 48_000));
    }
}
