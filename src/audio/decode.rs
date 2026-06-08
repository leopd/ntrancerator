//! File decoding via `symphonia` (spec §6.2).
//!
//! Phase 1 decodes the whole file into memory up front (songs comfortably fit),
//! which keeps both the playback feeder and the test path simple. This is pure
//! Rust with no device dependency, so the full decode path is exercised by the
//! e2e test.

use super::downmix_to_mono;
use anyhow::{anyhow, Context, Result};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// A fully decoded audio file: interleaved `f32` at the file's native rate.
#[derive(Clone, Debug)]
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved samples, length is a multiple of `channels`.
    pub interleaved: Vec<f32>,
}

impl DecodedAudio {
    /// Number of sample frames (per-channel length).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.interleaved.len() / self.channels as usize
        }
    }

    /// Total duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    /// Mono downmix of the whole buffer (averaging channels).
    pub fn to_mono(&self) -> Vec<f32> {
        let mut mono = Vec::with_capacity(self.frames());
        downmix_to_mono(&self.interleaved, self.channels, &mut mono);
        mono
    }
}

/// Decode an `.mp3`/`.wav` file fully into interleaved `f32` at its native rate.
pub fn decode_file(path: impl AsRef<Path>) -> Result<DecodedAudio> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening audio file {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probing audio format (is this a supported mp3/wav file?)")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track found"))?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("creating decoder for track")?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(0);
    let mut interleaved: Vec<f32> = Vec::new();
    // Reused across packets; sized on first use to the packet's capacity.
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // Clean EOF: symphonia reports it as an unexpected-eof IoError.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e).context("reading next packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                channels = spec.channels.count() as u16;

                let buf = sample_buf.get_or_insert_with(|| {
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, spec)
                });
                buf.copy_interleaved_ref(decoded);
                interleaved.extend_from_slice(buf.samples());
            }
            // Decode errors on a single packet are recoverable: skip it.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding packet"),
        }
    }

    if sample_rate == 0 || channels == 0 || interleaved.is_empty() {
        return Err(anyhow!(
            "decoded no audio from {} (rate={sample_rate}, channels={channels}, samples={})",
            path.display(),
            interleaved.len()
        ));
    }

    Ok(DecodedAudio {
        sample_rate,
        channels,
        interleaved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_an_error() {
        assert!(decode_file("/no/such/file.wav").is_err());
    }

    #[test]
    fn frames_and_duration_are_consistent() {
        let a = DecodedAudio {
            sample_rate: 1000,
            channels: 2,
            interleaved: vec![0.0; 4000], // 2000 frames
        };
        assert_eq!(a.frames(), 2000);
        assert!((a.duration_secs() - 2.0).abs() < 1e-9);
        assert_eq!(a.to_mono().len(), 2000);
    }
}
