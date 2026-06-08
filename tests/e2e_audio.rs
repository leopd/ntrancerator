//! End-to-end audio test (spec §15 acceptance: "a pure tone produces a
//! correctly-placed band ... that tracks pitch").
//!
//! This exercises the full file path with a real audio file on disk:
//!
//!   synthesize sine -> write .wav -> symphonia decode -> downmix -> STFT
//!   -> assert the peak bin maps back to the original frequency.
//!
//! It needs no audio device or GPU, so it runs headlessly under
//! `cargo test --no-default-features`.

use ntrancerator::audio::decode::decode_file;
use ntrancerator::config::WindowKind;
use ntrancerator::dsp::StftProducer;
use std::f32::consts::PI;

/// Write a stereo 16-bit WAV containing a sine at `freq` Hz to `path`.
///
/// The right channel is the negated left channel, so a correct mono downmix
/// (averaging) yields silence — a deliberately hostile case. We therefore put
/// the tone only on the left and keep the right at zero instead, so the downmix
/// preserves it at half amplitude.
fn write_sine_wav(path: &std::path::Path, freq: f32, sample_rate: u32, secs: f32) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let n = (sample_rate as f32 * secs) as usize;
    let amp = i16::MAX as f32 * 0.5;
    for i in 0..n {
        let s = (2.0 * PI * freq * i as f32 / sample_rate as f32).sin();
        let v = (s * amp) as i16;
        writer.write_sample(v).unwrap(); // left: tone
        writer.write_sample(0i16).unwrap(); // right: silence
    }
    writer.finalize().unwrap();
}

/// Index of the maximum-magnitude bin.
fn peak_bin(column: &[f32]) -> usize {
    column
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn wav_file_tone_decodes_and_lands_in_expected_bin() {
    let sr = 48_000u32;
    let fft = 2048usize;
    // Put the tone on a bin center to avoid spectral leakage: bin 80 -> 1875 Hz.
    let target_bin = 80;
    let freq = target_bin as f32 * sr as f32 / fft as f32;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("ntrancerator_e2e_{}.wav", std::process::id()));
    write_sine_wav(&path, freq, sr, 0.5);

    // Decode the real file from disk.
    let decoded = decode_file(&path).expect("decode wav");
    assert_eq!(decoded.sample_rate, sr);
    assert_eq!(decoded.channels, 2);
    assert!(decoded.duration_secs() > 0.4);

    // Downmix to mono and run the STFT, keeping the last completed column.
    let mono = decoded.to_mono();
    let mut producer = StftProducer::new(fft, fft / 2, WindowKind::Hann, decoded.sample_rate);
    let mut last_column: Vec<f32> = Vec::new();
    let columns = producer.process(&mono, |col| last_column = col.to_vec());

    assert!(columns > 0, "expected at least one STFT column");
    assert_eq!(last_column.len(), fft / 2 + 1);

    let pk = peak_bin(&last_column);
    assert_eq!(pk, target_bin, "peak bin {pk} != target {target_bin}");
    // And the peak bin maps back to (approximately) the original pitch.
    assert!((producer.bin_to_freq(pk) - freq).abs() < 1.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn decoded_pitch_tracks_a_different_frequency() {
    // A second frequency lands in a different, correctly-placed bin — i.e. the
    // band "tracks pitch" rather than being pinned to one location.
    let sr = 44_100u32;
    let fft = 4096usize;
    let target_bin = 200; // ~2153 Hz at 44.1k/4096
    let freq = target_bin as f32 * sr as f32 / fft as f32;

    let path = std::env::temp_dir().join(format!("ntrancerator_e2e2_{}.wav", std::process::id()));
    write_sine_wav(&path, freq, sr, 0.5);

    let decoded = decode_file(&path).expect("decode wav");
    let mono = decoded.to_mono();
    let mut producer = StftProducer::new(fft, fft / 2, WindowKind::Hann, decoded.sample_rate);
    let mut last_column: Vec<f32> = Vec::new();
    producer.process(&mono, |col| last_column = col.to_vec());

    let pk = peak_bin(&last_column);
    assert!(
        (pk as i32 - target_bin).abs() <= 1,
        "peak bin {pk} != target {target_bin}"
    );

    std::fs::remove_file(&path).ok();
}
