//! File playback + analysis tee (spec §6.2, §4 sync detail).
//!
//! With playback enabled, a `cpal` **output** stream is the clock: its callback
//! pulls decoded frames to play and tees a mono downmix of the very same frames
//! into the analysis ring, so audio and visuals stay aligned for free. With
//! `--no-audio-out`, a paced feeder thread instead pushes mono samples into the
//! ring at the file's real-time rate.
//!
//! Only compiled with the `playback` feature.

use super::analysis_ring;
use super::decode::DecodedAudio;
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SampleRate, StreamConfig};
use rtrb::Consumer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A running file-playback (or paced-feeder) source plus the consumer end of
/// its analysis ring. Keep this alive while visualizing; dropping it stops audio
/// and signals the feeder thread to exit.
pub struct FilePlayback {
    pub sample_rate: u32,
    pub channels: u16,
    pub consumer: Consumer<f32>,
    // Whichever driver is in use is held here to keep it alive.
    _stream: Option<cpal::Stream>,
    _feeder: Option<FeederHandle>,
}

impl super::Source for FilePlayback {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
}

impl super::RingSource for FilePlayback {
    fn consumer(&mut self) -> &mut Consumer<f32> {
        &mut self.consumer
    }
}

struct FeederHandle {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Drop for FeederHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Set up a file source. When `no_audio_out` is false, audio plays to the
/// default output device and is teed into the ring; otherwise a paced feeder
/// drives the ring silently.
pub fn open(decoded: DecodedAudio, no_audio_out: bool, fft_size: usize) -> Result<FilePlayback> {
    let sample_rate = decoded.sample_rate;
    let channels = decoded.channels;
    let (producer, consumer) = analysis_ring(fft_size, sample_rate);

    if no_audio_out {
        let stop = Arc::new(AtomicBool::new(false));
        let join = spawn_feeder(decoded, producer, stop.clone());
        log::info!("file source: paced feeder (no audio out) @ {sample_rate} Hz");
        return Ok(FilePlayback {
            sample_rate,
            channels,
            consumer,
            _stream: None,
            _feeder: Some(FeederHandle {
                stop,
                join: Some(join),
            }),
        });
    }

    let stream = build_output_stream(decoded, producer)?;
    stream.play().context("starting output stream")?;
    log::info!("file source: playback @ {sample_rate} Hz, {channels} ch");
    Ok(FilePlayback {
        sample_rate,
        channels,
        consumer,
        _stream: Some(stream),
        _feeder: None,
    })
}

/// Paced feeder: push mono samples into the ring at real-time rate (spec §6.2).
fn spawn_feeder(
    decoded: DecodedAudio,
    mut producer: rtrb::Producer<f32>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    let mono = decoded.to_mono();
    let sample_rate = decoded.sample_rate as f64;
    std::thread::spawn(move || {
        // Feed in ~10 ms chunks, pacing against a wall clock.
        let chunk = (sample_rate * 0.01).max(1.0) as usize;
        let start = Instant::now();
        let mut sent = 0usize;
        while sent < mono.len() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let end = (sent + chunk).min(mono.len());
            for &s in &mono[sent..end] {
                let _ = producer.push(s);
            }
            sent = end;
            // Sleep until this many samples "should" have elapsed.
            let target = Duration::from_secs_f64(sent as f64 / sample_rate);
            if let Some(remaining) = target.checked_sub(start.elapsed()) {
                std::thread::sleep(remaining);
            }
        }
    })
}

/// Output stream that plays decoded frames and tees a mono downmix to the ring.
fn build_output_stream(
    decoded: DecodedAudio,
    producer: rtrb::Producer<f32>,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no default output device")?;
    let out_format = device
        .default_output_config()
        .context("querying default output config")?
        .sample_format();

    let channels = decoded.channels;
    // Ask for the file's native layout; PipeWire handles rate conversion (spec §6.2).
    let config = StreamConfig {
        channels,
        sample_rate: SampleRate(decoded.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let err_fn = |e| log::error!("output stream error: {e}");

    macro_rules! make_stream {
        ($sample:ty) => {{
            let data = Arc::new(decoded.interleaved);
            let ch = channels as usize;
            let mut producer = producer;
            let mut pos = 0usize; // index into interleaved samples
            device.build_output_stream(
                &config,
                move |out: &mut [$sample], _: &cpal::OutputCallbackInfo| {
                    for frame in out.chunks_mut(ch) {
                        if pos + ch <= data.len() {
                            let mut sum = 0.0f32;
                            for (c, slot) in frame.iter_mut().enumerate() {
                                let v = data[pos + c];
                                *slot = <$sample as Sample>::from_sample(v);
                                sum += v;
                            }
                            let _ = producer.push(sum / ch as f32);
                            pos += ch;
                        } else {
                            // End of file: output silence, stop teeing.
                            for slot in frame.iter_mut() {
                                *slot = <$sample as Sample>::EQUILIBRIUM;
                            }
                        }
                    }
                },
                err_fn,
                None,
            )
        }};
    }

    let stream = match out_format {
        SampleFormat::F32 => make_stream!(f32),
        SampleFormat::I16 => make_stream!(i16),
        SampleFormat::U16 => make_stream!(u16),
        other => anyhow::bail!("unsupported output sample format: {other:?}"),
    }
    .context("building output stream")?;
    Ok(stream)
}
