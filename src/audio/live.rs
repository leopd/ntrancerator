//! Live capture via `cpal` (spec §6.3).
//!
//! Opens an input stream from the default (or a named) device, downmixes each
//! callback to mono, and pushes the samples into the analysis ring. The cpal
//! callback does only format conversion + downmix + a lock-free push — no
//! allocation, no locks, no logging, and it never panics.
//!
//! This module is only compiled with the `playback` feature; it cannot be unit
//! tested headlessly, so the testable downmix/ring logic lives in the parent
//! module instead.

use super::{analysis_ring, downmix_to_mono};
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, StreamConfig};
use rtrb::{Consumer, Producer};

/// A running live-capture stream plus the consumer end of its analysis ring.
///
/// The `stream` must be kept alive for capture to continue; dropping it stops
/// the device.
pub struct LiveCapture {
    pub sample_rate: u32,
    pub channels: u16,
    pub consumer: Consumer<f32>,
    _stream: cpal::Stream,
}

impl super::Source for LiveCapture {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
}

impl super::RingSource for LiveCapture {
    fn consumer(&mut self) -> &mut Consumer<f32> {
        &mut self.consumer
    }
}

/// Print available input devices and their default config (spec §10
/// `--list-devices`).
pub fn list_input_devices() -> Result<()> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    println!("Input devices:");
    for (i, device) in host
        .input_devices()
        .context("enumerating input devices")?
        .enumerate()
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());
        let marker = if Some(&name) == default_name.as_ref() {
            " (default)"
        } else {
            ""
        };
        let cfg = device
            .default_input_config()
            .map(|c| {
                format!(
                    "{} ch @ {} Hz, {:?}",
                    c.channels(),
                    c.sample_rate().0,
                    c.sample_format()
                )
            })
            .unwrap_or_else(|_| "<no default config>".into());
        println!("  [{i}] {name}{marker} — {cfg}");
    }
    Ok(())
}

/// Open an input stream from the default device, or a device whose name
/// contains `device_name`. `fft_size` sizes the analysis ring (spec §5).
pub fn open(device_name: Option<&str>, fft_size: usize) -> Result<LiveCapture> {
    let host = cpal::default_host();
    let device = select_device(&host, device_name)?;
    let supported = device
        .default_input_config()
        .context("querying default input config")?;

    let sample_format = supported.sample_format();
    let channels = supported.channels();
    let sample_rate = supported.sample_rate().0;
    let config: StreamConfig = supported.into();

    let (producer, consumer) = analysis_ring(fft_size, sample_rate);

    let stream = build_input_stream(&device, &config, sample_format, channels, producer)?;
    stream.play().context("starting input stream")?;

    log::info!(
        "live capture: {} ch @ {} Hz ({:?})",
        channels,
        sample_rate,
        sample_format
    );

    Ok(LiveCapture {
        sample_rate,
        channels,
        consumer,
        _stream: stream,
    })
}

fn select_device(host: &cpal::Host, device_name: Option<&str>) -> Result<Device> {
    match device_name {
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device available")),
        Some(want) => host
            .input_devices()
            .context("enumerating input devices")?
            .find(|d| d.name().map(|n| n.contains(want)).unwrap_or(false))
            .ok_or_else(|| anyhow!("no input device matching '{want}'")),
    }
}

/// Build an input stream for whatever sample format the device prefers,
/// converting to `f32`, downmixing to mono, and pushing into the ring.
fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    channels: u16,
    producer: Producer<f32>,
) -> Result<cpal::Stream> {
    let err_fn = |e| log::error!("input stream error: {e}");

    // One reusable scratch buffer captured by the callback (allocated once, on
    // the audio thread's first use it grows then never reallocates).
    macro_rules! make_stream {
        ($sample:ty) => {{
            let mut producer = producer;
            let mut mono: Vec<f32> = Vec::new();
            let mut interleaved: Vec<f32> = Vec::new();
            device.build_input_stream(
                config,
                move |data: &[$sample], _: &cpal::InputCallbackInfo| {
                    interleaved.clear();
                    interleaved.extend(data.iter().map(|s| f32::from_sample(*s)));
                    mono.clear();
                    downmix_to_mono(&interleaved, channels, &mut mono);
                    for &s in &mono {
                        // Drop samples rather than block if the consumer falls
                        // behind; never panic in the callback.
                        let _ = producer.push(s);
                    }
                },
                err_fn,
                None,
            )
        }};
    }

    let stream = match format {
        SampleFormat::F32 => make_stream!(f32),
        SampleFormat::I16 => make_stream!(i16),
        SampleFormat::U16 => make_stream!(u16),
        other => return Err(anyhow!("unsupported sample format: {other:?}")),
    }
    .context("building input stream")?;
    Ok(stream)
}
