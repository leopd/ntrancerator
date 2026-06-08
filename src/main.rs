//! `spectro` — N-Trancerator real-time spectrogram visualizer (spec §10, §13).
//!
//! Thin wiring layer: parse the CLI, build an audio source, and hand it to the
//! render loop. All substantive logic lives in the `ntrancerator` library.

use anyhow::Result;
use clap::Parser;
use ntrancerator::Config;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let config = Config::parse();
    config.validate()?;

    run(config)
}

#[cfg(all(feature = "playback", feature = "gui"))]
fn run(config: Config) -> Result<()> {
    use ntrancerator::audio::{self, RingSource};
    use ntrancerator::config::InputKind;

    // `--list-devices` short-circuits before any source is opened (spec §10).
    if config.list_devices {
        return audio::live::list_input_devices();
    }

    let source: Box<dyn RingSource> = match config.input {
        InputKind::Live => Box::new(audio::live::open(
            config.device.as_deref(),
            config.fft_size,
        )?),
        InputKind::File => {
            // `validate()` guarantees `file` is present in this branch.
            let path = config
                .file
                .clone()
                .expect("validated: --input file requires --file");
            log::info!("decoding {path}");
            let decoded = audio::decode::decode_file(&path)?;
            log::info!(
                "decoded {:.1}s, {} ch @ {} Hz",
                decoded.duration_secs(),
                decoded.channels,
                decoded.sample_rate
            );
            Box::new(audio::playback::open(
                decoded,
                config.no_audio_out,
                config.fft_size,
            )?)
        }
    };

    ntrancerator::render::gpu::run(config, source)
}

// Allow building/testing the core without the heavy platform deps. The binary
// still links, but explains what it needs to actually run.
#[cfg(not(all(feature = "playback", feature = "gui")))]
fn run(_config: Config) -> Result<()> {
    anyhow::bail!(
        "spectro was built without the required features. \
         Rebuild with `--features playback,gui` (the default) to run the visualizer."
    )
}
