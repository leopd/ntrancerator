//! CLI configuration and defaults (spec §10).
//!
//! This module is intentionally free of any audio/GPU dependencies so the whole
//! configuration surface — including validation and derived values — is unit
//! testable without a device or a display.

use clap::{Parser, ValueEnum};

/// Window function applied to each analysis frame before the FFT (spec §7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum WindowKind {
    Hann,
    Hamming,
    #[value(name = "blackman-harris")]
    BlackmanHarris,
}

/// Source of audio samples (spec §6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum InputKind {
    /// Live capture device (cpal input).
    Live,
    /// Decoded file (symphonia).
    File,
}

/// Colormap applied to normalized dB in the fragment shader (spec §9).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Colormap {
    Inferno,
    Magma,
    Viridis,
    Gray,
}

impl Colormap {
    /// Cycle order for the `C` keyboard control (spec §10).
    pub fn next(self) -> Colormap {
        match self {
            Colormap::Inferno => Colormap::Magma,
            Colormap::Magma => Colormap::Viridis,
            Colormap::Viridis => Colormap::Gray,
            Colormap::Gray => Colormap::Inferno,
        }
    }
}

/// Parsed command-line configuration for `spectro` (spec §10).
#[derive(Parser, Debug, Clone)]
#[command(
    name = "spectro",
    about = "Real-time GPU audio spectrogram visualizer",
    // Allow negative dB values like `--db-floor -100` to parse as numbers
    // rather than being mistaken for flags.
    allow_negative_numbers = true
)]
pub struct Config {
    // ---- Input ----
    /// Source type.
    #[arg(long, value_enum, default_value_t = InputKind::Live)]
    pub input: InputKind,

    /// .mp3/.wav path (required if `--input file`).
    #[arg(long)]
    pub file: Option<String>,

    /// Capture device for live input, by `--list-devices` index or a name
    /// substring (default: system default).
    #[arg(long)]
    pub device: Option<String>,

    /// List input devices and exit.
    #[arg(long, default_value_t = false)]
    pub list_devices: bool,

    /// File mode: visualize without playing audio.
    #[arg(long, default_value_t = false)]
    pub no_audio_out: bool,

    // ---- DSP ----
    /// FFT size (power of two).
    #[arg(long, default_value_t = 2048)]
    pub fft_size: usize,

    /// Hop size in samples (default: fft_size / 2).
    #[arg(long)]
    pub hop: Option<usize>,

    /// Analysis window function.
    #[arg(long, value_enum, default_value_t = WindowKind::Hann)]
    pub window: WindowKind,

    /// Lower dB bound mapped to the bottom of the colormap.
    #[arg(long, default_value_t = -100.0)]
    pub db_floor: f32,

    /// Upper dB bound mapped to the top of the colormap.
    #[arg(long, default_value_t = 0.0)]
    pub db_ceiling: f32,

    // ---- Display ----
    /// Lowest displayed frequency (Hz).
    #[arg(long, default_value_t = 20.0)]
    pub freq_min: f32,

    /// Highest displayed frequency (Hz); default is Nyquist (sample_rate / 2).
    #[arg(long)]
    pub freq_max: Option<f32>,

    /// Colormap.
    #[arg(long, value_enum, default_value_t = Colormap::Inferno)]
    pub colormap: Colormap,

    /// Start in borderless fullscreen.
    #[arg(long, default_value_t = false)]
    pub fullscreen: bool,

    /// Output to target for fullscreen, by monitor name or index.
    /// Default: the external/HDMI output if one is present, else primary.
    #[arg(long)]
    pub monitor: Option<String>,
}

/// Errors surfaced while validating a [`Config`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ConfigError {
    #[error("fft-size must be a power of two and >= 2 (got {0})")]
    FftSizeNotPow2(usize),
    #[error("hop must be in 1..=fft_size (got hop={hop}, fft_size={fft_size})")]
    HopOutOfRange { hop: usize, fft_size: usize },
    #[error("--input file requires --file <PATH>")]
    MissingFile,
    #[error("db-floor ({floor}) must be strictly less than db-ceiling ({ceiling})")]
    DbRange { floor: f32, ceiling: f32 },
    #[error("freq-min must be > 0 (got {0})")]
    FreqMinNonPositive(f32),
}

impl Config {
    /// Effective hop size, applying the `fft_size / 2` default (spec §7).
    pub fn hop_size(&self) -> usize {
        self.hop.unwrap_or(self.fft_size / 2)
    }

    /// Number of frequency bins produced by a real FFT of `fft_size` (spec §8).
    pub fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Resolve the displayed max frequency, defaulting to Nyquist (spec §9).
    pub fn effective_freq_max(&self, sample_rate: u32) -> f32 {
        self.freq_max.unwrap_or(sample_rate as f32 / 2.0)
    }

    /// Validate inter-field constraints that clap cannot express on its own.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.fft_size < 2 || !self.fft_size.is_power_of_two() {
            return Err(ConfigError::FftSizeNotPow2(self.fft_size));
        }
        let hop = self.hop_size();
        if hop == 0 || hop > self.fft_size {
            return Err(ConfigError::HopOutOfRange {
                hop,
                fft_size: self.fft_size,
            });
        }
        // `--list-devices` short-circuits before a source is built, so a missing
        // file is only an error when we actually intend to open one.
        if self.input == InputKind::File && self.file.is_none() && !self.list_devices {
            return Err(ConfigError::MissingFile);
        }
        if self.db_floor >= self.db_ceiling {
            return Err(ConfigError::DbRange {
                floor: self.db_floor,
                ceiling: self.db_ceiling,
            });
        }
        if self.freq_min <= 0.0 {
            return Err(ConfigError::FreqMinNonPositive(self.freq_min));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Config from argv, mirroring how `spectro` is actually invoked.
    fn parse(args: &[&str]) -> Config {
        let mut full = vec!["spectro"];
        full.extend_from_slice(args);
        Config::parse_from(full)
    }

    #[test]
    fn defaults_match_spec() {
        let c = parse(&[]);
        assert_eq!(c.input, InputKind::Live);
        assert_eq!(c.fft_size, 2048);
        assert_eq!(c.hop_size(), 1024);
        assert_eq!(c.window, WindowKind::Hann);
        assert_eq!(c.db_floor, -100.0);
        assert_eq!(c.db_ceiling, 0.0);
        assert_eq!(c.freq_min, 20.0);
        assert_eq!(c.colormap, Colormap::Inferno);
        assert!(!c.fullscreen);
        assert_eq!(c.monitor, None);
        c.validate().unwrap();
    }

    #[test]
    fn monitor_accepts_name_or_index() {
        assert_eq!(
            parse(&["--monitor", "HDMI-A-1"]).monitor.as_deref(),
            Some("HDMI-A-1")
        );
        assert_eq!(parse(&["--monitor", "1"]).monitor.as_deref(), Some("1"));
    }

    #[test]
    fn num_bins_and_nyquist() {
        let c = parse(&["--fft-size", "2048"]);
        assert_eq!(c.num_bins(), 1025);
        assert_eq!(c.effective_freq_max(48_000), 24_000.0);
        let c2 = parse(&["--freq-max", "10000"]);
        assert_eq!(c2.effective_freq_max(48_000), 10_000.0);
    }

    #[test]
    fn rejects_non_power_of_two_fft() {
        let err = parse(&["--fft-size", "1000"]).validate().unwrap_err();
        assert_eq!(err, ConfigError::FftSizeNotPow2(1000));
    }

    #[test]
    fn rejects_hop_larger_than_fft() {
        let err = parse(&["--fft-size", "1024", "--hop", "2048"])
            .validate()
            .unwrap_err();
        assert_eq!(
            err,
            ConfigError::HopOutOfRange {
                hop: 2048,
                fft_size: 1024
            }
        );
    }

    #[test]
    fn file_input_requires_path() {
        let err = parse(&["--input", "file"]).validate().unwrap_err();
        assert_eq!(err, ConfigError::MissingFile);
        // ...unless we are only listing devices.
        parse(&["--input", "file", "--list-devices"])
            .validate()
            .unwrap();
        parse(&["--input", "file", "--file", "song.wav"])
            .validate()
            .unwrap();
    }

    #[test]
    fn rejects_inverted_db_range() {
        let err = parse(&["--db-floor", "0", "--db-ceiling", "-100"])
            .validate()
            .unwrap_err();
        assert!(matches!(err, ConfigError::DbRange { .. }));
    }

    #[test]
    fn rejects_nonpositive_freq_min() {
        let err = parse(&["--freq-min", "0"]).validate().unwrap_err();
        assert_eq!(err, ConfigError::FreqMinNonPositive(0.0));
    }

    #[test]
    fn colormap_cycles_through_all_four() {
        let mut cm = Colormap::Inferno;
        let mut seen = vec![cm];
        for _ in 0..4 {
            cm = cm.next();
            seen.push(cm);
        }
        // Cycles back to the start after exactly four steps.
        assert_eq!(seen[0], seen[4]);
        assert_eq!(seen[1], Colormap::Magma);
        assert_eq!(seen[2], Colormap::Viridis);
        assert_eq!(seen[3], Colormap::Gray);
    }
}
