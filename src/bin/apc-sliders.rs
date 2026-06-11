use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use midir::{MidiInput, MidiInputPort};

/// The APC mini mk2 has 9 sliders sending CC 48..=56 on channel 0.
const DEFAULT_CC_FIRST: u8 = 48;
const DEFAULT_CC_LAST: u8 = 56;
/// Read slider values from an Akai APC mini mk2 (or similar MIDI controller)
/// and display them as a text bar chart, polling at a configurable rate.
#[derive(Parser)]
#[command(name = "apc-sliders")]
struct Cli {
    /// Substring to match against MIDI port names.
    /// The first port whose name contains this string (case-insensitive) is used.
    #[arg(short, long, default_value = "APC mini mk2")]
    port: String,

    /// Poll / display refresh interval in milliseconds.
    #[arg(short = 'r', long, default_value_t = 500)]
    poll_ms: u64,

    /// First CC number in the slider range (inclusive).
    #[arg(long, default_value_t = DEFAULT_CC_FIRST)]
    cc_first: u8,

    /// Last CC number in the slider range (inclusive).
    #[arg(long, default_value_t = DEFAULT_CC_LAST)]
    cc_last: u8,

    /// List available MIDI input ports and exit.
    #[arg(short, long)]
    list: bool,
}

fn find_port(midi_in: &MidiInput, needle: &str) -> Result<MidiInputPort> {
    let needle_lower = needle.to_lowercase();
    let ports = midi_in.ports();
    for port in &ports {
        let name = midi_in
            .port_name(port)
            .unwrap_or_else(|_| "(unknown)".into());
        if name.to_lowercase().contains(&needle_lower) {
            return Ok(port.clone());
        }
    }
    bail!(
        "no MIDI port matching {:?} found. Run with --list to see available ports.",
        needle
    );
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.cc_last < cli.cc_first {
        bail!("--cc-last ({}) must be >= --cc-first ({})", cli.cc_last, cli.cc_first);
    }
    let num_sliders = (cli.cc_last - cli.cc_first + 1) as usize;

    let midi_in = MidiInput::new("apc-sliders").context("failed to create MIDI input")?;

    if cli.list {
        let ports = midi_in.ports();
        if ports.is_empty() {
            println!("No MIDI input ports found.");
        } else {
            println!("MIDI input ports:");
            for (i, port) in ports.iter().enumerate() {
                let name = midi_in
                    .port_name(port)
                    .unwrap_or_else(|_| "(unknown)".into());
                println!("  {i}: {name}");
            }
        }
        return Ok(());
    }

    let port = find_port(&midi_in, &cli.port)?;
    let port_name = midi_in
        .port_name(&port)
        .unwrap_or_else(|_| "(unknown)".into());
    println!("Connected to: {port_name}");
    println!(
        "Reading CC {}..={} ({num_sliders} sliders), refreshing every {} ms",
        cli.cc_first, cli.cc_last, cli.poll_ms
    );
    println!("Press Ctrl-C to quit.\n");

    let values: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0u8; num_sliders]));
    let values_cb = Arc::clone(&values);
    let cc_first = cli.cc_first;
    let cc_last = cli.cc_last;

    // The connection must be kept alive — dropping it closes the port.
    let _conn = midi_in
        .connect(
            &port,
            "apc-sliders-read",
            move |_timestamp, message, _| {
                // MIDI CC message: status 0xB0..0xBF, then cc number, then value.
                if message.len() >= 3 && (message[0] & 0xF0) == 0xB0 {
                    let cc = message[1];
                    let val = message[2];
                    if cc >= cc_first && cc <= cc_last {
                        let idx = (cc - cc_first) as usize;
                        if let Ok(mut v) = values_cb.lock() {
                            v[idx] = val;
                        }
                    }
                }
            },
            (),
        )
        .map_err(|e| anyhow::anyhow!("failed to connect to MIDI port: {e}"))?;

    let bar_width: usize = 40;
    let poll = Duration::from_millis(cli.poll_ms);

    loop {
        let snapshot = values.lock().unwrap().clone();

        // Move cursor up to overwrite previous output (after the first iteration).
        // We print num_sliders + 1 lines (header + one per slider).
        print!("\x1B[{}A", num_sliders + 1);

        println!(
            " Slider │ Val │ {:<bar_width$}",
            "",
            bar_width = bar_width
        );
        for (i, &val) in snapshot.iter().enumerate() {
            let filled = (val as usize * bar_width) / 127;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            println!(
                "   {:<4} │ {:>3} │ {bar}",
                i + 1,
                val,
            );
        }

        thread::sleep(poll);
    }
}
