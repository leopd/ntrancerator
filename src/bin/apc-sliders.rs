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

/// Try to apply a raw MIDI message to the slider value array.
/// Returns `Some(index)` if a value was updated, `None` otherwise.
fn apply_cc(message: &[u8], cc_first: u8, cc_last: u8, values: &mut [u8]) -> Option<usize> {
    if message.len() < 3 {
        return None;
    }
    // CC status byte: 0xB0..0xBF (any channel).
    if (message[0] & 0xF0) != 0xB0 {
        return None;
    }
    let cc = message[1];
    let val = message[2];
    if cc < cc_first || cc > cc_last {
        return None;
    }
    let idx = (cc - cc_first) as usize;
    if idx >= values.len() {
        return None;
    }
    values[idx] = val;
    Some(idx)
}

/// Render a single slider bar of the given `width` for a MIDI value (0..=127).
fn render_bar(val: u8, width: usize) -> String {
    let filled = (val as usize * width) / 127;
    "█".repeat(filled) + &"░".repeat(width - filled)
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
                if let Ok(mut v) = values_cb.lock() {
                    apply_cc(message, cc_first, cc_last, &mut v);
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
            let bar = render_bar(val, bar_width);
            println!(
                "   {:<4} │ {:>3} │ {bar}",
                i + 1,
                val,
            );
        }

        thread::sleep(poll);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- apply_cc ---

    #[test]
    fn cc_in_range_updates_correct_slot() {
        let mut vals = [0u8; 9];
        // CC 50 with cc_first=48 → index 2.
        let msg = [0xB0, 50, 100];
        assert_eq!(apply_cc(&msg, 48, 56, &mut vals), Some(2));
        assert_eq!(vals[2], 100);
    }

    #[test]
    fn cc_updates_leave_other_slots_unchanged() {
        let mut vals = [10u8; 9];
        let msg = [0xB0, 48, 77];
        apply_cc(&msg, 48, 56, &mut vals);
        assert_eq!(vals[0], 77);
        for &v in &vals[1..] {
            assert_eq!(v, 10);
        }
    }

    #[test]
    fn cc_below_range_is_ignored() {
        let mut vals = [0u8; 9];
        let msg = [0xB0, 47, 100];
        assert_eq!(apply_cc(&msg, 48, 56, &mut vals), None);
        assert!(vals.iter().all(|&v| v == 0));
    }

    #[test]
    fn cc_above_range_is_ignored() {
        let mut vals = [0u8; 9];
        let msg = [0xB0, 57, 100];
        assert_eq!(apply_cc(&msg, 48, 56, &mut vals), None);
        assert!(vals.iter().all(|&v| v == 0));
    }

    #[test]
    fn non_cc_status_bytes_are_ignored() {
        let mut vals = [0u8; 9];
        // Note On (0x90), Note Off (0x80), Program Change (0xC0).
        for status in [0x90, 0x80, 0xC0] {
            let msg = [status, 50, 100];
            assert_eq!(apply_cc(&msg, 48, 56, &mut vals), None);
        }
        assert!(vals.iter().all(|&v| v == 0));
    }

    #[test]
    fn cc_on_any_channel_is_accepted() {
        let mut vals = [0u8; 9];
        // Channel 5 → status 0xB5.
        let msg = [0xB5, 48, 64];
        assert_eq!(apply_cc(&msg, 48, 56, &mut vals), Some(0));
        assert_eq!(vals[0], 64);
    }

    #[test]
    fn short_message_is_ignored() {
        let mut vals = [0u8; 9];
        assert_eq!(apply_cc(&[0xB0], 48, 56, &mut vals), None);
        assert_eq!(apply_cc(&[0xB0, 50], 48, 56, &mut vals), None);
        assert_eq!(apply_cc(&[], 48, 56, &mut vals), None);
    }

    #[test]
    fn boundary_cc_values() {
        let mut vals = [0u8; 9];
        // First in range.
        let msg = [0xB0, 48, 0];
        assert_eq!(apply_cc(&msg, 48, 56, &mut vals), Some(0));
        assert_eq!(vals[0], 0);
        // Last in range.
        let msg = [0xB0, 56, 127];
        assert_eq!(apply_cc(&msg, 48, 56, &mut vals), Some(8));
        assert_eq!(vals[8], 127);
    }

    #[test]
    fn single_cc_range() {
        let mut vals = [0u8; 1];
        let msg = [0xB0, 48, 99];
        assert_eq!(apply_cc(&msg, 48, 48, &mut vals), Some(0));
        assert_eq!(vals[0], 99);
        // One above.
        let msg = [0xB0, 49, 99];
        assert_eq!(apply_cc(&msg, 48, 48, &mut vals), None);
    }

    #[test]
    fn successive_updates_overwrite() {
        let mut vals = [0u8; 9];
        apply_cc(&[0xB0, 50, 10], 48, 56, &mut vals);
        apply_cc(&[0xB0, 50, 20], 48, 56, &mut vals);
        apply_cc(&[0xB0, 50, 30], 48, 56, &mut vals);
        assert_eq!(vals[2], 30);
    }

    // --- render_bar ---

    #[test]
    fn bar_zero_is_all_empty() {
        let bar = render_bar(0, 40);
        assert_eq!(bar, "░".repeat(40));
    }

    #[test]
    fn bar_max_is_all_filled() {
        let bar = render_bar(127, 40);
        assert_eq!(bar, "█".repeat(40));
    }

    #[test]
    fn bar_has_correct_width() {
        for width in [10, 20, 40, 80] {
            for val in [0, 1, 63, 64, 126, 127] {
                let bar = render_bar(val, width);
                assert_eq!(
                    bar.chars().count(),
                    width,
                    "width={width}, val={val}"
                );
            }
        }
    }

    #[test]
    fn bar_midpoint_is_roughly_half() {
        let bar = render_bar(64, 40);
        let filled = bar.chars().filter(|&c| c == '█').count();
        // 64/127 * 40 = 20.15 → 20 filled.
        assert_eq!(filled, 20);
    }

    #[test]
    fn bar_zero_width() {
        let bar = render_bar(127, 0);
        assert_eq!(bar, "");
    }

    #[test]
    fn bar_is_monotonically_non_decreasing() {
        let width = 40;
        let mut prev_filled = 0;
        for val in 0..=127u8 {
            let bar = render_bar(val, width);
            let filled = bar.chars().filter(|&c| c == '█').count();
            assert!(
                filled >= prev_filled,
                "val={val}: filled={filled} < prev={prev_filled}"
            );
            prev_filled = filled;
        }
    }

    // --- CLI defaults ---

    #[test]
    fn cli_defaults() {
        let cli = Cli::parse_from(["apc-sliders"]);
        assert_eq!(cli.port, "APC mini mk2");
        assert_eq!(cli.poll_ms, 500);
        assert_eq!(cli.cc_first, 48);
        assert_eq!(cli.cc_last, 56);
        assert!(!cli.list);
    }

    #[test]
    fn cli_custom_args() {
        let cli = Cli::parse_from([
            "apc-sliders",
            "--port", "My Controller",
            "--poll-ms", "100",
            "--cc-first", "0",
            "--cc-last", "7",
            "--list",
        ]);
        assert_eq!(cli.port, "My Controller");
        assert_eq!(cli.poll_ms, 100);
        assert_eq!(cli.cc_first, 0);
        assert_eq!(cli.cc_last, 7);
        assert!(cli.list);
    }
}
