//! Pure monitor-selection policy for `--monitor` (spec §9 / §10).
//!
//! This is factored out of the `winit` driver so it can be unit tested without a
//! display or an event loop: the `gpu` module enumerates the real monitors and
//! delegates the *decision* to [`match_monitor_index`], then maps the returned
//! index back to a `winit` `MonitorHandle`.

/// Choose which monitor to target for borderless fullscreen.
///
/// Given the enumerated monitor `names` (in winit's order), the index of the
/// `primary` monitor, and the user's `--monitor` request, return the index of
/// the monitor to target — or `None` to let winit pick the current monitor.
///
/// Policy:
/// - `Some(idx)` that parses as a number → that monitor index (only if in range).
/// - `Some(name)` → first monitor whose name contains `name` (case-insensitive).
/// - `None` → prefer the first non-primary monitor when more than one exists
///   (the external/HDMI heuristic), else the primary / let winit decide.
/// - A request matching nothing → `None` (caller warns and falls back).
pub fn match_monitor_index(
    names: &[String],
    primary: usize,
    requested: Option<&str>,
) -> Option<usize> {
    match requested {
        Some(req) => {
            if let Ok(idx) = req.parse::<usize>() {
                return (idx < names.len()).then_some(idx);
            }
            let needle = req.to_lowercase();
            names.iter().position(|n| n.to_lowercase().contains(&needle))
        }
        None => {
            if names.len() > 1 {
                // Prefer the first output that isn't the primary panel.
                Some((0..names.len()).find(|&i| i != primary).unwrap_or(primary))
            } else {
                // Single (or no) output: let winit use the current monitor.
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn matches_name_case_insensitive_substring() {
        let n = names(&["eDP-1", "HDMI-A-1"]);
        // Primary is the internal panel (index 0).
        assert_eq!(match_monitor_index(&n, 0, Some("hdmi")), Some(1));
        assert_eq!(match_monitor_index(&n, 0, Some("HDMI-A-1")), Some(1));
    }

    #[test]
    fn numeric_request_selects_by_index() {
        let n = names(&["eDP-1", "HDMI-A-1", "DP-2"]);
        assert_eq!(match_monitor_index(&n, 0, Some("2")), Some(2));
        // Out-of-range index matches nothing.
        assert_eq!(match_monitor_index(&n, 0, Some("9")), None);
    }

    #[test]
    fn default_prefers_non_primary_output() {
        // Target B: internal panel (eDP) is primary; HDMI is the external output.
        let n = names(&["eDP-1", "HDMI-A-1"]);
        assert_eq!(match_monitor_index(&n, 0, None), Some(1));
        // If the external output were primary instead, fall back to the other.
        assert_eq!(match_monitor_index(&n, 1, None), Some(0));
    }

    #[test]
    fn default_single_output_lets_winit_decide() {
        // Target A (cage): exactly one output, possibly named "Unknown-1".
        let n = names(&["Unknown-1"]);
        assert_eq!(match_monitor_index(&n, 0, None), None);
    }

    #[test]
    fn unmatched_name_returns_none() {
        let n = names(&["eDP-1", "HDMI-A-1"]);
        assert_eq!(match_monitor_index(&n, 0, Some("DP-9")), None);
    }
}
