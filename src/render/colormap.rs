//! Colormap lookup tables (spec §9).
//!
//! Each colormap is built as a 256-entry RGBA8 LUT, uploaded as a 256×1 texture
//! and sampled by the fragment shader. The perceptual maps (inferno/magma/
//! viridis) are stored as a handful of anchor colors sampled from matplotlib and
//! linearly interpolated to 256 entries — close enough for visualization and far
//! cheaper than embedding the full tables.

use crate::config::Colormap;

/// Number of entries in a colormap LUT.
pub const LUT_LEN: usize = 256;

// Anchors are evenly spaced in t = [0, 1]; index i sits at t = i/(N-1).
type Anchors = &'static [[f32; 3]];

const INFERNO: Anchors = &[
    [0.0015, 0.0005, 0.0139],
    [0.0870, 0.0440, 0.2240],
    [0.2580, 0.0390, 0.4060],
    [0.4510, 0.1220, 0.4200],
    [0.6430, 0.2000, 0.3540],
    [0.8270, 0.3140, 0.2280],
    [0.9520, 0.5310, 0.0840],
    [0.9880, 0.9980, 0.6450],
];

const MAGMA: Anchors = &[
    [0.0010, 0.0000, 0.0140],
    [0.0780, 0.0540, 0.2110],
    [0.2320, 0.0590, 0.4380],
    [0.4100, 0.0910, 0.4330],
    [0.5880, 0.1490, 0.4040],
    [0.7700, 0.2150, 0.3300],
    [0.9290, 0.4120, 0.1450],
    [0.9870, 0.9910, 0.7490],
];

const VIRIDIS: Anchors = &[
    [0.2670, 0.0050, 0.3290],
    [0.2830, 0.1310, 0.4490],
    [0.2540, 0.2650, 0.5300],
    [0.2070, 0.3720, 0.5530],
    [0.1640, 0.4710, 0.5580],
    [0.1280, 0.5670, 0.5510],
    [0.2670, 0.7490, 0.4410],
    [0.9930, 0.9060, 0.1440],
];

fn anchors_for(cm: Colormap) -> Option<Anchors> {
    match cm {
        Colormap::Inferno => Some(INFERNO),
        Colormap::Magma => Some(MAGMA),
        Colormap::Viridis => Some(VIRIDIS),
        Colormap::Gray => None,
    }
}

/// Sample a colormap at `t` in `[0, 1]`, returning linear RGB in `[0, 1]`.
pub fn sample(cm: Colormap, t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let Some(anchors) = anchors_for(cm) else {
        return [t, t, t]; // grayscale
    };
    let segments = anchors.len() - 1;
    let scaled = t * segments as f32;
    let i = (scaled.floor() as usize).min(segments - 1);
    let frac = scaled - i as f32;
    let a = anchors[i];
    let b = anchors[i + 1];
    [
        a[0] + (b[0] - a[0]) * frac,
        a[1] + (b[1] - a[1]) * frac,
        a[2] + (b[2] - a[2]) * frac,
    ]
}

/// Build the 256-entry RGBA8 LUT for `cm` (alpha fixed at 255), laid out as
/// `[r, g, b, a, r, g, b, a, ...]` ready for upload as an `Rgba8Unorm` texture.
pub fn lut_rgba8(cm: Colormap) -> Vec<u8> {
    let mut out = Vec::with_capacity(LUT_LEN * 4);
    for i in 0..LUT_LEN {
        let t = i as f32 / (LUT_LEN - 1) as f32;
        let [r, g, b] = sample(cm, t);
        out.push((r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push(255);
    }
    out
}

/// Relative luminance of a linear RGB triple (Rec. 709 weights).
#[cfg(test)]
fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Colormap; 4] = [
        Colormap::Inferno,
        Colormap::Magma,
        Colormap::Viridis,
        Colormap::Gray,
    ];

    #[test]
    fn lut_has_correct_size_and_opaque_alpha() {
        for cm in ALL {
            let lut = lut_rgba8(cm);
            assert_eq!(lut.len(), LUT_LEN * 4);
            for px in lut.chunks_exact(4) {
                assert_eq!(px[3], 255, "alpha must be opaque");
            }
        }
    }

    #[test]
    fn sample_clamps_out_of_range_t() {
        for cm in ALL {
            assert_eq!(sample(cm, -1.0), sample(cm, 0.0));
            assert_eq!(sample(cm, 2.0), sample(cm, 1.0));
        }
    }

    #[test]
    fn endpoints_match_anchor_table() {
        // t=0 and t=1 should reproduce the first/last anchors exactly.
        let lo = sample(Colormap::Inferno, 0.0);
        assert!((lo[0] - INFERNO[0][0]).abs() < 1e-6);
        let hi = sample(Colormap::Viridis, 1.0);
        assert!((hi[1] - VIRIDIS[VIRIDIS.len() - 1][1]).abs() < 1e-6);
    }

    #[test]
    fn perceptual_maps_get_brighter_overall() {
        // The dark-to-light maps should end far brighter than they start.
        for cm in [Colormap::Inferno, Colormap::Magma, Colormap::Viridis] {
            let lo = luminance(sample(cm, 0.0));
            let hi = luminance(sample(cm, 1.0));
            assert!(hi > lo + 0.3, "{cm:?}: hi {hi} not >> lo {lo}");
        }
    }

    #[test]
    fn gray_is_neutral_and_linear() {
        assert_eq!(sample(Colormap::Gray, 0.0), [0.0, 0.0, 0.0]);
        assert_eq!(sample(Colormap::Gray, 0.5), [0.5, 0.5, 0.5]);
        assert_eq!(sample(Colormap::Gray, 1.0), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn inferno_starts_dark() {
        assert!(luminance(sample(Colormap::Inferno, 0.0)) < 0.05);
    }
}
