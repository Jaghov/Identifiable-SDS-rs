//! Color palettes used by `snlds-viz` for state-aware visualisations.
//!
//! All hand-picked numeric values (palette colors, viridis anchors, image dimensions,
//! sparsity thresholds) live here so that callers in `log.rs` and binaries don't
//! sprinkle magic numbers around.

/// Categorical palette (RGB, 0..=255) for discrete-state visuals (state strip rows,
/// Markov-chain node fills). Colors come from matplotlib's `tab10`.
///
/// Indexing wraps with `state_color(k)` when `k >= STATE_PALETTE.len()`.
pub const STATE_PALETTE: &[[u8; 3]] = &[
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
];

/// Pixel height of a single-sequence "state strip" image (Figure-6 style band).
///
/// One row of a strip image is `1 × T × 3`, which Rerun renders as a 1-pixel-tall band
/// that's hard to read; we tile vertically to `STATE_STRIP_HEIGHT` rows for visibility.
pub const STATE_STRIP_HEIGHT: u32 = 16;

/// Minimum |q_{ij}| at which a transition is treated as structurally present.
/// Edges below this are skipped when logging a Markov-chain graph so the layout
/// isn't dominated by numerical noise / dense fully-connected matrices.
pub const TRANSITION_EDGE_EPSILON: f32 = 1e-3;

/// Five anchor stops sampled from matplotlib's `viridis` (at t = 0, 0.25, 0.5, 0.75, 1).
const VIRIDIS_ANCHORS: [[f32; 3]; 5] = [
    [68.0, 1.0, 84.0],
    [59.0, 82.0, 139.0],
    [33.0, 145.0, 140.0],
    [94.0, 201.0, 98.0],
    [253.0, 231.0, 37.0],
];

/// Map a continuous value `value ∈ [0, 1]` to an RGB triplet using a 5-stop viridis
/// approximation. Inputs are clamped to `[0, 1]` first.
pub fn viridis_rgb(value: f32) -> [u8; 3] {
    let clamped = value.clamp(0.0, 1.0);
    let scaled = clamped * (VIRIDIS_ANCHORS.len() as f32 - 1.0);
    let lower_idx = scaled.floor() as usize;
    let upper_idx = (lower_idx + 1).min(VIRIDIS_ANCHORS.len() - 1);
    let frac = scaled - lower_idx as f32;
    let interp = |a: f32, b: f32| -> u8 { (a + (b - a) * frac).round().clamp(0.0, 255.0) as u8 };
    [
        interp(VIRIDIS_ANCHORS[lower_idx][0], VIRIDIS_ANCHORS[upper_idx][0]),
        interp(VIRIDIS_ANCHORS[lower_idx][1], VIRIDIS_ANCHORS[upper_idx][1]),
        interp(VIRIDIS_ANCHORS[lower_idx][2], VIRIDIS_ANCHORS[upper_idx][2]),
    ]
}

/// Pick a categorical color for a discrete state index, wrapping if `state_index`
/// exceeds the palette length.
pub fn state_color(state_index: usize) -> [u8; 3] {
    STATE_PALETTE[state_index % STATE_PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints_match_anchors() {
        let low = viridis_rgb(0.0);
        let high = viridis_rgb(1.0);
        assert_eq!(low, [68, 1, 84]);
        assert_eq!(high, [253, 231, 37]);
    }

    #[test]
    fn viridis_clamps_out_of_range_inputs() {
        assert_eq!(viridis_rgb(-0.5), viridis_rgb(0.0));
        assert_eq!(viridis_rgb(2.0), viridis_rgb(1.0));
    }

    #[test]
    fn state_color_wraps_palette() {
        let palette_len = STATE_PALETTE.len();
        assert_eq!(state_color(0), state_color(palette_len));
        assert_eq!(state_color(1), state_color(palette_len + 1));
    }
}
