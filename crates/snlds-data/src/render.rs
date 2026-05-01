//! Rendering 2-D latent trajectories to RGB image frames.
//!
//! Lives in `snlds-data` rather than `snlds-viz` because rendered frames are also a
//! valid simulator output (image observations consumed by the CNN encoder/decoder
//! path). `snlds-viz` re-exports this module to preserve its existing public API.

use ndarray::{Array3, Array4, ArrayView2};

/// World-space bounds matching the Python `_draw` reference.
const X_MIN: f32 = -3.0;
const X_MAX: f32 = 4.0;
const Y_MIN: f32 = -4.0;

/// Ball colour in `[0, 1]` RGB: `[173/255, 146/255, 0/255]`.
const BALL_COLOR: [f32; 3] = [173.0 / 255.0, 146.0 / 255.0, 0.0];

/// Background colour: `[81/255, 88/255, 93/255]`.
const BG_COLOR: [f32; 3] = [81.0 / 255.0, 88.0 / 255.0, 93.0 / 255.0];

/// Render a 2-D latent trajectory into a float32 video.
///
/// `latents` shape: `[T, 2]`. Panics if `latents.ncols() != 2`.
/// Returns `[T, res, res, 3]`, values ∈ `[0, 1]`.
pub fn draw_sequence(latents: ArrayView2<f32>, res: usize) -> Array4<f32> {
    assert_eq!(
        latents.ncols(),
        2,
        "draw_sequence requires dim_latent == 2 (got {})",
        latents.ncols()
    );

    let t_len = latents.nrows();
    let space_res = (X_MAX - X_MIN) / res as f32;
    let radius = (1.0 / space_res) as usize;

    // Allocate one scratch buffer for blur, reused across frames.
    let mut blur_scratch = Array3::<f32>::zeros([res, res, 3]);
    let mut frames = Array4::<f32>::zeros([t_len, res, res, 3]);

    for time_idx in 0..t_len {
        let world_x = latents[[time_idx, 0]];
        let world_y = latents[[time_idx, 1]];

        // Map world (x, y) → pixel (col, row).
        let centre_col = ((world_x - X_MIN) / space_res) as isize;
        let centre_row = ((world_y - Y_MIN) / space_res) as isize;

        // Fill circle.
        let radius_i = radius as isize;
        for dy in -radius_i..=radius_i {
            for dx in -radius_i..=radius_i {
                if dx * dx + dy * dy <= radius_i * radius_i {
                    let col = centre_col + dx;
                    let row = centre_row + dy;
                    if col >= 0 && col < res as isize && row >= 0 && row < res as isize {
                        let col_idx = col as usize;
                        let row_idx = row as usize;
                        frames[[time_idx, row_idx, col_idx, 0]] = BALL_COLOR[0];
                        frames[[time_idx, row_idx, col_idx, 1]] = BALL_COLOR[1];
                        frames[[time_idx, row_idx, col_idx, 2]] = BALL_COLOR[2];
                    }
                }
            }
        }

        // 2×2 box blur on this frame in-place using scratch buffer.
        {
            let frame = frames.index_axis(ndarray::Axis(0), time_idx);
            box_blur_2x2(frame.view(), res, &mut blur_scratch);
        }
        frames
            .index_axis_mut(ndarray::Axis(0), time_idx)
            .assign(&blur_scratch);
    }

    // Add background and clamp.
    for time_idx in 0..t_len {
        for row in 0..res {
            for col in 0..res {
                for ch in 0..3 {
                    let pixel = frames[[time_idx, row, col, ch]] + BG_COLOR[ch];
                    frames[[time_idx, row, col, ch]] = pixel.clamp(0.0, 1.0);
                }
            }
        }
    }

    frames
}

/// 2×2 box blur matching `scipy.ndimage.uniform_filter(frame, size=2, mode='constant', cval=0)`.
///
/// Each output pixel is the mean of the 2×2 block formed by the pixel and its left, top, and
/// top-left neighbours. Missing neighbours are zero-padded (never divided by fewer than 4),
/// so pixels on the top/left border are always divided by 4.
/// Ball-coloured source pixels therefore bleed one pixel to the right and downward in the output.
fn box_blur_2x2(frame: ndarray::ArrayView3<f32>, res: usize, out: &mut Array3<f32>) {
    for row in 0..res {
        for col in 0..res {
            for ch in 0..3 {
                let mut sum = frame[[row, col, ch]];
                if col > 0 {
                    sum += frame[[row, col - 1, ch]];
                }
                if row > 0 {
                    sum += frame[[row - 1, col, ch]];
                }
                if row > 0 && col > 0 {
                    sum += frame[[row - 1, col - 1, ch]];
                }
                // Always divide by 4 (zero-padding for missing neighbours).
                out[[row, col, ch]] = sum / 4.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn values_in_range() {
        let latents = arr2(&[[0.5_f32, -0.5]]);
        let frames = draw_sequence(latents.view(), 64);
        for pixel in frames.iter() {
            assert!(*pixel >= 0.0 && *pixel <= 1.0, "pixel {pixel} out of [0,1]");
        }
    }

    #[test]
    fn ball_pixel_lit() {
        let res = 64_usize;
        let space_res = (X_MAX - X_MIN) / res as f32;
        let col = ((0.5 - X_MIN) / space_res) as usize;
        let row = ((-0.5 - Y_MIN) / space_res) as usize;

        let latents = arr2(&[[0.5_f32, -0.5]]);
        let frames = draw_sequence(latents.view(), res);

        // Centre pixel should be brighter than background alone in both yellow channels.
        let r = frames[[0, row, col, 0]];
        let g = frames[[0, row, col, 1]];
        assert!(
            r > BG_COLOR[0] + 1e-3 && g > BG_COLOR[1] + 1e-3,
            "centre pixel not brighter than background: r={r} g={g}"
        );
    }

    #[test]
    fn blur_softens_edges() {
        // The blur bleeds ball colour one pixel to the right/down of the circle boundary.
        // Check that a pixel at (col+radius+1, same row as centre) is brighter than background.
        let res = 64_usize;
        let latents = arr2(&[[0.5_f32, -0.5]]);
        let frames = draw_sequence(latents.view(), res);

        let space_res = (X_MAX - X_MIN) / res as f32;
        let radius = (1.0 / space_res) as usize;
        let col = ((0.5 - X_MIN) / space_res) as usize;
        let row = ((-0.5 - Y_MIN) / space_res) as usize;

        // Pixel just outside the circle edge — should receive ball colour from the blur.
        let edge_col = (col + radius + 1).min(res - 1);
        let pixel_r = frames[[0, row, edge_col, 0]];
        let bg_r = BG_COLOR[0];
        assert!(
            pixel_r > bg_r + 1e-3,
            "blur had no measurable effect on edge pixel: {pixel_r} vs bg {bg_r}"
        );
    }
}
