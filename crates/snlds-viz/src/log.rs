use anyhow::Context;
use ndarray::{ArrayView2, ArrayView4};
use rerun::RecordingStream;

/// Log a 2-D latent trajectory for one sequence as a `LineStrips2D`.
///
/// `latents`: `[T, 2]`. Only call when `dim_latent == 2`.
pub fn log_latent_z(
    rec: &RecordingStream,
    seq_idx: i64,
    latents: ArrayView2<f32>,
) -> anyhow::Result<()> {
    rec.set_time_sequence("sequence", seq_idx);
    let points: Vec<[f32; 2]> = latents.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    rec.log("snlds/latent/z", &rerun::LineStrips2D::new([points]))
        .context("log snlds/latent/z")
}

/// Log a 2-D observation trajectory for one sequence as a `LineStrips2D`.
///
/// `obs`: `[T, 2]`. Only call when `dim_obs == 2`.
pub fn log_obs_x(rec: &RecordingStream, seq_idx: i64, obs: ArrayView2<f32>) -> anyhow::Result<()> {
    rec.set_time_sequence("sequence", seq_idx);
    let points: Vec<[f32; 2]> = obs.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    rec.log("snlds/obs/x", &rerun::LineStrips2D::new([points]))
        .context("log snlds/obs/x")
}

/// Log discrete state `s_t` per timestep as a `Scalars` timeseries.
pub fn log_state_s(rec: &RecordingStream, seq_idx: i64, states: &[i32]) -> anyhow::Result<()> {
    for (t, &state) in states.iter().enumerate() {
        rec.set_time_sequence("sequence", seq_idx);
        rec.set_time_sequence("time", t as i64);
        rec.log("snlds/state/s", &rerun::Scalars::single(state as f64))
            .context("log snlds/state/s")?;
    }
    Ok(())
}

/// Log rendered frames (from `draw_sequence`) per timestep as `Image` entities.
///
/// `frames`: shape `[T, res, res, 3]`, values ∈ `[0, 1]`.
pub fn log_render_frames(
    rec: &RecordingStream,
    seq_idx: i64,
    frames: ArrayView4<f32>,
) -> anyhow::Result<()> {
    let t_len = frames.shape()[0];
    let res = frames.shape()[1];

    for t in 0..t_len {
        rec.set_time_sequence("sequence", seq_idx);
        rec.set_time_sequence("time", t as i64);

        let frame_slice = frames.index_axis(ndarray::Axis(0), t);
        let u8_pixels: Vec<u8> = frame_slice
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();

        let image = rerun::Image::from_elements(
            &u8_pixels,
            [res as u32, res as u32],
            rerun::ColorModel::RGB,
        );

        rec.log("snlds/render/frame", &image)
            .context("log snlds/render/frame")?;
    }
    Ok(())
}
