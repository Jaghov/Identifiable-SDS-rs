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

/// Log posterior state marginals γ_{t,k} for one sequence.
///
/// `gamma`: `[T, K]` — each row should sum to 1. Logs one `Scalars` entity per state
/// under `snlds/state/gamma_{k}` at each timestep.
pub fn log_posteriors(
    rec: &RecordingStream,
    seq_idx: i64,
    gamma: ArrayView2<f32>,
) -> anyhow::Result<()> {
    let num_timesteps = gamma.nrows();
    let num_states = gamma.ncols();
    for timestep in 0..num_timesteps {
        rec.set_time_sequence("sequence", seq_idx);
        rec.set_time_sequence("time", timestep as i64);
        for state_idx in 0..num_states {
            let entity_path = format!("snlds/state/gamma_{state_idx}");
            rec.log(
                entity_path.as_str(),
                &rerun::Scalars::single(gamma[[timestep, state_idx]] as f64),
            )
            .context("log snlds/state/gamma_k")?;
        }
    }
    Ok(())
}

/// Log reconstructed observations for one sequence.
///
/// `x_hat`: `[T, obs_dim]`.
/// - `obs_dim == 2`: logs a single `LineStrips2D` under `snlds/obs/x_hat`
/// - Otherwise: logs each dimension as `Scalars` under `snlds/obs/x_hat_d{dim_idx}` per timestep
pub fn log_reconstructions(
    rec: &RecordingStream,
    seq_idx: i64,
    x_hat: ArrayView2<f32>,
) -> anyhow::Result<()> {
    rec.set_time_sequence("sequence", seq_idx);
    let obs_dim = x_hat.ncols();
    if obs_dim == 2 {
        let points: Vec<[f32; 2]> = x_hat
            .rows()
            .into_iter()
            .map(|row| [row[0], row[1]])
            .collect();
        rec.log("snlds/obs/x_hat", &rerun::LineStrips2D::new([points]))
            .context("log snlds/obs/x_hat")?;
    } else {
        let num_timesteps = x_hat.nrows();
        for timestep in 0..num_timesteps {
            rec.set_time_sequence("sequence", seq_idx);
            rec.set_time_sequence("time", timestep as i64);
            for dim_idx in 0..obs_dim {
                let entity_path = format!("snlds/obs/x_hat_d{dim_idx}");
                rec.log(
                    entity_path.as_str(),
                    &rerun::Scalars::single(x_hat[[timestep, dim_idx]] as f64),
                )
                .context("log snlds/obs/x_hat_d*")?;
            }
        }
    }
    Ok(())
}

/// Log training diagnostics for one optimizer step.
///
/// Uses the `train_step` timeline — does not set `sequence` or `time`.
pub fn log_train_scalars(
    rec: &RecordingStream,
    step: i64,
    elbo: f32,
    mse: f32,
    temperature: f32,
) -> anyhow::Result<()> {
    rec.set_time_sequence("train_step", step);
    rec.log("snlds/train/elbo", &rerun::Scalars::single(elbo as f64))
        .context("log snlds/train/elbo")?;
    rec.log("snlds/train/mse", &rerun::Scalars::single(mse as f64))
        .context("log snlds/train/mse")?;
    rec.log(
        "snlds/train/temperature",
        &rerun::Scalars::single(temperature as f64),
    )
    .context("log snlds/train/temperature")?;
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
