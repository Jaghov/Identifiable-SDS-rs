use crate::colormap::{state_color, viridis_rgb, STATE_STRIP_HEIGHT, TRANSITION_EDGE_EPSILON};
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

/// Log a `[K, K]` row-stochastic transition matrix `Q` as a Rerun graph
/// (`GraphNodes` + `GraphEdges`).
///
/// Nodes are labeled `s0..s{K-1}` and colored with [`state_color`]. Edges are
/// directed `i → j` with weight `Q[i,j]`, and edges with `|w| < TRANSITION_EDGE_EPSILON`
/// are dropped so dense / numerically-noisy matrices stay legible.
///
/// The same `entity_path` should be reused across calls (e.g. `snlds/markov/q_true`
/// and `snlds/markov/q_inferred` for side-by-side comparison).
pub fn log_transition_matrix(
    rec: &RecordingStream,
    entity_path: &str,
    q_matrix: ArrayView2<f32>,
) -> anyhow::Result<()> {
    let num_states = q_matrix.nrows();
    anyhow::ensure!(
        q_matrix.ncols() == num_states,
        "transition matrix must be square, got {:?}",
        q_matrix.dim()
    );

    let node_ids: Vec<String> = (0..num_states)
        .map(|state_idx| format!("s{state_idx}"))
        .collect();
    let node_colors: Vec<[u8; 3]> = (0..num_states).map(state_color).collect();

    rec.log(
        entity_path,
        &rerun::GraphNodes::new(node_ids.clone())
            .with_labels(node_ids.clone())
            .with_colors(node_colors),
    )
    .with_context(|| format!("log graph nodes at {entity_path}"))?;

    let mut edges: Vec<(String, String)> = Vec::new();
    for from_idx in 0..num_states {
        for to_idx in 0..num_states {
            let weight = q_matrix[[from_idx, to_idx]];
            if weight.abs() < TRANSITION_EDGE_EPSILON {
                continue;
            }
            edges.push((node_ids[from_idx].clone(), node_ids[to_idx].clone()));
        }
    }

    rec.log(
        entity_path,
        &rerun::GraphEdges::new(edges).with_graph_type(rerun::components::GraphType::Directed),
    )
    .with_context(|| format!("log graph edges at {entity_path}"))?;

    // Edge weight labels aren't supported by `GraphEdges` in rerun 0.31, so log the full
    // `Q` matrix as a sibling heatmap entity (`{entity_path}/weights`) so users can see
    // exact transition probabilities alongside the graph.
    log_q_weight_heatmap(rec, &format!("{entity_path}/weights"), q_matrix)?;
    Ok(())
}

fn log_q_weight_heatmap(
    rec: &RecordingStream,
    entity_path: &str,
    q_matrix: ArrayView2<f32>,
) -> anyhow::Result<()> {
    let num_states = q_matrix.nrows();
    let mut rgb_bytes: Vec<u8> = Vec::with_capacity(num_states * num_states * 3);
    for from_idx in 0..num_states {
        for to_idx in 0..num_states {
            let color = viridis_rgb(q_matrix[[from_idx, to_idx]]);
            rgb_bytes.extend_from_slice(&color);
        }
    }
    let image = rerun::Image::from_color_model_and_bytes(
        rgb_bytes,
        [num_states as u32, num_states as u32],
        rerun::ColorModel::RGB,
        rerun::ChannelDatatype::U8,
    );
    rec.log(entity_path, &image)
        .with_context(|| format!("log Q heatmap at {entity_path}"))
}

/// Log a sequence of integer states as a horizontal Figure-6-style colored band.
///
/// Renders a `STATE_STRIP_HEIGHT × T × 3` `Image`: each column is a timestep, each
/// row tile shares the same color so the band is visible (a `1 × T` image is too
/// thin to read in Rerun's image view).
///
/// Negative state ids are clamped to 0 (defensive — `i32` payloads from SafeTensors
/// should always be non-negative state ids).
pub fn log_state_strip(
    rec: &RecordingStream,
    entity_path: &str,
    states: &[i32],
) -> anyhow::Result<()> {
    let num_timesteps = states.len();
    anyhow::ensure!(num_timesteps > 0, "log_state_strip: states is empty");

    let strip_height = STATE_STRIP_HEIGHT as usize;
    let mut rgb_bytes: Vec<u8> = Vec::with_capacity(strip_height * num_timesteps * 3);
    for _ in 0..strip_height {
        for &state in states {
            let color = state_color(state.max(0) as usize);
            rgb_bytes.extend_from_slice(&color);
        }
    }

    let image = rerun::Image::from_color_model_and_bytes(
        rgb_bytes,
        [num_timesteps as u32, STATE_STRIP_HEIGHT],
        rerun::ColorModel::RGB,
        rerun::ChannelDatatype::U8,
    );
    rec.log(entity_path, &image)
        .with_context(|| format!("log state strip at {entity_path}"))
}

/// Log posterior state marginals `γ_{t,k}` as a `K × T` viridis heatmap (Figure 6
/// of arXiv:2305.15925, bottom panel).
///
/// `gamma`: `[T, K]`. Each row in the output image corresponds to one state
/// (top-to-bottom is `k = 0..K-1`); each column to one timestep. Values are
/// clamped to `[0, 1]` by [`viridis_rgb`].
pub fn log_gamma_heatmap(
    rec: &RecordingStream,
    entity_path: &str,
    gamma: ArrayView2<f32>,
) -> anyhow::Result<()> {
    let num_timesteps = gamma.nrows();
    let num_states = gamma.ncols();
    anyhow::ensure!(
        num_timesteps > 0 && num_states > 0,
        "log_gamma_heatmap: empty gamma {:?}",
        gamma.dim()
    );

    let mut rgb_bytes: Vec<u8> = Vec::with_capacity(num_timesteps * num_states * 3);
    for state_idx in 0..num_states {
        for time_idx in 0..num_timesteps {
            let color = viridis_rgb(gamma[[time_idx, state_idx]]);
            rgb_bytes.extend_from_slice(&color);
        }
    }

    let image = rerun::Image::from_color_model_and_bytes(
        rgb_bytes,
        [num_timesteps as u32, num_states as u32],
        rerun::ColorModel::RGB,
        rerun::ChannelDatatype::U8,
    );
    rec.log(entity_path, &image)
        .with_context(|| format!("log gamma heatmap at {entity_path}"))
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
