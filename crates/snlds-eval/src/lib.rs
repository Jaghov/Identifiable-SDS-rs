//! Inference + Rerun logging for a trained `VariationalSnlds` checkpoint.
//!
//! The library exposes a single entry point [`run_eval`] used by the `snlds-eval`
//! binary; the binary just translates `clap` arguments into [`EvalConfig`].
//!
//! Outputs logged to Rerun (per `--sequences`):
//! - `snlds/markov/q_inferred` and `snlds/markov/q_inferred/weights` — softmax(`q_logits`).
//! - `snlds/state/strip_inferred` — argmax(γ) per timestep, as a colored band.
//! - `snlds/state/gamma` — posterior `γ` heatmap (Figure 6 of arXiv:2305.15925).
//! - `snlds/obs/x_hat[_d{d}]` — model reconstructions per sequence.
//!
//! The companion `snlds-viz` binary still owns the ground-truth side
//! (`q_true`, `strip_true`); a typical workflow logs both into the same `.rrd`
//! by running `snlds-viz` first and `snlds-eval` second with `--append`.

use anyhow::Context;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, Tensor};
use ndarray::{Array2, Array3, Axis};
use snlds_data::Manifest;
use snlds_model::{SnldsConfig, VariationalSnlds};
use snlds_train::data::load_train_obs;
use std::path::PathBuf;

/// Configuration for one evaluation run.
#[derive(Clone, Debug)]
pub struct EvalConfig {
    /// Directory containing `sequences.safetensors` + `metadata.json`.
    pub data_dir: PathBuf,
    /// Path to a `CompactRecorder` checkpoint produced by `snlds-train`.
    pub checkpoint: PathBuf,
    /// Recording stream output (file path) for new recordings; ignored when `spawn=true`.
    pub output: PathBuf,
    /// If `true`, spawn the live Rerun viewer instead of writing to `output`.
    pub spawn: bool,
    /// Number of sequences from `obs_train` to log.
    pub sequences: usize,
    /// MLP hidden dimension used during training (must match the checkpoint).
    pub hidden_dim: usize,
    /// Softmax temperature for `q_logits` / `pi_logits` (matches the training value used at inference).
    pub temperature: f32,
}

/// Run inference + Rerun logging end-to-end.
pub fn run_eval<B: Backend>(config: &EvalConfig, device: &B::Device) -> anyhow::Result<()> {
    let obs_tensor = load_train_obs::<B>(&config.data_dir, device)
        .with_context(|| format!("load obs_train from {:?}", config.data_dir))?;
    let manifest = obs_tensor.manifest.clone();
    let model = load_checkpoint::<B>(config, &manifest, device)?;

    let num_seqs = config.sequences.min(manifest.num_samples);
    if num_seqs == 0 {
        anyhow::bail!(
            "--sequences resolved to 0 (manifest has {} samples)",
            manifest.num_samples
        );
    }

    // Subset to the first `num_seqs` sequences so the forward pass stays cheap.
    let obs_subset =
        obs_tensor
            .obs
            .clone()
            .slice([0..num_seqs, 0..manifest.seq_length, 0..manifest.dim_obs]);

    // beta > 0 guarantees state_posteriors is populated.
    let forward_output = model.forward(
        obs_subset,
        /* beta = */ 1.0,
        /* obs_noise_var = */ 0.1,
        config.temperature,
    );
    let gamma = forward_output
        .state_posteriors
        .ok_or_else(|| anyhow::anyhow!("forward produced no posteriors (beta must be > 0)"))?;
    let x_hat = forward_output.obs_reconstructed;

    // Q = softmax(q_logits / temperature, dim=-1)  — same convention as the model's forward pass.
    let q_inferred_tensor = softmax(model.q_logits.val() / config.temperature, 1);
    let q_inferred = tensor2_to_array(q_inferred_tensor)?;
    let gamma_array = tensor3_to_array(gamma)?;
    let x_hat_array = tensor3_to_array(x_hat)?;

    let recording = if config.spawn {
        rerun::RecordingStreamBuilder::new("snlds-eval")
            .spawn()
            .context("spawn Rerun viewer")?
    } else {
        rerun::RecordingStreamBuilder::new("snlds-eval")
            .save(&config.output)
            .with_context(|| format!("open output {:?}", config.output))?
    };

    snlds_viz::log_transition_matrix(&recording, "snlds/markov/q_inferred", q_inferred.view())?;

    for seq_idx in 0..num_seqs {
        let seq_idx_i64 = seq_idx as i64;
        recording.set_time_sequence("sequence", seq_idx_i64);

        let gamma_seq = gamma_array.index_axis(Axis(0), seq_idx);
        let x_hat_seq = x_hat_array.index_axis(Axis(0), seq_idx);

        snlds_viz::log_gamma_heatmap(&recording, "snlds/state/gamma", gamma_seq.view())?;

        let inferred_states: Vec<i32> = gamma_seq
            .axis_iter(Axis(0))
            .map(|row| {
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (state_idx, &value) in row.iter().enumerate() {
                    if value > best_val {
                        best_val = value;
                        best_idx = state_idx;
                    }
                }
                best_idx as i32
            })
            .collect();
        snlds_viz::log_state_strip(&recording, "snlds/state/strip_inferred", &inferred_states)?;

        snlds_viz::log_posteriors(&recording, seq_idx_i64, gamma_seq.view())?;
        snlds_viz::log_reconstructions(&recording, seq_idx_i64, x_hat_seq.view())?;
    }

    if !config.spawn {
        println!("Saved to {:?}", config.output);
    }
    Ok(())
}

fn load_checkpoint<B: Backend>(
    config: &EvalConfig,
    manifest: &Manifest,
    device: &B::Device,
) -> anyhow::Result<VariationalSnlds<B>> {
    let snlds_config = SnldsConfig::new(
        manifest.dim_obs,
        manifest.dim_latent,
        config.hidden_dim,
        manifest.num_states,
    );
    let model: VariationalSnlds<B> = snlds_config.init(device);
    let recorder = CompactRecorder::new();
    let record = recorder
        .load(config.checkpoint.clone(), device)
        .with_context(|| format!("load checkpoint {:?}", config.checkpoint))?;
    Ok(model.load_record(record))
}

fn tensor2_to_array<B: Backend>(tensor: Tensor<B, 2>) -> anyhow::Result<Array2<f32>> {
    let [rows, cols] = tensor.dims();
    let data = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("convert 2-D tensor to f32 vec failed: {err:?}"))?;
    Array2::from_shape_vec((rows, cols), data).context("reshape into Array2")
}

fn tensor3_to_array<B: Backend>(tensor: Tensor<B, 3>) -> anyhow::Result<Array3<f32>> {
    let [batch_size, seq_len, last_dim] = tensor.dims();
    let data = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("convert 3-D tensor to f32 vec failed: {err:?}"))?;
    Array3::from_shape_vec((batch_size, seq_len, last_dim), data).context("reshape into Array3")
}
