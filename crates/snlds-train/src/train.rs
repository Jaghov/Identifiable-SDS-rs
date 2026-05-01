//! Adam minibatch training loop for [`VariationalSnlds`].

use crate::data::ObsTensor;
use crate::snapshot::{TrainSnapshot, DEFAULT_OBS_NOISE_VAR, TRAIN_SNAPSHOT_SCHEMA_VERSION};
use anyhow::Context;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use snlds_model::{EncoderKind, SnldsConfig, VariationalSnlds};
use std::path::{Path, PathBuf};

/// Hyper-parameters and IO paths for one training run.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub beta: f32,
    pub temperature: f32,
    pub grad_clip: f32,
    pub checkpoint_every: usize,
    pub hidden_dim: usize,
    /// Variance of the diagonal Gaussian observation noise used in the ELBO
    /// reconstruction term. Persisted in `train_config.json` so `snlds-eval` can
    /// reproduce the same number at inference time.
    pub obs_noise_var: f32,
    pub seed: u64,
    pub resume_from: Option<PathBuf>,
    /// Encoder/decoder family forwarded to `SnldsConfig::kind`. Default is
    /// `EncoderKind::Mlp` (Python-`factored` parity); set to
    /// `EncoderKind::Cnn { res }` to train on flat-RGB image observations.
    pub kind: EncoderKind,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("checkpoints"),
            epochs: 100,
            batch_size: 32,
            learning_rate: 3e-4,
            beta: 1.0,
            temperature: 1.0,
            grad_clip: 1.0,
            checkpoint_every: 10,
            hidden_dim: 64,
            obs_noise_var: DEFAULT_OBS_NOISE_VAR,
            seed: 0,
            resume_from: None,
            kind: EncoderKind::default(),
        }
    }
}

/// Diagnostics returned by [`train`] for each epoch.
#[derive(Clone, Debug)]
pub struct EpochStats {
    pub epoch: usize,
    pub mean_loss: f32,
    pub mean_recon: f32,
}

/// Build a fresh `VariationalSnlds` from `manifest` + `config.hidden_dim`.
///
/// Forwards `config.kind` to [`SnldsConfig::kind`]; for
/// [`EncoderKind::Cnn { res }`] callers must ensure `manifest.dim_obs ==
/// 3 * res * res`. The model `init` panics with a descriptive message if not.
pub fn build_model_config(config: &TrainConfig, manifest: &snlds_data::Manifest) -> SnldsConfig {
    SnldsConfig::new(
        manifest.dim_obs,
        manifest.dim_latent,
        config.hidden_dim,
        manifest.num_states,
    )
    .with_kind(config.kind.clone())
}

/// Run the training loop with a freshly initialised model. See
/// [`train_with_model`] for the variant that accepts an externally prepared
/// model (e.g. from M5 warm-start).
pub fn train<B: AutodiffBackend>(
    config: &TrainConfig,
    obs_tensor: ObsTensor<B>,
    device: &B::Device,
) -> anyhow::Result<Vec<EpochStats>> {
    let model_config = build_model_config(config, &obs_tensor.manifest);
    let model: VariationalSnlds<B> = model_config.init(device);
    train_with_model(config, model, obs_tensor, device)
}

/// Run the training loop starting from `initial_model`. Used by the M5 warm-start
/// path so the caller can hand in a model whose parameters were transferred from
/// a fitted [`snlds_msm::NeuralMsm`].
pub fn train_with_model<B: AutodiffBackend>(
    config: &TrainConfig,
    initial_model: VariationalSnlds<B>,
    obs_tensor: ObsTensor<B>,
    device: &B::Device,
) -> anyhow::Result<Vec<EpochStats>> {
    let manifest = &obs_tensor.manifest;
    let mut model = initial_model;

    if let Some(path) = config.resume_from.as_ref() {
        let recorder = CompactRecorder::new();
        let record = recorder
            .load(path.clone(), device)
            .with_context(|| format!("load checkpoint {:?}", path))?;
        model = model.load_record(record);
    }

    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(config.grad_clip)))
        .init::<B, VariationalSnlds<B>>();

    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create output dir {:?}", config.output_dir))?;

    let snapshot = TrainSnapshot {
        schema_version: TRAIN_SNAPSHOT_SCHEMA_VERSION,
        hidden_dim: config.hidden_dim,
        beta: config.beta,
        temperature: config.temperature,
        obs_noise_var: config.obs_noise_var,
        kind: config.kind.clone(),
    };
    snapshot
        .save(&config.output_dir)
        .context("write train_config.json snapshot")?;

    let num_sequences = manifest.num_samples;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let obs_full = obs_tensor.obs;

    let mut history = Vec::with_capacity(config.epochs);

    for epoch_idx in 0..config.epochs {
        let mut sequence_order: Vec<usize> = (0..num_sequences).collect();
        sequence_order.shuffle(&mut rng);

        let mut epoch_loss_sum = 0.0_f32;
        let mut epoch_recon_sum = 0.0_f32;
        let mut step_count = 0_usize;

        for batch_indices in sequence_order.chunks(config.batch_size) {
            let batch_obs = gather_batch(obs_full.clone(), batch_indices);

            let output = model.forward(
                batch_obs,
                config.beta,
                config.obs_noise_var,
                config.temperature,
            );
            let loss = output.elbo.clone().neg().sum();
            let recon_value = scalar_value(&output.recon_loss);
            let loss_value = scalar_value(&loss);

            let gradients = loss.backward();
            let grad_params = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(config.learning_rate, model, grad_params);

            epoch_loss_sum += loss_value;
            epoch_recon_sum += recon_value;
            step_count += 1;
        }

        let stats = EpochStats {
            epoch: epoch_idx,
            mean_loss: epoch_loss_sum / step_count.max(1) as f32,
            mean_recon: epoch_recon_sum / step_count.max(1) as f32,
        };
        println!(
            "epoch {:04}: mean_loss={:.4} mean_recon_log_prob={:.4}",
            stats.epoch, stats.mean_loss, stats.mean_recon
        );

        if config.checkpoint_every > 0 && (epoch_idx + 1) % config.checkpoint_every == 0 {
            save_checkpoint(&model, &config.output_dir, epoch_idx)?;
        }
        history.push(stats);
    }

    save_checkpoint(&model, &config.output_dir, config.epochs.saturating_sub(1))?;
    Ok(history)
}

fn gather_batch<B: AutodiffBackend>(obs: Tensor<B, 3>, indices: &[usize]) -> Tensor<B, 3> {
    let slices: Vec<Tensor<B, 3>> = indices
        .iter()
        .map(|&seq_idx| {
            let [_n, seq_len, obs_dim] = obs.dims();
            obs.clone()
                .slice([seq_idx..seq_idx + 1, 0..seq_len, 0..obs_dim])
        })
        .collect();
    Tensor::cat(slices, 0)
}

fn scalar_value<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map(|values| values.first().copied().unwrap_or(f32::NAN))
        .unwrap_or(f32::NAN)
}

fn save_checkpoint<B: AutodiffBackend>(
    model: &VariationalSnlds<B>,
    output_dir: &Path,
    epoch_idx: usize,
) -> anyhow::Result<()> {
    let path = output_dir.join(format!("checkpoint_{epoch_idx:04}.mpk"));
    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(path.clone(), &recorder)
        .with_context(|| format!("save checkpoint {:?}", path))?;
    Ok(())
}
