//! Adam minibatch training loop for [`VariationalSnlds`].

use crate::data::ObsTensor;
use crate::snapshot::{TrainSnapshot, DEFAULT_OBS_NOISE_VAR, TRAIN_SNAPSHOT_SCHEMA_VERSION};
use anyhow::Context;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
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
    /// Print a line every N minibatches within each epoch (`0` = off; `1` = every batch).
    pub log_every_batch: usize,
    /// Also print the learned row-stochastic `Q` every N minibatches (`0` = epoch end only).
    pub transition_log_every_batches: usize,
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
            log_every_batch: 1,
            transition_log_every_batches: 0,
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
pub fn train<B: AutodiffBackend + Backend<FloatElem = f32>>(
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
pub fn train_with_model<B: AutodiffBackend + Backend<FloatElem = f32>>(
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
        flow_snlds: None,
    };
    snapshot
        .save(&config.output_dir)
        .context("write train_config.json snapshot")?;

    let num_sequences = manifest.num_samples;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let obs_full = obs_tensor.obs;

    let mut history = Vec::with_capacity(config.epochs);

    crate::training_log::log_true_transition_matrix_from_data(
        &config.data_dir,
        manifest.num_states,
    );

    for epoch_idx in 0..config.epochs {
        let mut sequence_order: Vec<usize> = (0..num_sequences).collect();
        sequence_order.shuffle(&mut rng);

        let n_batches = sequence_order.chunks(config.batch_size).count();
        let mut epoch_loss_sum = 0.0_f32;
        let mut epoch_recon_sum = 0.0_f32;
        let mut step_count = 0_usize;

        for (batch_i, batch_indices) in sequence_order.chunks(config.batch_size).enumerate() {
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

            if crate::training_log::should_log_minibatch(config.log_every_batch, batch_i, n_batches)
            {
                println!(
                    "epoch {:04} batch {:04}/{} loss={:.4} recon_log_prob={:.4}",
                    epoch_idx,
                    batch_i + 1,
                    n_batches,
                    loss_value,
                    recon_value
                );
            }

            let gradients = loss.backward();
            let grad_params = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(config.learning_rate, model, grad_params);

            let t_every = config.transition_log_every_batches;
            if crate::training_log::should_log_transition_every_n_batches(t_every, batch_i) {
                crate::training_log::log_learned_transition_matrix(
                    "",
                    epoch_idx,
                    model.q_logits.val(),
                    config.temperature,
                    Some((batch_i + 1, n_batches)),
                );
            }

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
            "epoch {:04} end: mean_loss={:.4} mean_recon_log_prob={:.4}",
            stats.epoch, stats.mean_loss, stats.mean_recon
        );
        crate::training_log::log_learned_transition_matrix(
            "",
            epoch_idx,
            model.q_logits.val(),
            config.temperature,
            None,
        );

        log_checkpoint_recon_mse(&model, obs_full.clone(), num_sequences, config, epoch_idx);

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

fn log_checkpoint_recon_mse<B: AutodiffBackend + Backend<FloatElem = f32>>(
    model: &VariationalSnlds<B>,
    obs_full: Tensor<B, 3>,
    num_sequences: usize,
    config: &TrainConfig,
    epoch_idx: usize,
) {
    let n = config.batch_size.min(num_sequences);
    if n == 0 {
        return;
    }
    let idx: Vec<usize> = (0..n).collect();
    let batch_obs = gather_batch(obs_full, &idx);
    let out = model.forward(
        batch_obs.clone(),
        config.beta,
        config.obs_noise_var,
        config.temperature,
    );
    let mse = crate::checkpoint_recon::tensor_mean_mse(batch_obs, out.obs_reconstructed);
    let rmse = mse.sqrt();
    println!(
        "checkpoint {:04} recon_mse={:.6} recon_rmse={:.6}",
        epoch_idx, mse, rmse
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
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

#[cfg(test)]
mod transition_log_schedule_tests {
    use super::{train, TrainConfig};
    use crate::data::load_train_obs;
    use crate::training_log;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use serial_test::serial;
    use snlds_data::{
        generate_train_test, save_train_test, GenConfig, Manifest, SimulatorKind,
        MANIFEST_SCHEMA_VERSION,
    };
    use snlds_model::EncoderKind;
    use std::path::Path;

    type B = Autodiff<NdArray<f32>>;

    fn tiny_train_dir(dir: &Path) {
        let cfg = GenConfig {
            seed: 7,
            num_states: 3,
            dim_obs: 4,
            dim_latent: 2,
            seq_length: 5,
            num_samples: 4,
            sparsity_prob: 0.0,
            kind: SimulatorKind::Poly,
            poly_degree: 2,
            ..GenConfig::default()
        };
        let manifest = Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            seed: cfg.seed,
            num_states: cfg.num_states,
            dim_obs: cfg.dim_obs,
            dim_latent: cfg.dim_latent,
            seq_length: cfg.seq_length,
            num_samples: cfg.num_samples,
            sparsity_prob: cfg.sparsity_prob,
            data_type: "poly".into(),
            degree: Some(cfg.poly_degree),
            init_noise_std: cfg.init_noise_std,
            init_mean_std: cfg.init_mean_std,
            transition_step_var: cfg.transition_step_var,
            emission_hidden_dim: cfg.emission_hidden_dim,
            num_samples_eval: 0,
        };
        let train_test = generate_train_test(&cfg).expect("generate");
        save_train_test(dir, &train_test, &manifest).expect("save");
    }

    #[test]
    #[serial]
    fn transition_log_every_batch_triggers_mid_epoch_q() {
        training_log::reset_q_log_counters_for_test();
        let data_dir = tempfile::tempdir().expect("data");
        let output_dir = tempfile::tempdir().expect("out");
        tiny_train_dir(data_dir.path());
        let device = NdArrayDevice::default();
        let obs = load_train_obs::<B>(data_dir.path(), &device).expect("load");
        let config = TrainConfig {
            data_dir: data_dir.path().into(),
            output_dir: output_dir.path().into(),
            epochs: 1,
            batch_size: 1,
            transition_log_every_batches: 1,
            checkpoint_every: 0,
            log_every_batch: 0,
            learning_rate: 3e-4,
            beta: 1.0,
            temperature: 1.0,
            grad_clip: 1.0,
            hidden_dim: 8,
            obs_noise_var: 5e-4,
            seed: 0,
            resume_from: None,
            kind: EncoderKind::Mlp,
        };
        train(&config, obs, &device).expect("train");
        assert_eq!(training_log::q_log_mid_epoch_count_for_test(), 4);
        assert_eq!(training_log::q_log_epoch_end_count_for_test(), 1);
    }

    #[test]
    #[serial]
    fn transition_log_every_two_batches() {
        training_log::reset_q_log_counters_for_test();
        let data_dir = tempfile::tempdir().expect("data");
        let output_dir = tempfile::tempdir().expect("out");
        tiny_train_dir(data_dir.path());
        let device = NdArrayDevice::default();
        let obs = load_train_obs::<B>(data_dir.path(), &device).expect("load");
        let config = TrainConfig {
            data_dir: data_dir.path().into(),
            output_dir: output_dir.path().into(),
            epochs: 1,
            batch_size: 1,
            transition_log_every_batches: 2,
            checkpoint_every: 0,
            log_every_batch: 0,
            learning_rate: 3e-4,
            beta: 1.0,
            temperature: 1.0,
            grad_clip: 1.0,
            hidden_dim: 8,
            obs_noise_var: 5e-4,
            seed: 0,
            resume_from: None,
            kind: EncoderKind::Mlp,
        };
        train(&config, obs, &device).expect("train");
        assert_eq!(training_log::q_log_mid_epoch_count_for_test(), 2);
        assert_eq!(training_log::q_log_epoch_end_count_for_test(), 1);
    }
}
