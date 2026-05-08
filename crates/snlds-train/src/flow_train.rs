//! Training loop for joint [`snlds_model::FlowSnlds`] (Neural PCA + switching prior).

use crate::data::{SequenceBatcher, SequenceDataset};
use crate::snapshot::{FlowSnldsSnapshotMeta, TrainSnapshot, TRAIN_SNAPSHOT_SCHEMA_VERSION};
use anyhow::Context;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::Dataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module, Param};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Tensor};
use snlds_model::{
    glow_flattened_latent_dim, log_p_z_isotropic, CouplingType, FlowSnlds, FlowSnldsConfig,
    PcaSvdBackend, TriangularInverse,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct FlowTrainConfig {
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub temperature: f32,
    pub grad_clip: f32,
    pub checkpoint_every: usize,
    pub hidden_dim: usize,
    pub obs_noise_var: f32,
    pub seed: u64,
    pub resume_from: Option<PathBuf>,
    pub res: usize,
    pub w_msm: f32,
    pub w_npca: f32,
    pub glow_levels: usize,
    pub glow_steps: usize,
    pub glow_hidden_features: usize,
    pub glow_coupling: CouplingType,
    /// Print every N minibatches per epoch (`0` = off).
    pub log_every_batch: usize,
    pub weight_decay: f32,
    /// Print learned row-stochastic `Q` every N minibatches (`0` = epoch end only).
    pub transition_log_every_batches: usize,
    pub npca_householder: bool,
    pub npca_householder_reflectors: usize,
}

#[derive(Clone, Debug)]
pub struct FlowEpochStats {
    pub epoch: usize,
    pub mean_loss: f32,
    pub mean_joint: f32,
    /// Mean per minibatch of `sum_{n,t} log|det ∂z/∂x_{n,t}| / N` (same `N` as [`FlowForwardOutput::npca_loglik`]).
    pub mean_logdet_sum_over_n: f32,
    /// Mean per minibatch of `sum_{n,t} log p(z_r) / N` (isotropic Gaussian residual).
    pub mean_log_p_tail_sum_over_n: f32,
    pub mean_msm_loglik: f32,
    pub mean_npca_loglik: f32,
}

pub fn build_flow_config(
    config: &FlowTrainConfig,
    manifest: &snlds_data::Manifest,
) -> FlowSnldsConfig {
    let obs_dim = manifest.dim_obs;
    assert_eq!(
        obs_dim,
        3 * config.res * config.res,
        "FlowSnlds expects dim_obs == 3*res*res"
    );
    FlowSnldsConfig::new(
        obs_dim,
        manifest.dim_latent,
        config.hidden_dim,
        manifest.num_states,
        config.res,
        config.glow_levels,
        config.glow_steps,
        config.glow_hidden_features,
    )
    .with_coupling_type(config.glow_coupling)
    .with_householder_rotation(config.npca_householder)
    .with_householder_reflectors(config.npca_householder_reflectors)
}

/// Train from a [`SequenceDataset`] using Burn's DataLoader for batch-level streaming.
pub fn train_flow_from_dataset<B>(
    config: &FlowTrainConfig,
    dataset: SequenceDataset,
    val_dataset: Option<SequenceDataset>,
    device: &B::Device,
) -> anyhow::Result<Vec<FlowEpochStats>>
where
    B: AutodiffBackend + Backend<FloatElem = f32> + PcaSvdBackend + TriangularInverse,
    B::InnerBackend: PcaSvdBackend + Backend<FloatElem = f32> + TriangularInverse,
{
    let flow_config = build_flow_config(config, &dataset.manifest);
    let model: FlowSnlds<B> = flow_config.init(device);
    train_flow_with_dataset(config, model, dataset, val_dataset, device)
}

pub fn train_flow_with_dataset<B>(
    config: &FlowTrainConfig,
    initial_model: FlowSnlds<B>,
    dataset: SequenceDataset,
    val_dataset: Option<SequenceDataset>,
    device: &B::Device,
) -> anyhow::Result<Vec<FlowEpochStats>>
where
    B: AutodiffBackend + Backend<FloatElem = f32> + PcaSvdBackend + TriangularInverse,
    B::InnerBackend: PcaSvdBackend + Backend<FloatElem = f32> + TriangularInverse,
{
    let manifest = dataset.manifest.clone();
    let mut model = initial_model;

    if let Some(path) = config.resume_from.as_ref() {
        let recorder = CompactRecorder::new();
        let record = recorder
            .load(path.clone(), device)
            .with_context(|| format!("load FlowSNLDS checkpoint {:?}", path))?;
        model = model.load_record(record);
        model.npca.sync_training_mode_after_load();
    }

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(config.grad_clip)))
        .with_weight_decay(config.weight_decay)
        .init::<B, FlowSnlds<B>>();

    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create output dir {:?}", config.output_dir))?;

    let total_d = glow_flattened_latent_dim(3, config.glow_levels, config.res, config.res);

    let snapshot = TrainSnapshot {
        schema_version: TRAIN_SNAPSHOT_SCHEMA_VERSION,
        hidden_dim: config.hidden_dim,
        beta: 0.0,
        temperature: config.temperature,
        obs_noise_var: config.obs_noise_var,
        kind: snlds_model::EncoderKind::Cnn { res: config.res },
        flow_snlds: Some(FlowSnldsSnapshotMeta {
            w_msm: config.w_msm,
            w_npca: config.w_npca,
            res: config.res,
            glow_levels: config.glow_levels,
            glow_steps: config.glow_steps,
            glow_hidden_features: config.glow_hidden_features,
            glow_coupling: match config.glow_coupling {
                CouplingType::Affine => "affine".to_string(),
                CouplingType::Additive => "additive".to_string(),
            },
            total_latent_dim: total_d,
            npca_rotation: if config.npca_householder {
                "householder".to_string()
            } else {
                "svd".to_string()
            },
            npca_householder_reflectors: config
                .npca_householder
                .then_some(config.npca_householder_reflectors),
        }),
    };
    snapshot
        .save(&config.output_dir)
        .context("write train_config.json snapshot")?;

    let mut history = Vec::with_capacity(config.epochs);
    let bpd_divisor = (manifest.seq_length * manifest.dim_obs) as f32 * std::f32::consts::LN_2;

    crate::training_log::log_true_transition_matrix_from_data(
        &config.data_dir,
        manifest.num_states,
    );

    let batcher = SequenceBatcher {
        seq_length: manifest.seq_length,
        obs_dim: manifest.dim_obs,
    };

    let total_sequences = dataset.len();
    let n_batches = (total_sequences + config.batch_size - 1) / config.batch_size;

    // EMA of |npca| and |msm| for smooth adaptive cap (currently disabled).
    let mut _ema_npca = 0.0_f32;
    let mut _ema_msm = 1.0_f32;
    const _EMA_ALPHA: f32 = 0.01;

    let mut q_perturbations_remaining = 3_usize;

    const GRAD_ACCUM_STEPS: usize = 16;

    for epoch_idx in 0..config.epochs {
        let dataloader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .shuffle(config.seed + epoch_idx as u64)
            .set_device(device.clone())
            .build(dataset.clone());

        let mut epoch_loss_sum = 0.0_f32;
        let mut epoch_joint_sum = 0.0_f32;
        let mut epoch_logdet_sum = 0.0_f32;
        let mut epoch_tail_sum = 0.0_f32;
        let mut epoch_msm_sum = 0.0_f32;
        let mut epoch_npca_sum = 0.0_f32;
        let mut epoch_scaled_msm_sum = 0.0_f32;
        let mut epoch_effective_w_sum = 0.0_f32;
        let mut step_count = 0_usize;

        let mut grad_accumulator: GradientsAccumulator<FlowSnlds<B>> =
            GradientsAccumulator::new();
        let mut accum_count = 0_usize;

        for (batch_i, batch) in dataloader.iter().enumerate() {
            let batch_obs = batch.obs;
            let [batch_size, _, _] = batch_obs.dims();

            let output = model.forward(
                batch_obs,
                config.w_msm,
                config.w_npca,
                config.temperature,
                true,
            );
            let msm_v = scalar_value(&output.msm_loglik);
            let npca_v = scalar_value(&output.npca_loglik);

            // // Adaptive w_msm: linear warmup from 50% to 150% of EMA-smoothed cap.
            // ema_npca = EMA_ALPHA * npca_v.abs() + (1.0 - EMA_ALPHA) * ema_npca;
            // ema_msm = EMA_ALPHA * msm_v.abs() + (1.0 - EMA_ALPHA) * ema_msm;
            // let cap = ema_npca / ema_msm.max(1e-8);
            // let warmup_epochs = config.w_msm.max(1e-8);
            // let progress = (epoch_idx as f32 + batch_i as f32 / n_batches as f32) / warmup_epochs;
            // let ramp = 0.10 + 1.00 * progress.min(1.0);
            // let effective_w_msm = cap * ramp;
            let effective_w_msm = config.w_msm;
            let scaled_msm = effective_w_msm * msm_v;

            let loss = output.loss.clone() / GRAD_ACCUM_STEPS as f32;

            let loss_v = scalar_value(&output.loss);
            let joint = -(loss_v as f64) as f32;

            let log_det_d = output.npca_output.log_det.clone().detach();
            let z_r = FlowSnlds::<B>::compute_z_r(
                &output.npca_output.z_pca.clone().detach(),
                &output.npca_output.z_prefix.clone().detach(),
                model.latent_dim_switching(),
            );
            let log_p_tail = log_p_z_isotropic(z_r);
            let jacobian_sum_over_n = tensor1d_sum_f32(&log_det_d) / batch_size as f32;
            let log_p_tail_sum_over_n = tensor1d_sum_f32(&log_p_tail) / batch_size as f32;

            if crate::training_log::should_log_minibatch(config.log_every_batch, batch_i, n_batches)
            {
                let bpd_log = -npca_v / bpd_divisor;
                println!(
                    "flow epoch {:04} batch {:04}/{} joint={:.4} msm_loglik={:.4} scaled_msm={:.4} w_msm_eff={:.2} bpd={:.4} jacobian={:.4}",
                    epoch_idx,
                    batch_i + 1,
                    n_batches,
                    joint,
                    msm_v,
                    scaled_msm,
                    effective_w_msm,
                    bpd_log,
                    jacobian_sum_over_n,
                );
            }

            let gradients = loss.backward();
            let grad_params = GradientsParams::from_grads(gradients, &model);
            grad_accumulator.accumulate(&model, grad_params);
            accum_count += 1;

            if accum_count >= GRAD_ACCUM_STEPS || batch_i + 1 == n_batches {
                let accumulated = grad_accumulator.grads();
                model = optimizer.step(config.learning_rate, model, accumulated);
                accum_count = 0;
            }

            // Q-logit perturbation every half-epoch to break symmetry (first 3 times).
            if q_perturbations_remaining > 0 && batch_i > 0 && batch_i % (n_batches / 2) == 0 {
                let q = model.q_logits.val();
                let noise = Tensor::random(q.dims(), Distribution::Normal(0.0, 0.1), device);
                model.q_logits = Param::from_tensor((q + noise).detach());
                q_perturbations_remaining -= 1;
                println!(
                    "flow epoch {:04} batch {:04}/{}: perturbed q_logits (σ=0.5, {} remaining)",
                    epoch_idx,
                    batch_i + 1,
                    n_batches,
                    q_perturbations_remaining,
                );
            }

            let t_every = config.transition_log_every_batches;
            if crate::training_log::should_log_transition_every_n_batches(t_every, batch_i) {
                crate::training_log::log_learned_transition_matrix(
                    "flow ",
                    epoch_idx,
                    model.q_logits.val(),
                    config.temperature,
                    Some((batch_i + 1, n_batches)),
                );
            }

            epoch_loss_sum += loss_v;
            epoch_joint_sum += joint;
            epoch_logdet_sum += jacobian_sum_over_n;
            epoch_tail_sum += log_p_tail_sum_over_n;
            epoch_msm_sum += msm_v;
            epoch_npca_sum += npca_v;
            epoch_scaled_msm_sum += scaled_msm;
            epoch_effective_w_sum += effective_w_msm;
            step_count += 1;
        }

        let sc = step_count.max(1) as f32;
        let stats = FlowEpochStats {
            epoch: epoch_idx,
            mean_loss: epoch_loss_sum / sc,
            mean_joint: epoch_joint_sum / sc,
            mean_logdet_sum_over_n: epoch_logdet_sum / sc,
            mean_log_p_tail_sum_over_n: epoch_tail_sum / sc,
            mean_msm_loglik: epoch_msm_sum / sc,
            mean_npca_loglik: epoch_npca_sum / sc,
        };
        let mean_bpd = -stats.mean_npca_loglik / bpd_divisor;
        let mean_scaled_msm = epoch_scaled_msm_sum / sc;
        let mean_effective_w = epoch_effective_w_sum / sc;
        println!(
            "flow epoch {:04} end: mean_joint={:.4} mean_msm_loglik={:.4} mean_scaled_msm={:.4} mean_w_msm_eff={:.2} mean_bpd={:.4} mean_jacobian={:.4}",
            stats.epoch,
            stats.mean_joint,
            stats.mean_msm_loglik,
            mean_scaled_msm,
            mean_effective_w,
            mean_bpd,
            stats.mean_logdet_sum_over_n,
        );
        crate::training_log::log_learned_transition_matrix(
            "flow ",
            epoch_idx,
            model.q_logits.val(),
            config.temperature,
            None,
        );

        // Validation pass on the inner (non-autodiff) backend — no graph allocated.
        if let Some(ref val_ds) = val_dataset {
            let val_model = model.valid();
            let val_batcher = SequenceBatcher {
                seq_length: manifest.seq_length,
                obs_dim: manifest.dim_obs,
            };
            let val_loader = DataLoaderBuilder::new(val_batcher)
                .batch_size(config.batch_size)
                .set_device(device.clone())
                .build(val_ds.clone());

            let mut val_npca_sum = 0.0_f32;
            let mut val_msm_sum = 0.0_f32;
            let mut val_logdet_sum = 0.0_f32;
            let mut val_mse_sum = 0.0_f64;
            let mut val_count = 0_usize;
            let mut val_frames = 0_usize;

            for batch in val_loader.iter() {
                let [bs, seq_t, _obs_d] = batch.obs.dims();
                let out = val_model.forward(batch.obs.clone(), 1.0, 1.0, config.temperature, false);
                val_npca_sum += out.npca_loglik.into_data().to_vec::<f32>().unwrap()[0];
                val_msm_sum += out.msm_loglik.into_data().to_vec::<f32>().unwrap()[0];
                let logdet_sum: f32 = out
                    .npca_output
                    .log_det
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .sum();
                val_logdet_sum += logdet_sum / bs as f32;

                let x_hat = val_model.decode_observations(
                    out.npca_output.z_pca,
                    out.npca_output.z_prefix,
                    &out.npca_output.latent_shapes,
                    out.npca_output.batch_stats,
                    (bs, seq_t),
                );
                let diff = batch.obs - x_hat;
                let mse: f32 = (diff.clone() * diff)
                    .mean()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
                val_mse_sum += mse as f64 * (bs * seq_t) as f64;
                val_frames += bs * seq_t;
                val_count += 1;
            }
            let vc = val_count.max(1) as f32;
            let val_bpd = -(val_npca_sum / vc) / bpd_divisor;
            let val_mse = val_mse_sum / val_frames.max(1) as f64;
            println!(
                "flow epoch {:04} val: bpd={:.4} msm_loglik={:.4} jacobian={:.4} recon_mse={:.6}",
                epoch_idx,
                val_bpd,
                val_msm_sum / vc,
                val_logdet_sum / vc,
                val_mse,
            );
        }

        if config.checkpoint_every > 0 && (epoch_idx + 1) % config.checkpoint_every == 0 {
            save_flow_checkpoint(&model, &config.output_dir, epoch_idx)?;
        }
        history.push(stats);
    }

    save_flow_checkpoint(&model, &config.output_dir, config.epochs.saturating_sub(1))?;
    Ok(history)
}

fn scalar_value<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map(|values| values.first().copied().unwrap_or(f32::NAN))
        .unwrap_or(f32::NAN)
}

fn tensor1d_sum_f32<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map(|values| values.iter().sum())
        .unwrap_or(0.0)
}

fn save_flow_checkpoint<B: AutodiffBackend + Backend<FloatElem = f32>>(
    model: &FlowSnlds<B>,
    output_dir: &Path,
    epoch_idx: usize,
) -> anyhow::Result<()> {
    let path = output_dir.join(format!("flow_checkpoint_{epoch_idx:04}.mpk"));
    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(path.clone(), &recorder)
        .with_context(|| format!("save FlowSNLDS checkpoint {:?}", path))?;
    Ok(())
}
