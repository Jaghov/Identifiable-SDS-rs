//! M5 NeuralMSM warm-start orchestration for `snlds-train`.
//!
//! Mirrors the Python `train_snlds.py` pre-train loop:
//! 1. PCA-reduce observations to `dim_latent`.
//! 2. Run `--msm-restarts` random restarts of [`snlds_msm::NeuralMsm`]; keep the
//!    one with the highest mean log-likelihood.
//! 3. Transfer the best MSM's transition nets, `q_logits`, `pi_logits`,
//!    `init_mean`, `init_cov_factor`, and `emission_cov_factor` into a fresh
//!    [`VariationalSnlds`].

use anyhow::Context;
use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use ndarray::Array3;
use snlds_model::{SnldsConfig, VariationalSnlds};
use snlds_msm::{pca_fit_transform, transfer_into_snlds, NeuralMsm, NeuralMsmConfig};

/// Hyper-parameters for the warm-start phase. Off by default — populate only when
/// `snlds-train` is invoked with `--msm-init`.
#[derive(Clone, Debug)]
pub struct MsmWarmStartConfig {
    pub restarts: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub hidden_dim: usize,
}

impl Default for MsmWarmStartConfig {
    fn default() -> Self {
        Self {
            restarts: 3,
            epochs: 30,
            batch_size: 32,
            learning_rate: 7e-3,
            hidden_dim: 16,
        }
    }
}

/// Run PCA + MSM warm-start and return a freshly initialised SNLDS model with
/// the warm-started parameters copied in.
///
/// `obs_train_array` is the raw observation tensor `[N, T, dim_obs]` (typically
/// the output of [`crate::load_train_obs_array`]).
pub fn run_warm_start<B: AutodiffBackend>(
    config: &MsmWarmStartConfig,
    snlds_config: &SnldsConfig,
    obs_train_array: &Array3<f32>,
    device: &B::Device,
) -> anyhow::Result<VariationalSnlds<B>> {
    let reduced = pca_fit_transform(obs_train_array, snlds_config.latent_dim)
        .context("PCA reduction for MSM warm-start")?;

    let (num_sequences, seq_length, latent_dim) = reduced.dim();
    let (raw, _offset) = reduced.into_raw_vec_and_offset();
    let reduced_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(raw, [num_sequences, seq_length, latent_dim]),
        device,
    );

    let msm_config = NeuralMsmConfig::new(latent_dim, snlds_config.num_states)
        .with_hidden_dim(config.hidden_dim);

    let mut best: Option<(f32, NeuralMsm<B>)> = None;
    for restart_idx in 0..config.restarts.max(1) {
        let initial = msm_config.init::<B>(device);
        let (fitted, history) = initial.fit(
            reduced_tensor.clone(),
            config.epochs,
            config.batch_size,
            config.learning_rate,
        );
        let final_log_likelihood = history.last().copied().unwrap_or(f32::NEG_INFINITY);
        println!(
            "msm warm-start restart {restart_idx}: final mean log-likelihood = {final_log_likelihood:.4}"
        );
        if best
            .as_ref()
            .map(|(score, _)| final_log_likelihood > *score)
            .unwrap_or(true)
        {
            best = Some((final_log_likelihood, fitted));
        }
    }

    let (_, best_msm) = best.expect("at least one MSM restart should produce a model");
    let snlds = snlds_config.init::<B>(device);
    let warm_started = transfer_into_snlds(best_msm, snlds)
        .context("transfer MSM parameters into VariationalSnlds")?;
    Ok(warm_started)
}
