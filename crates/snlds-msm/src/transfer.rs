//! Copy fitted [`NeuralMsm`] parameters into a [`VariationalSnlds`] instance.
//!
//! Matches the Python `train_snlds.py` warm-start: `model.transitions = best.transitions`,
//! `model.Q = best.Q.log()` etc. We map directly because we already train the MSM with
//! `q_logits` rather than a stochastic matrix, so no extra `log` step is needed.

use anyhow::{ensure, Context};
use burn::{
    module::Param,
    tensor::{backend::Backend, Tensor},
};
use snlds_model::VariationalSnlds;

use crate::msm::NeuralMsm;

/// Replace the warm-startable subset of [`VariationalSnlds`] parameters with the fitted
/// MSM values. The MSM's `obs_dim` must match the SNLDS `latent_dim` (typical pipeline:
/// PCA reduces obs to `dim_latent`, then MSM is trained at `obs_dim = dim_latent`).
pub fn transfer_into_snlds<B: Backend>(
    msm: NeuralMsm<B>,
    mut snlds: VariationalSnlds<B>,
) -> anyhow::Result<VariationalSnlds<B>> {
    let snlds_num_states = snlds.q_logits.val().dims()[0];
    let snlds_latent_dim = snlds.init_mean.val().dims()[1];
    let msm_num_states = msm.q_logits.val().dims()[0];
    let msm_obs_dim = msm.init_mean.val().dims()[1];

    ensure!(
        msm_num_states == snlds_num_states,
        "num_states mismatch: msm {msm_num_states} vs snlds {snlds_num_states}"
    );
    ensure!(
        msm_obs_dim == snlds_latent_dim,
        "MSM obs_dim ({msm_obs_dim}) must equal SNLDS latent_dim ({snlds_latent_dim}); \
         apply PCA to obs before fitting the MSM"
    );
    ensure!(
        msm.transition_nets.len() == snlds.transition_nets.len(),
        "transition net count mismatch: msm {} vs snlds {}",
        msm.transition_nets.len(),
        snlds.transition_nets.len()
    );

    snlds.transition_nets = msm.transition_nets;
    snlds.q_logits = move_param(msm.q_logits.val()).context("transfer q_logits")?;
    snlds.pi_logits = move_param(msm.pi_logits.val()).context("transfer pi_logits")?;
    snlds.init_mean = move_param(msm.init_mean.val()).context("transfer init_mean")?;
    snlds.init_cov_factor =
        move_param(msm.init_cov_factor.val()).context("transfer init_cov_factor")?;
    snlds.emission_cov_factor =
        move_param(msm.emission_cov_factor.val()).context("transfer emission_cov_factor")?;

    Ok(snlds)
}

fn move_param<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> anyhow::Result<Param<Tensor<B, D>>> {
    Ok(Param::from_tensor(tensor))
}
