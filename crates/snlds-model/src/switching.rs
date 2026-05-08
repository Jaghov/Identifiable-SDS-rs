//! Shared switching / local-evidence helpers for SNLDS (used by [`VariationalSnlds`] and [`FlowSnlds`]).
//!
//! [`VariationalSnlds`]: crate::VariationalSnlds
//! [`FlowSnlds`]: crate::FlowSnlds

use crate::mlp::Mlp;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use std::f32::consts::PI;

pub(crate) const COV_EPS: f32 = 1e-6;

/// Log p(z; μ_k, diag(σ²_k)) for all K states simultaneously.
///
/// - `z_batch`: `[N, latent_dim]`
/// - `means`: `[K, latent_dim]`
/// - `variances`: `[K, latent_dim]`  (must be positive)
///
/// Returns `[N, K]`.
pub fn diagonal_mvn_log_prob_all_states<B: Backend>(
    z_batch: Tensor<B, 2>,
    means: Tensor<B, 2>,
    variances: Tensor<B, 2>,
    batch_size: usize,
    num_states: usize,
    latent_dim: usize,
) -> Tensor<B, 2> {
    let log_2pi = (2.0_f32 * PI).ln();

    let z_expanded = z_batch
        .unsqueeze_dim::<3>(1)
        .expand([batch_size, num_states, latent_dim]);
    let means_expanded = means
        .unsqueeze::<3>()
        .expand([batch_size, num_states, latent_dim]);
    let var_expanded = variances
        .unsqueeze::<3>()
        .expand([batch_size, num_states, latent_dim]);

    let diff = z_expanded - means_expanded;
    (var_expanded.clone().log() + diff.powf_scalar(2.0_f32) / var_expanded + log_2pi)
        .sum_dim(2)
        .reshape([batch_size, num_states])
        * -0.5_f32
}

/// Log p(z_t | z_{t-1}, s_t) and log p(z_1 | s_1) under diagonal Gaussian transitions.
///
/// Returns local evidence `[N, T, K]`.
pub fn compute_local_evidence<B: Backend>(
    latent_samples: Tensor<B, 3>,
    transition_nets: &[Mlp<B>],
    init_mean: Tensor<B, 2>,
    init_cov_factor: Tensor<B, 2>,
    emission_cov_factor: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, latent_dim] = latent_samples.dims();
    let num_states = transition_nets.len();

    let z_first = latent_samples
        .clone()
        .slice([0..batch_size, 0..1, 0..latent_dim])
        .reshape([batch_size, latent_dim]);

    let init_var = init_cov_factor.powf_scalar(2.0_f32) + COV_EPS;

    let init_log_prob = diagonal_mvn_log_prob_all_states(
        z_first, init_mean, init_var, batch_size, num_states, latent_dim,
    );
    let init_log_prob = init_log_prob.unsqueeze_dim::<3>(1);

    if seq_len == 1 {
        return init_log_prob;
    }

    let z_prev = latent_samples
        .clone()
        .slice([0..batch_size, 0..seq_len - 1, 0..latent_dim]);

    let emission_var = emission_cov_factor.powf_scalar(2.0_f32) + COV_EPS;

    let transition_means = {
        let flat_prev = z_prev.reshape([batch_size * (seq_len - 1), latent_dim]);
        let per_state: Vec<Tensor<B, 2>> = transition_nets
            .iter()
            .map(|net| net.forward(flat_prev.clone()))
            .collect();
        Tensor::stack(per_state, 1)
    };

    let z_next = latent_samples
        .slice([0..batch_size, 1..seq_len, 0..latent_dim])
        .reshape([batch_size * (seq_len - 1), latent_dim])
        .unsqueeze_dim::<3>(1)
        .expand([batch_size * (seq_len - 1), num_states, latent_dim]);

    let emission_var_exp =
        emission_var
            .unsqueeze::<3>()
            .expand([batch_size * (seq_len - 1), num_states, latent_dim]);

    let diff = z_next - transition_means;
    let log_2pi = (2.0_f32 * PI).ln();
    let transition_log_prob =
        (emission_var_exp.clone().log() + diff.powf_scalar(2.0_f32) / emission_var_exp + log_2pi)
            .sum_dim(2)
            .reshape([batch_size * (seq_len - 1), num_states])
            * -0.5_f32;

    let transition_log_prob = transition_log_prob.reshape([batch_size, seq_len - 1, num_states]);

    Tensor::cat(vec![init_log_prob, transition_log_prob], 1)
}
