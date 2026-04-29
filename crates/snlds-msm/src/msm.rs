//! Burn analogue of `models.NeuralMSM` from the Python project.
//!
//! ## Simplifications vs Python
//! - **Diagonal** initial covariance and emission covariance (Python uses full
//!   covariance Cholesky factors). Aligns with the diagonal `init_cov_factor`
//!   and `emission_cov_factor` already used by [`snlds_model::VariationalSnlds`].
//! - Transition activation is **softplus** (matches the SNLDS transition nets
//!   we want to warm-start), not `cos` as in the Python NeuralMSM.
//! - **Lag = 1** only (no causal multi-lag MLP variant).
//! - Optimisation uses straight Adam minibatch maximisation of the marginal
//!   log-likelihood `sum_t log Z_t / batch` rather than the EM-style closed-form
//!   update on `Q` from Python (gradient-based update is enough for warm-start).

use burn::{
    config::Config,
    module::{Module, Param},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        activation::log_softmax,
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};
use snlds_core::hmm::log_forward;
use snlds_model::{Mlp, MlpConfig};
use std::f32::consts::PI;

const COV_EPS: f32 = 1e-6;

/// Configuration for [`NeuralMsm`].
#[derive(Config, Debug)]
pub struct NeuralMsmConfig {
    /// Observation (= reduced) dimension. After PCA this is typically `dim_latent`.
    pub obs_dim: usize,
    /// Number of discrete states K.
    pub num_states: usize,
    /// Hidden dim for transition MLPs (Python default 16).
    #[config(default = "16")]
    pub hidden_dim: usize,
}

impl NeuralMsmConfig {
    /// Initialise a fresh [`NeuralMsm`] on `device` with random weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> NeuralMsm<B> {
        let transition_nets = (0..self.num_states)
            .map(|_| MlpConfig::softplus(self.obs_dim, self.obs_dim, self.hidden_dim).init(device))
            .collect();

        let q_logits =
            Param::from_tensor(Tensor::zeros([self.num_states, self.num_states], device));
        let pi_logits = Param::from_tensor(Tensor::zeros([self.num_states], device));

        let init_mean = Param::from_tensor(Tensor::random(
            [self.num_states, self.obs_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        ));
        let init_cov_factor = Param::from_tensor(Tensor::random(
            [self.num_states, self.obs_dim],
            burn::tensor::Distribution::Uniform(0.1, 1.0),
            device,
        ));
        let emission_cov_factor =
            Param::from_tensor(Tensor::ones([self.num_states, self.obs_dim], device));

        NeuralMsm {
            transition_nets,
            q_logits,
            pi_logits,
            init_mean,
            init_cov_factor,
            emission_cov_factor,
        }
    }
}

/// Simplified NeuralMSM: per-state transition MLP with diagonal Gaussian emissions.
#[derive(Module, Debug)]
pub struct NeuralMsm<B: Backend> {
    /// One MLP per discrete state — same shape as [`snlds_model::VariationalSnlds::transition_nets`].
    pub transition_nets: Vec<Mlp<B>>,
    /// Transition logits Q, `[K, K]`; `log p(s_t|s_{t-1}) = log_softmax(Q, dim=-1)`.
    pub q_logits: Param<Tensor<B, 2>>,
    /// Initial state logits π, `[K]`.
    pub pi_logits: Param<Tensor<B, 1>>,
    /// Per-state initial mean, `[K, obs_dim]`.
    pub init_mean: Param<Tensor<B, 2>>,
    /// Diagonal factor for initial covariance; variance = factor² + COV_EPS.
    pub init_cov_factor: Param<Tensor<B, 2>>,
    /// Diagonal factor for emission covariance; variance = factor² + COV_EPS.
    pub emission_cov_factor: Param<Tensor<B, 2>>,
}

impl<B: Backend> NeuralMsm<B> {
    fn num_states(&self) -> usize {
        self.q_logits.val().dims()[0]
    }

    /// Local evidence `[N, T, K]` = log p(x_t | x_{t-1}, s_t).
    fn compute_local_evidence(&self, obs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, obs_dim] = obs.dims();
        let num_states = self.num_states();

        let init_var = self.init_cov_factor.val().powf_scalar(2.0_f32) + COV_EPS;
        let emission_var = self.emission_cov_factor.val().powf_scalar(2.0_f32) + COV_EPS;

        let log_2pi = (2.0_f32 * PI).ln();

        let x_first = obs
            .clone()
            .slice([0..batch_size, 0..1, 0..obs_dim])
            .reshape([batch_size, obs_dim]);

        let init_log_prob = diagonal_mvn_log_prob_all_states(
            x_first,
            self.init_mean.val(),
            init_var,
            batch_size,
            num_states,
            obs_dim,
        )
        .unsqueeze_dim::<3>(1);

        if seq_len == 1 {
            return init_log_prob;
        }

        let x_prev = obs
            .clone()
            .slice([0..batch_size, 0..seq_len - 1, 0..obs_dim]);
        let flat_prev = x_prev.reshape([batch_size * (seq_len - 1), obs_dim]);
        let per_state: Vec<Tensor<B, 2>> = self
            .transition_nets
            .iter()
            .map(|net| net.forward(flat_prev.clone()))
            .collect();
        let transition_means = Tensor::stack::<3>(per_state, 1);

        let x_next = obs
            .slice([0..batch_size, 1..seq_len, 0..obs_dim])
            .reshape([batch_size * (seq_len - 1), obs_dim])
            .unsqueeze_dim::<3>(1)
            .expand([batch_size * (seq_len - 1), num_states, obs_dim]);

        let emission_var_exp =
            emission_var
                .unsqueeze::<3>()
                .expand([batch_size * (seq_len - 1), num_states, obs_dim]);

        let diff = x_next - transition_means;
        let transition_log_prob = (emission_var_exp.clone().log()
            + diff.powf_scalar(2.0_f32) / emission_var_exp
            + log_2pi)
            .sum_dim(2)
            .reshape([batch_size * (seq_len - 1), num_states])
            * -0.5_f32;

        let transition_log_prob =
            transition_log_prob.reshape([batch_size, seq_len - 1, num_states]);

        Tensor::cat(vec![init_log_prob, transition_log_prob], 1)
    }

    /// Mean per-sequence marginal log-likelihood `sum_t log Z_t / N`.
    pub fn marginal_log_likelihood(&self, obs: Tensor<B, 3>) -> Tensor<B, 1> {
        let [batch_size, _seq_len, _obs_dim] = obs.dims();
        let log_local_evidence = self.compute_local_evidence(obs);
        let log_pi = log_softmax(self.pi_logits.val(), 0);
        let log_trans = log_softmax(self.q_logits.val(), 1);
        let (_log_alpha, log_z) = log_forward(log_local_evidence, log_pi, log_trans);
        log_z.sum_dim(1).reshape([batch_size]).sum() / batch_size as f32
    }
}

impl<B: AutodiffBackend> NeuralMsm<B> {
    /// Fit the model with Adam on minibatches; returns the trained model and per-epoch
    /// mean log-likelihood.
    pub fn fit(
        mut self,
        obs: Tensor<B, 3>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) -> (Self, Vec<f32>) {
        let [num_sequences, seq_len, obs_dim] = obs.dims();
        let mut optimizer = AdamConfig::new().init::<B, NeuralMsm<B>>();
        let mut history = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            let mut epoch_ll = 0.0_f32;
            let mut step_count = 0_usize;

            let mut start = 0;
            while start < num_sequences {
                let end = (start + batch_size).min(num_sequences);
                let batch = obs.clone().slice([start..end, 0..seq_len, 0..obs_dim]);
                let log_likelihood = self.marginal_log_likelihood(batch);
                let loss = log_likelihood.clone().neg();

                let log_likelihood_value = log_likelihood
                    .into_data()
                    .to_vec::<f32>()
                    .ok()
                    .and_then(|values| values.first().copied())
                    .unwrap_or(f32::NAN);

                let gradients = loss.backward();
                let grad_params = GradientsParams::from_grads(gradients, &self);
                self = optimizer.step(learning_rate, self, grad_params);

                epoch_ll += log_likelihood_value;
                step_count += 1;
                start = end;
            }
            history.push(epoch_ll / step_count.max(1) as f32);
        }

        (self, history)
    }
}

fn diagonal_mvn_log_prob_all_states<B: Backend>(
    z_batch: Tensor<B, 2>,
    means: Tensor<B, 2>,
    variances: Tensor<B, 2>,
    batch_size: usize,
    num_states: usize,
    obs_dim: usize,
) -> Tensor<B, 2> {
    let log_2pi = (2.0_f32 * PI).ln();

    let z_expanded = z_batch
        .unsqueeze_dim::<3>(1)
        .expand([batch_size, num_states, obs_dim]);
    let means_expanded = means
        .unsqueeze::<3>()
        .expand([batch_size, num_states, obs_dim]);
    let var_expanded = variances
        .unsqueeze::<3>()
        .expand([batch_size, num_states, obs_dim]);

    let diff = z_expanded - means_expanded;
    (var_expanded.clone().log() + diff.powf_scalar(2.0_f32) / var_expanded + log_2pi)
        .sum_dim(2)
        .reshape([batch_size, num_states])
        * -0.5_f32
}
