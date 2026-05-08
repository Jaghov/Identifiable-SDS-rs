use crate::cnn::{validate_cnn_res, CnnDecoder, CnnDecoderConfig, CnnEncoder, CnnEncoderConfig};
use crate::mlp::{Mlp, MlpConfig};
use crate::switching::compute_local_evidence;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{activation::log_softmax, backend::Backend, Tensor},
};
use snlds_core::hmm::{log_backward, log_forward};
use std::f32::consts::PI;

/// Encoder/decoder family selector for [`SnldsConfig`].
///
/// `Mlp` is the Python-`factored` parity path: a 2-hidden-layer leaky-ReLU MLP
/// pair operating on flat observations. `Cnn { res }` swaps in
/// [`CnnEncoder`]/[`CnnDecoder`] for `res × res × 3` image observations
/// (Python `CNNFastEncoder` / `CNNFastDecoder`); `obs_dim` must equal
/// `3 * res * res` and `res` must be a power of two `≥ 16`.
///
/// The enum is shaped so a future `Flow` variant for the planned `FlowSNLDS`
/// encoder can be added with no churn to `forward()` — every variant ultimately
/// produces the same `[N*T, 2*latent_dim]` / `[N*T, obs_dim]` interface via
/// [`SnldsEncoder`] / [`SnldsDecoder`].
#[non_exhaustive]
#[derive(Config, Debug, PartialEq)]
pub enum EncoderKind {
    /// Two-hidden-layer leaky-ReLU MLP pair (Python `factored` parity).
    Mlp,
    /// CNN encoder/decoder pair for flat RGB image observations.
    Cnn {
        /// Input frame side length; `obs_dim` MUST equal `3 * res * res`.
        res: usize,
    },
}

// `#[derive(Default)]` would be cleaner but doesn't compose with Burn's
// `#[derive(Config)]` macro on enums (the `#[default]` variant attribute is
// rejected as unknown). Hand-rolling keeps `EncoderKind::Mlp` as the default.
#[allow(clippy::derivable_impls)]
impl Default for EncoderKind {
    fn default() -> Self {
        EncoderKind::Mlp
    }
}

/// Layout configuration for [`VariationalSnlds`].
///
/// All **structural** hyper-parameters (tensor shapes, MLP widths, `K`) live here; the
/// `Module` only stores differentiable sub-modules and parameter tensors.
///
/// # Note
///
/// **Runtime scalars are deliberately *not* on this struct.** `beta` (KL weight),
/// `obs_noise_var` (fixed σ² for the Gaussian decoder likelihood), and `temperature`
/// (Gumbel-softmax annealing) are passed to [`VariationalSnlds::forward`] at every
/// step so callers can anneal them. The `snlds-train` crate owns these values via
/// its `TrainConfig` and persists them in `train_config.json` next to checkpoints
/// (see the `snlds-train::snapshot` module) so that `snlds-eval` can reproduce the
/// same numbers without a second source of truth.
#[derive(Config, Debug)]
pub struct SnldsConfig {
    /// Observation dimension.
    pub obs_dim: usize,
    /// Continuous latent dimension.
    pub latent_dim: usize,
    /// Hidden dimension for all MLPs.
    pub hidden_dim: usize,
    /// Number of discrete states K.
    pub num_states: usize,
    /// Encoder/decoder family. Default `Mlp` keeps every existing call site
    /// behaviour-identical; switch to `Cnn { res }` for image observations.
    #[config(default = "EncoderKind::Mlp")]
    pub kind: EncoderKind,
}

impl SnldsConfig {
    /// Initialise a new [`VariationalSnlds`] with random weights on `device`.
    ///
    /// # Panics
    ///
    /// Panics if `kind == EncoderKind::Cnn { res }` and either `res` is not a
    /// power of two `≥ 16` or `obs_dim != 3 * res * res`. Both are call-site
    /// configuration errors; eval/train CLIs surface them as user-facing errors
    /// before reaching this point.
    pub fn init<B: Backend>(&self, device: &B::Device) -> VariationalSnlds<B> {
        let transition_nets = (0..self.num_states)
            .map(|_| {
                MlpConfig::softplus(self.latent_dim, self.latent_dim, self.hidden_dim).init(device)
            })
            .collect();

        let (encoder, decoder) = match self.kind {
            EncoderKind::Mlp => {
                let encoder = SnldsEncoder::Mlp(
                    MlpConfig::leaky_relu(self.obs_dim, 2 * self.latent_dim, self.hidden_dim)
                        .init(device),
                );
                let decoder = SnldsDecoder::Mlp(
                    MlpConfig::leaky_relu(self.latent_dim, self.obs_dim, self.hidden_dim)
                        .init(device),
                );
                (encoder, decoder)
            }
            EncoderKind::Cnn { res } => {
                validate_cnn_res(res).expect("EncoderKind::Cnn { res } must be power-of-2 >= 16");
                assert_eq!(
                    self.obs_dim,
                    3 * res * res,
                    "EncoderKind::Cnn {{ res: {res} }} requires obs_dim == 3*res*res"
                );
                let encoder = SnldsEncoder::Cnn(
                    CnnEncoderConfig {
                        res,
                        output_dim: 2 * self.latent_dim,
                        hidden_dim: self.hidden_dim,
                    }
                    .init(device),
                );
                let decoder = SnldsDecoder::Cnn(
                    CnnDecoderConfig {
                        res,
                        input_dim: self.latent_dim,
                        hidden_dim: self.hidden_dim,
                    }
                    .init(device),
                );
                (encoder, decoder)
            }
        };

        let q_logits =
            Param::from_tensor(Tensor::zeros([self.num_states, self.num_states], device));
        let pi_logits = Param::from_tensor(Tensor::zeros([self.num_states], device));

        // init_mean ~ N(0,1), shape [K, latent_dim]
        let init_mean = Param::from_tensor(Tensor::random(
            [self.num_states, self.latent_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        ));
        // init_cov_factor: random positive scalar per state, scaled ~5 matching Python init
        let init_cov_factor = Param::from_tensor(Tensor::random(
            [self.num_states, self.latent_dim],
            burn::tensor::Distribution::Uniform(0.1, 1.0),
            device,
        ));
        // emission_cov_factor: identity (all ones) matching Python eye init
        let emission_cov_factor =
            Param::from_tensor(Tensor::ones([self.num_states, self.latent_dim], device));

        VariationalSnlds {
            transition_nets,
            encoder,
            decoder,
            q_logits,
            pi_logits,
            init_mean,
            init_cov_factor,
            emission_cov_factor,
        }
    }
}

/// Encoder wrapper that dispatches between the MLP and CNN variants. Both
/// variants take `[N*T, obs_dim]` and emit `[N*T, 2*latent_dim]`.
///
/// Variants differ in size (the CNN holds `Vec<Conv2d>` plus a projection MLP);
/// each `VariationalSnlds` carries exactly one encoder, so the unused-bytes
/// cost flagged by the lint is irrelevant in practice.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum SnldsEncoder<B: Backend> {
    /// Two-hidden-layer leaky-ReLU MLP (Python `factored` parity).
    Mlp(Mlp<B>),
    /// CNN encoder for flat RGB image observations (Python `CNNFastEncoder`).
    Cnn(CnnEncoder<B>),
}

impl<B: Backend> SnldsEncoder<B> {
    /// Forward: `[N*T, obs_dim] → [N*T, 2*latent_dim]`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            SnldsEncoder::Mlp(mlp) => mlp.forward(input),
            SnldsEncoder::Cnn(cnn) => cnn.forward(input),
        }
    }
}

/// Decoder wrapper that dispatches between the MLP and CNN variants. Both
/// variants take `[N*T, latent_dim]` and emit `[N*T, obs_dim]`.
///
/// See [`SnldsEncoder`] for why `clippy::large_enum_variant` is silenced.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum SnldsDecoder<B: Backend> {
    /// Two-hidden-layer leaky-ReLU MLP (Python `factored` parity).
    Mlp(Mlp<B>),
    /// CNN decoder for flat RGB image observations (Python `CNNFastDecoder`).
    Cnn(CnnDecoder<B>),
}

impl<B: Backend> SnldsDecoder<B> {
    /// Forward: `[N*T, latent_dim] → [N*T, obs_dim]`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            SnldsDecoder::Mlp(mlp) => mlp.forward(input),
            SnldsDecoder::Cnn(cnn) => cnn.forward(input),
        }
    }
}

/// Variational SNLDS model.
///
/// Matches `VariationalSNLDS` from Python with `encoder_type='factored'` and
/// `inference='alpha'` for [`SnldsEncoder::Mlp`], or `encoder_type='video'`
/// with the CNN bypass (no temporal LSTM) for [`SnldsEncoder::Cnn`].
/// Temperature annealing is not included (M4+).
#[derive(Module, Debug)]
pub struct VariationalSnlds<B: Backend> {
    /// K transition MLPs: p(z_t | z_{t-1}, s_t), one per discrete state.
    pub transition_nets: Vec<Mlp<B>>,
    /// Encoder q(z | x): obs_dim → 2 * latent_dim (mean ‖ log-variance).
    pub encoder: SnldsEncoder<B>,
    /// Decoder p(x | z): latent_dim → obs_dim.
    pub decoder: SnldsDecoder<B>,
    /// Transition logits Q, shape [K, K]; log p(s_t | s_{t-1}) = log_softmax(Q / temp, dim=-1).
    pub q_logits: Param<Tensor<B, 2>>,
    /// Initial state logits π, shape [K]; log p(s_1) = log_softmax(π / temp).
    pub pi_logits: Param<Tensor<B, 1>>,
    /// Per-state initial mean, shape [K, latent_dim].
    pub init_mean: Param<Tensor<B, 2>>,
    /// Diagonal factor of per-state initial covariance, shape [K, latent_dim].
    /// Actual variance = init_cov_factor² + COV_EPS.
    pub init_cov_factor: Param<Tensor<B, 2>>,
    /// Diagonal factor of per-state emission covariance, shape [K, latent_dim].
    /// Actual variance = emission_cov_factor² + COV_EPS.
    pub emission_cov_factor: Param<Tensor<B, 2>>,
}

/// Outputs of [`VariationalSnlds::forward`].
#[derive(Debug)]
pub struct ForwardOutput<B: Backend> {
    /// Reconstructed observations, shape [N, T, obs_dim].
    pub obs_reconstructed: Tensor<B, 3>,
    /// Sampled latent trajectory, shape [N, T, latent_dim].
    pub latent_samples: Tensor<B, 3>,
    /// Soft state posteriors γ_{t,k} = p(s_t | x_{1:T}), shape [N, T, K].  None when β=0.
    pub state_posteriors: Option<Tensor<B, 3>>,
    /// Scalar ELBO (to maximise / negate for loss).
    pub elbo: Tensor<B, 1>,
    /// Reconstruction term log p(x | z), positive contribution to ELBO.
    pub recon_loss: Tensor<B, 1>,
    /// Entropy of q(z) — positive contribution to ELBO.
    pub entropy_q: Tensor<B, 1>,
    /// MSM term: sum log Z_t / N (inference='alpha').
    pub msm_loss: Tensor<B, 1>,
}

impl<B: Backend> VariationalSnlds<B> {
    // ── helpers ──────────────────────────────────────────────────────────────

    fn num_states(&self) -> usize {
        self.q_logits.val().dims()[0]
    }

    fn latent_dim(&self) -> usize {
        self.init_mean.val().dims()[1]
    }

    /// Reparameterised sample from q(z | x).
    ///
    /// Returns `(z_sample, z_mean, z_log_var)`, each shape [N*T, latent_dim].
    fn encode_obs(&self, obs: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch_size, seq_len, obs_dim] = obs.dims();
        let latent_dim = self.latent_dim();

        // Flatten time into batch: [N, T, D_obs] → [N*T, D_obs]
        let flat_obs = obs.reshape([batch_size * seq_len, obs_dim]);

        // Encoder output: [N*T, 2*latent_dim]
        let encoder_output = self.encoder.forward(flat_obs);

        let z_mean = encoder_output
            .clone()
            .slice([0..batch_size * seq_len, 0..latent_dim]);
        let z_log_var = encoder_output.slice([0..batch_size * seq_len, latent_dim..2 * latent_dim]);

        // Reparameterisation: z = μ + σ·ε, ε ~ N(0,I)
        let noise = Tensor::random(
            [batch_size * seq_len, latent_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &z_mean.device(),
        );
        let z_std = (z_log_var.clone() * 0.5).exp();
        let z_sample = z_mean.clone() + z_std * noise;

        // Restore time dimension: [N*T, latent_dim] → [N, T, latent_dim]
        (
            z_sample.reshape([batch_size, seq_len, latent_dim]),
            z_mean.reshape([batch_size, seq_len, latent_dim]),
            z_log_var.reshape([batch_size, seq_len, latent_dim]),
        )
    }

    /// Log p(z_t | z_{t-1}, s_t) and log p(z_1 | s_1) under diagonal Gaussian.
    ///
    /// Returns local evidence `[N, T, K]`.
    fn compute_local_evidence(&self, latent_samples: Tensor<B, 3>) -> Tensor<B, 3> {
        compute_local_evidence(
            latent_samples,
            &self.transition_nets,
            self.init_mean.val(),
            self.init_cov_factor.val(),
            self.emission_cov_factor.val(),
        )
    }

    /// ELBO under `inference='alpha'`: sum_t log Z_t / N.
    ///
    /// Returns (elbo, recon_loss, entropy_q, msm_loss).
    fn compute_elbo(
        &self,
        obs: Tensor<B, 3>,
        obs_reconstructed: Tensor<B, 3>,
        z_log_var: Tensor<B, 3>,
        log_z: Tensor<B, 2>,
        beta: f32,
        obs_noise_var: f32,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let [batch_size, _seq_len, _obs_dim] = obs.dims();
        let device = obs.device();

        // Reconstruction: sum_t log N(x_t; x̂_t, σ²I)
        let log_2pi = (2.0_f32 * PI).ln();
        let recon_log_prob = -(((obs - obs_reconstructed).powf_scalar(2.0_f32) / obs_noise_var
            + obs_noise_var.ln()
            + log_2pi)
            * 0.5_f32)
            .sum_dim(2) // sum over obs_dim
            .sum_dim(1) // sum over time
            .reshape([batch_size]); // [N]

        let recon_loss = recon_log_prob.sum() / batch_size as f32;

        // Entropy of q(z): -log q(z) = 0.5 * sum(1 + log_var + log(2π))
        let entropy_q = (z_log_var + log_2pi + 1.0_f32)
            .sum_dim(2)
            .sum_dim(1)
            .reshape([batch_size]);
        let entropy_q = (entropy_q.sum() * 0.5_f32) / batch_size as f32;

        // MSM term: sum_t log Z_t / N  (inference='alpha')
        let msm_loss = if beta > 0.0 {
            log_z.sum_dim(1).reshape([batch_size]).sum() / batch_size as f32
        } else {
            Tensor::zeros([1], &device)
        };

        let elbo = recon_loss.clone() + entropy_q.clone() + msm_loss.clone() * beta;

        (elbo, recon_loss, entropy_q, msm_loss)
    }

    /// Full forward pass.
    ///
    /// # Arguments
    ///
    /// - `obs`: observation tensor `[N, T, obs_dim]`.
    /// - `beta`: KL weight on the discrete-state ELBO term. Must be `> 0` for the
    ///   forward pass to populate `state_posteriors`; setting `beta = 0` disables the
    ///   MSM term entirely (useful for debugging the encoder/decoder in isolation).
    /// - `obs_noise_var`: fixed observation noise variance σ² scaling the squared-error
    ///   term in the Gaussian decoder log-likelihood. The Python reference uses `5e-4`;
    ///   `snlds-train` accepts it on the CLI and persists it to `train_config.json` so
    ///   `snlds-eval` reproduces the same number without a second source of truth.
    /// - `temperature`: scales `Q` and `π` logits (default 1.0; anneal toward 0
    ///   during training).
    ///
    /// See the `# Note` section on [`SnldsConfig`] for why these scalars are not on
    /// the config struct.
    pub fn forward(
        &self,
        obs: Tensor<B, 3>,
        beta: f32,
        obs_noise_var: f32,
        temperature: f32,
    ) -> ForwardOutput<B> {
        let [batch_size, seq_len, obs_dim] = obs.dims();
        let num_states = self.num_states();
        let device = obs.device();

        // 1. Encode
        let (latent_samples, _encoder_mean, z_log_var) = self.encode_obs(obs.clone());

        // 2. Local evidence [N, T, K]
        let log_local_evidence = if beta > 0.0 {
            self.compute_local_evidence(latent_samples.clone())
        } else {
            Tensor::zeros([batch_size, seq_len, num_states], &device)
        };

        // 3. HMM forward pass (M2 kernels)
        let log_pi = log_softmax(self.pi_logits.val() / temperature, 0);
        let log_trans = log_softmax(self.q_logits.val() / temperature, 1);
        let (log_alpha, log_z) = log_forward(log_local_evidence.clone(), log_pi, log_trans.clone());

        // 4. Decode
        let latent_dim = self.latent_dim();
        let obs_hat = self
            .decoder
            .forward(
                latent_samples
                    .clone()
                    .reshape([batch_size * seq_len, latent_dim]),
            )
            .reshape([batch_size, seq_len, obs_dim]);

        // 5. ELBO
        let (elbo, recon_loss, entropy_q, msm_loss) = self.compute_elbo(
            obs,
            obs_hat.clone(),
            z_log_var,
            log_z.clone(),
            beta,
            obs_noise_var,
        );

        // 6. Posteriors γ (detached — not used in training gradient directly)
        let state_posteriors = if beta > 0.0 {
            let log_beta = log_backward(log_local_evidence, log_trans, log_z);
            Some((log_alpha + log_beta).exp())
        } else {
            None
        };

        ForwardOutput {
            obs_reconstructed: obs_hat,
            latent_samples,
            state_posteriors,
            elbo,
            recon_loss,
            entropy_q,
            msm_loss,
        }
    }
}
