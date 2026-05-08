//! Flow SNLDS: Neural PCA encoder on images + HMM switching on the leading-`L` latent subspace.
//!
//! Training objective is **not** the variational ELBO: joint
//! `w_msm * MSM_loglik + w_npca * NPCA_loglik` (maximize → minimize negative).

use crate::mlp::{Mlp, MlpConfig};
use crate::npca::{
    default_glow_config_for_npca, flat_nhwc_rows_to_nchw, glow_flattened_latent_dim,
    glow_last_split_dim, log_p_z_isotropic, NeuralPca, NeuralPcaConfig, NeuralPcaOutput,
    PcaSvdBackend,
};
use crate::switching::compute_local_evidence;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{activation::log_softmax, backend::Backend, Tensor},
};
use glow_flow::prelude::{CouplingType, DequantizeConfig, TriangularInverse};
use snlds_core::hmm::{log_backward, log_forward};

/// Configuration for [`FlowSnlds`] (image observations via Neural PCA).
#[derive(Config, Debug)]
pub struct FlowSnldsConfig {
    /// Flat observation dimension; must be `3 * res * res`.
    pub obs_dim: usize,
    /// Continuous switching dimension `L` (leading PCA coords). Must satisfy `L <= last_split_dim`.
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub num_states: usize,
    /// RGB square frame side (height = width = `res`).
    pub res: usize,
    pub glow_levels: usize,
    pub glow_steps: usize,
    pub glow_hidden_features: usize,
    #[config(default = "CouplingType::Affine")]
    pub coupling_type: CouplingType,
    #[config(default = false)]
    pub householder_rotation: bool,
    #[config(default = "32")]
    pub householder_reflectors: usize,
    /// Pixel depth (bits) for the Dequantize layer that runs before NPCA.
    /// Bouncing-ball frames are smooth, so we coarsen to 5 bits by default.
    #[config(default = "5")]
    pub pixel_depth: u32,
}

impl FlowSnldsConfig {
    /// Total flattened NPCA latent dim `D` from Glow layout.
    pub fn total_latent_dim(&self) -> usize {
        glow_flattened_latent_dim(3, self.glow_levels, self.res, self.res)
    }

    /// Validate layout and build a [`FlowSnlds`] on `device`.
    ///
    /// # Panics
    ///
    /// If `obs_dim != 3*res*res`, `latent_dim > last_split_dim`, or `res` is invalid for Glow.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FlowSnlds<B> {
        assert_eq!(
            self.obs_dim,
            3 * self.res * self.res,
            "FlowSnlds requires obs_dim == 3 * res * res"
        );
        let d = self.total_latent_dim();
        let last_split_d = glow_last_split_dim(3, self.glow_levels, self.res, self.res);
        assert!(
            self.latent_dim <= last_split_d,
            "latent_dim L={} must be <= last_split_dim={}",
            self.latent_dim,
            last_split_d
        );

        let transition_nets = (0..self.num_states)
            .map(|_| {
                MlpConfig::softplus(self.latent_dim, self.latent_dim, self.hidden_dim).init(device)
            })
            .collect();

        let npca_config = NeuralPcaConfig::new(
            default_glow_config_for_npca(
                3,
                self.glow_levels,
                self.glow_steps,
                self.glow_hidden_features,
                self.coupling_type,
            ),
            d,
            last_split_d,
        )
        .with_householder_rotation(self.householder_rotation)
        .with_householder_reflectors(self.householder_reflectors);

        let npca = npca_config.init(device);

        let q_logits = Param::from_tensor(Tensor::random(
            [self.num_states, self.num_states],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        ));
        let pi_logits = Param::from_tensor(Tensor::zeros([self.num_states], device));

        let init_mean = Param::from_tensor(Tensor::random(
            [self.num_states, self.latent_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        ));
        let init_cov_factor = Param::from_tensor(Tensor::random(
            [self.num_states, self.latent_dim],
            burn::tensor::Distribution::Uniform(0.1, 1.0),
            device,
        ));
        let emission_cov_factor =
            Param::from_tensor(Tensor::ones([self.num_states, self.latent_dim], device));

        FlowSnlds {
            res: self.res,
            latent_dim: self.latent_dim,
            total_latent_dim: d,
            pixel_depth: self.pixel_depth,
            npca,
            transition_nets,
            q_logits,
            pi_logits,
            init_mean,
            init_cov_factor,
            emission_cov_factor,
        }
    }
}

/// Outputs of [`FlowSnlds::forward`].
#[derive(Debug)]
pub struct FlowForwardOutput<B: Backend> {
    /// Leading PCA latent trajectory `z_pca[:, :L]`, `[N, T, L]`.
    pub latent_samples: Tensor<B, 3>,
    /// Optional state posteriors when `w_msm > 0`, `[N, T, K]`.
    pub state_posteriors: Option<Tensor<B, 3>>,
    /// Scalar joint objective value: `w_msm * msm + w_npca * npca` (to **maximize**).
    pub joint_objective: Tensor<B, 1>,
    /// `sum_{n,t} log Z_{n,t} / N` (same scale as [`crate::VariationalSnlds`] `msm_loss`).
    pub msm_loglik: Tensor<B, 1>,
    /// `sum_{n,t} (log_det + log p(z_r))_frame / N` over the minibatch.
    pub npca_loglik: Tensor<B, 1>,
    /// Per frame `log|det ∂z/∂x| + log p(z_r)`, shape `[N, T]`.
    pub npca_logprob_frames: Tensor<B, 2>,
    /// Training loss scalar: `-joint_objective` (minimize).
    pub loss: Tensor<B, 1>,
    /// Cached Neural PCA forward (for diagnostics or callers that need `z_pca`).
    pub npca_output: NeuralPcaOutput<B>,
}

#[derive(Module, Debug)]
pub struct FlowSnlds<B: Backend> {
    /// Frame side length (not a learned parameter).
    #[module(ignore)]
    res: usize,
    /// Switching latent dim `L`.
    #[module(ignore)]
    latent_dim: usize,
    /// NPCA flattened dim `D`.
    #[module(ignore)]
    total_latent_dim: usize,
    /// Pixel depth used by the pre-NPCA Dequantize layer.
    #[module(ignore)]
    pixel_depth: u32,
    pub npca: NeuralPca<B>,
    pub transition_nets: Vec<Mlp<B>>,
    pub q_logits: Param<Tensor<B, 2>>,
    pub pi_logits: Param<Tensor<B, 1>>,
    pub init_mean: Param<Tensor<B, 2>>,
    pub init_cov_factor: Param<Tensor<B, 2>>,
    pub emission_cov_factor: Param<Tensor<B, 2>>,
}

impl<B: Backend> FlowSnlds<B> {
    pub fn num_states(&self) -> usize {
        self.q_logits.val().dims()[0]
    }

    pub fn latent_dim_switching(&self) -> usize {
        self.latent_dim
    }

    pub fn total_latent_dim(&self) -> usize {
        self.total_latent_dim
    }

    pub fn res(&self) -> usize {
        self.res
    }

    /// Compute `z_r = cat(z_prefix, z_pca[L..])` — the residual governed by the isotropic prior.
    pub fn compute_z_r(z_pca: &Tensor<B, 2>, z_prefix: &Tensor<B, 2>, l: usize) -> Tensor<B, 2> {
        let [b, d_pca] = z_pca.dims();
        let pca_tail = z_pca.clone().slice([0..b, l..d_pca]);
        let [_, p] = z_prefix.dims();
        if p > 0 {
            Tensor::cat(vec![z_prefix.clone(), pca_tail], 1)
        } else {
            pca_tail
        }
    }
}

impl<B: Backend + PcaSvdBackend> FlowSnlds<B> {
    /// Run the pre-NPCA Dequantize layer on float-`[0, 1]` HWC obs.
    fn dequantize_obs(&self, obs: Tensor<B, 3>, train: bool) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let [n, t, obs_dim] = obs.dims();
        debug_assert_eq!(obs_dim, 3 * self.res * self.res);
        let device = obs.device();

        let flat = obs.reshape([n * t, obs_dim]);
        let x_nchw = flat_nhwc_rows_to_nchw(flat, self.res);
        let x_pix = x_nchw.mul_scalar(255.0).clamp(0.0, 255.0).floor();

        let dq = DequantizeConfig::new(self.pixel_depth).init::<B>(&device);
        if train {
            dq.forward_train(x_pix)
        } else {
            let dims = x_pix.dims();
            let log_det = dq.discretization_penalty(&dims, &device);
            (dq.forward(x_pix), log_det)
        }
    }

    /// Encode flat HWC sequence observations with Neural PCA.
    ///
    /// Returns the NPCA output, the leading `[N, T, L]` latent slice (`z_lead`),
    /// and the per-frame Dequantize log-det `[N*T]`.
    pub fn encode_npca(
        &self,
        obs: Tensor<B, 3>,
        train: bool,
    ) -> (NeuralPcaOutput<B>, Tensor<B, 3>, Tensor<B, 1>) {
        let [n, t, _] = obs.dims();
        let (x4, log_det_dq) = self.dequantize_obs(obs, train);
        let out = self.npca.forward(x4);
        let z_lead = out
            .z_pca
            .clone()
            .slice([0..n * t, 0..self.latent_dim])
            .reshape([n, t, self.latent_dim]);
        (out, z_lead, log_det_dq)
    }

    /// Joint forward: NPCA encode + HMM on `z_lead`.
    ///
    /// `z_lead = z_pca[0..L]` from the rotated suffix.
    /// `z_r = cat(z_prefix, z_pca[L..])` gets the isotropic Gaussian prior.
    pub fn forward(
        &self,
        obs: Tensor<B, 3>,
        w_msm: f32,
        w_npca: f32,
        temperature: f32,
        train: bool,
    ) -> FlowForwardOutput<B> {
        let [batch_size, seq_len, _obs_dim] = obs.dims();
        let device = obs.device();

        let (npca_out, latent_samples, log_det_dq) = self.encode_npca(obs.clone(), train);

        let (msm_loglik, state_posteriors) = if w_msm > 0.0 {
            let log_local = compute_local_evidence(
                latent_samples.clone(),
                &self.transition_nets,
                self.init_mean.val(),
                self.init_cov_factor.val(),
                self.emission_cov_factor.val(),
            );
            let log_pi = log_softmax(self.pi_logits.val() / temperature, 0);
            let log_trans = log_softmax(self.q_logits.val() / temperature, 1);
            let (log_alpha, log_z) = log_forward(log_local.clone(), log_pi, log_trans.clone());
            let msm = log_z.clone().sum_dim(1).reshape([batch_size]).sum() / batch_size as f32;
            let log_beta = log_backward(log_local, log_trans, log_z);
            let posteriors = (log_alpha + log_beta).exp();
            (msm, Some(posteriors))
        } else {
            (Tensor::zeros([1], &device), None)
        };

        // Isotropic Gaussian prior on z_r = cat(z_prefix, z_pca[L..]).
        let z_r = Self::compute_z_r(&npca_out.z_pca, &npca_out.z_prefix, self.latent_dim);
        let log_p_z_r = log_p_z_isotropic(z_r);

        let npca_logprob_flat = npca_out.log_det.clone() + log_p_z_r + log_det_dq;
        let npca_loglik = npca_logprob_flat.clone().sum() / batch_size as f32;
        let npca_logprob_frames = npca_logprob_flat.reshape([batch_size, seq_len]);

        let joint_objective = msm_loglik
            .clone()
            .mul_scalar(w_msm)
            .add(npca_loglik.clone().mul_scalar(w_npca));
        let loss = joint_objective.clone().neg();

        FlowForwardOutput {
            latent_samples,
            state_posteriors,
            joint_objective,
            msm_loglik,
            npca_loglik,
            npca_logprob_frames,
            loss,
            npca_output: npca_out,
        }
    }
}

impl<B: Backend> FlowSnlds<B> {
    /// Decode with the **full** NPCA output from the paired encode (no tail sampling).
    pub fn decode_observations(
        &self,
        z_pca: Tensor<B, 2>,
        z_prefix: Tensor<B, 2>,
        latent_shapes: &[[usize; 4]],
        batch_stats: (Tensor<B, 1>, Tensor<B, 1>),
        seq_shape: (usize, usize),
    ) -> Tensor<B, 3>
    where
        B: TriangularInverse,
    {
        let (n, t) = seq_shape;
        let nt = n * t;
        let [rows, _d] = z_pca.dims();
        debug_assert_eq!(rows, nt, "z_pca rows must equal N*T");

        let frames = self.npca.inverse(z_pca, z_prefix, latent_shapes, batch_stats);
        let flat = frames
            .permute([0, 2, 3, 1])
            .reshape([nt, 3 * self.res * self.res]);
        flat.reshape([n, t, 3 * self.res * self.res])
    }

    /// Build full `(z_pca, z_prefix)` from `z_lead` by sampling `z_r ~ N(0, I)`,
    /// then splitting it back into the prefix and PCA tail.
    pub fn z_pca_with_sampled_tail(
        &self,
        z_lead: Tensor<B, 2>,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [b, l] = z_lead.dims();
        debug_assert_eq!(l, self.latent_dim);
        let p = self.npca.prefix_dim();
        let last_split_dim = self.total_latent_dim - p;
        let pca_tail_dim = last_split_dim - l;
        let z_r_dim = p + pca_tail_dim;

        let z_r: Tensor<B, 2> = Tensor::random(
            [b, z_r_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        let z_prefix = if p > 0 {
            z_r.clone().slice([0..b, 0..p])
        } else {
            Tensor::empty([b, 0], device)
        };
        let pca_tail = z_r.slice([0..b, p..z_r_dim]);
        let z_pca = Tensor::cat(vec![z_lead, pca_tail], 1);

        (z_pca, z_prefix)
    }

    /// Rollout / generative recon: sample tail, then inverse (requires `TriangularInverse`).
    pub fn decode_from_leading_state(
        &self,
        z_lead_nt: Tensor<B, 2>,
        latent_shapes: &[[usize; 4]],
        batch_stats: (Tensor<B, 1>, Tensor<B, 1>),
        seq_shape: (usize, usize),
    ) -> Tensor<B, 3>
    where
        B: TriangularInverse,
    {
        let device = z_lead_nt.device();
        let (z_pca, z_prefix) = self.z_pca_with_sampled_tail(z_lead_nt, &device);
        self.decode_observations(z_pca, z_prefix, latent_shapes, batch_stats, seq_shape)
    }
}
