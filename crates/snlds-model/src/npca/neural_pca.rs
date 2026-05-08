//! `NeuralPCA` wrapper: Glow + block (BatchNorm + SVD rotation **or** Householder stack).

use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use glow_flow::prelude::{Glow, GlowConfig, TriangularInverse};

use super::flatten::{flatten_zs, unflatten_zs};
use super::householder::HouseholderStack;
use super::pca_batchnorm::{BnOutput, PcaBatchNorm};
use super::pca_rotate::{project_mean_rotation, PcaRotate, PcaSvdBackend, RotateOutput};

#[derive(Config, Debug)]
pub struct NeuralPcaConfig {
    pub glow: GlowConfig,
    pub total_latent_dim: usize,
    /// Dimension of the last (deepest) Glow split. BN + rotation only operate on
    /// these dims; earlier splits pass through unchanged.
    pub last_split_dim: usize,
    /// When true, use a learned Householder stack instead of per-batch SVD rotation.
    #[config(default = false)]
    pub householder_rotation: bool,
    /// Number of reflectors `K` (clamped to `[1, D]` at init). Ignored when `householder_rotation` is false.
    #[config(default = "32")]
    pub householder_reflectors: usize,
}

impl NeuralPcaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NeuralPca<B> {
        let glow = self.glow.init::<B>(device);
        let d = self.total_latent_dim;
        let d_npca = self.last_split_dim;
        assert!(
            d_npca > 0 && d_npca <= d,
            "last_split_dim must be in (0, {d}], got {d_npca}"
        );
        let prefix_dim = d - d_npca;

        let rotation = if self.householder_rotation {
            let k = self.householder_reflectors.min(d_npca).max(1);
            PostBnRotation::Householder(HouseholderStack::new(k, d_npca, device))
        } else {
            PostBnRotation::Svd(PcaRotate::new(d_npca, device))
        };

        NeuralPca {
            glow,
            prefix_dim,
            batchnorm: PcaBatchNorm::new(d_npca, device),
            rotation,
        }
    }
}

#[derive(Debug)]
pub struct NeuralPcaOutput<B: Backend> {
    /// Rotated suffix from the last Glow split, shape `[B, last_split_dim]`.
    pub z_pca: Tensor<B, 2>,
    /// Unrotated prefix from earlier Glow splits, shape `[B, prefix_dim]`.
    /// Empty (zero-width) when the Glow has only one level.
    pub z_prefix: Tensor<B, 2>,
    /// `log|det ∂z/∂x|` per sample, shape `[B]`.
    pub log_det: Tensor<B, 1>,
    pub batch_stats: (Tensor<B, 1>, Tensor<B, 1>),
    pub v_matrix: Option<Tensor<B, 2>>,
    pub latent_shapes: Vec<[usize; 4]>,
}

#[derive(Module, Debug)]
enum PostBnRotation<B: Backend> {
    Svd(PcaRotate<B>),
    Householder(HouseholderStack<B>),
}

#[derive(Module, Debug)]
pub struct NeuralPca<B: Backend> {
    pub glow: Glow<B>,
    /// Number of flattened dims from earlier Glow splits that bypass BN+rotation.
    /// Zero when NPCA processes all dims.
    #[module(ignore)]
    prefix_dim: usize,
    pub batchnorm: PcaBatchNorm<B>,
    rotation: PostBnRotation<B>,
}

impl<B: Backend> NeuralPca<B> {
    /// True when the post-BN map is a Householder stack (no SVD / no `v_matrix` in forward).
    pub fn rotation_is_householder(&self) -> bool {
        matches!(self.rotation, PostBnRotation::Householder(_))
    }

    /// Freeze **BatchNorm only** (and set training off). Used for Householder checkpoints (no `V` to average).
    pub fn freeze_bn_stats_only(
        &mut self,
        bn_batch_means: Tensor<B, 2>,
        bn_batch_vars: Tensor<B, 2>,
    ) {
        self.batchnorm.set_stats((bn_batch_means, bn_batch_vars));
        self.set_training(false);
    }

    /// Freeze SVD rotation (`mean rotation` of `V`) and BatchNorm to averaged training statistics.
    ///
    /// `bn_batch_means` / `bn_batch_vars`: shape `[M, D]` — one row per forward pass's
    /// batch mean / variance after Φ (same as [`NeuralPcaOutput::batch_stats`] when `stats` is `None`).
    pub fn freeze_stats(
        &mut self,
        v_matrices: Tensor<B, 3>,
        bn_batch_means: Tensor<B, 2>,
        bn_batch_vars: Tensor<B, 2>,
        device: &B::Device,
    ) {
        match &mut self.rotation {
            PostBnRotation::Svd(r) => {
                let v_tilde = project_mean_rotation::<B>(v_matrices, device);
                r.freeze(v_tilde);
            }
            PostBnRotation::Householder(_) => {
                let _ = (v_matrices, device);
            }
        }
        self.batchnorm.set_stats((bn_batch_means, bn_batch_vars));
        self.set_training(false);
    }

    pub fn set_training(&mut self, training: bool) {
        match &mut self.rotation {
            PostBnRotation::Svd(r) => r.set_training(training),
            PostBnRotation::Householder(_) => {}
        }
    }

    /// Number of prefix dims that bypass NPCA (0 when processing all dims).
    pub fn prefix_dim(&self) -> usize {
        self.prefix_dim
    }
}

impl<B: Backend + PcaSvdBackend> NeuralPca<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> NeuralPcaOutput<B> {
        let device = x.device();
        let (zs, glow_log_det) = self.glow.forward(x);

        let (z_flat, shapes) = flatten_zs(&zs);
        let [batch, d_total] = z_flat.dims();

        let z_suffix = if self.prefix_dim > 0 {
            z_flat.clone().slice([0..batch, self.prefix_dim..d_total])
        } else {
            z_flat.clone()
        };

        let z_prefix = if self.prefix_dim > 0 {
            z_flat.slice([0..batch, 0..self.prefix_dim])
        } else {
            Tensor::empty([batch, 0], &device)
        };

        let BnOutput {
            z_bn,
            log_det: bn_log_det,
            batch_stats,
        } = self.batchnorm.forward(z_suffix);

        let (z_pca, v_matrix) = match &self.rotation {
            PostBnRotation::Svd(r) => {
                let RotateOutput { z_pca, v_matrix } = r.forward(z_bn);
                (z_pca, v_matrix)
            }
            PostBnRotation::Householder(h) => (h.forward(z_bn), None),
        };

        let total_log_det = glow_log_det + bn_log_det;

        NeuralPcaOutput {
            z_pca,
            z_prefix,
            log_det: total_log_det,
            batch_stats,
            v_matrix,
            latent_shapes: shapes,
        }
    }
}

impl<B: Backend<FloatElem = f32>> NeuralPca<B> {
    /// After [`burn::module::Module::load_record`], set rotate eval mode when
    /// [`PcaRotate::v_tilde`] is not the identity (frozen checkpoints replace it;
    /// the `training` flag is not always restored by the recorder).
    pub fn sync_training_mode_after_load(&mut self) {
        match &self.rotation {
            PostBnRotation::Svd(r) => {
                let v = r.v_tilde.val();
                let d = v.dims()[0];
                let eye = Tensor::eye(d, &v.device());
                let dist: f32 = (v.clone() - eye).abs().mean().into_scalar();
                if dist > 1e-4f32 {
                    self.set_training(false);
                }
            }
            PostBnRotation::Householder(_) => {
                self.set_training(false);
            }
        }
    }
}

impl<B: Backend + TriangularInverse> NeuralPca<B> {
    /// Inverse: from `(z_pca, z_prefix)` back to `x`.
    ///
    /// `z_pca`: rotated suffix, shape `[B, last_split_dim]`.
    /// `z_prefix`: unrotated prefix, shape `[B, prefix_dim]`.
    pub fn inverse(
        &self,
        z_pca: Tensor<B, 2>,
        z_prefix: Tensor<B, 2>,
        latent_shapes: &[[usize; 4]],
        batch_stats: (Tensor<B, 1>, Tensor<B, 1>),
    ) -> Tensor<B, 4> {
        let z_bn = match &self.rotation {
            PostBnRotation::Svd(r) => r.inverse(z_pca),
            PostBnRotation::Householder(h) => h.inverse(z_pca),
        };
        let z_suffix = self.batchnorm.inverse(z_bn, batch_stats);

        let z_flat = if self.prefix_dim > 0 {
            Tensor::cat(vec![z_prefix, z_suffix], 1)
        } else {
            z_suffix
        };

        let zs = unflatten_zs(z_flat, latent_shapes);
        self.glow.inverse(zs)
    }
}
