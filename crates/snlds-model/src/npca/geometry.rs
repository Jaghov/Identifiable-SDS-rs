//! Glow layout helpers for Neural PCA (matches multi-scale `zs` flattening in [`super::flatten_zs`]).

use burn::prelude::Backend;
use burn::tensor::Tensor;
use glow_flow::prelude::{CouplingType, GlowConfig};

/// Total flattened latent dimension `D` produced by Glow with `num_levels` squeezes
/// on an `height x width x in_channels` input. Must match the layout used in
/// [`crate::npca::flatten_zs`] integration tests and Glow-rs multi-level structure.
pub fn glow_flattened_latent_dim(
    in_channels: usize,
    num_levels: usize,
    height: usize,
    width: usize,
) -> usize {
    let (prefix, last) = glow_split_dims(in_channels, num_levels, height, width);
    prefix + last
}

/// Dimension of the **last** Glow split (the deepest level, kept whole — not halved).
pub fn glow_last_split_dim(
    in_channels: usize,
    num_levels: usize,
    height: usize,
    width: usize,
) -> usize {
    glow_split_dims(in_channels, num_levels, height, width).1
}

/// `(prefix_dim, last_split_dim)` — dimensions of the earlier splits concatenated
/// vs the final (deepest) level.
fn glow_split_dims(
    in_channels: usize,
    num_levels: usize,
    height: usize,
    width: usize,
) -> (usize, usize) {
    let mut prefix = 0usize;
    let mut ch = in_channels;
    let mut hh = height;
    let mut ww = width;
    let mut last = 0usize;
    for l in 0..num_levels {
        ch *= 4;
        hh /= 2;
        ww /= 2;
        if l < num_levels - 1 {
            prefix += (ch / 2) * hh * ww;
            ch /= 2;
        } else {
            last = ch * hh * ww;
        }
    }
    (prefix, last)
}

/// Default Glow hyperparameters for Neural PCA training / experiments.
///
/// `coupling_type`: [`CouplingType::Affine`] (full Glow scale) vs [`CouplingType::Additive`]
/// (NICE-style, `log_det = 0` per coupling step in `glow_flow`).
pub fn default_glow_config_for_npca(
    in_channels: usize,
    num_levels: usize,
    num_steps: usize,
    hidden_features: usize,
    coupling_type: CouplingType,
) -> GlowConfig {
    GlowConfig::new(in_channels)
        .with_num_levels(num_levels)
        .with_num_steps(num_steps)
        .with_hidden_features(hidden_features)
        .with_coupling_type(coupling_type)
}

/// Convert flat HWC observation rows `[B, res*res*3]` (same layout as
/// [`crate::cnn::CnnEncoder`]) to NCHW `[B, 3, res, res]` for Glow / NeuralPca.
pub fn flat_nhwc_rows_to_nchw<B: Backend>(flat: Tensor<B, 2>, res: usize) -> Tensor<B, 4> {
    let [batch, flat_dim] = flat.dims();
    debug_assert_eq!(flat_dim, res * res * 3, "expected dim_obs = 3*res*res");
    let nhwc = flat.reshape([batch, res, res, 3]);
    nhwc.permute([0, 3, 1, 2])
}
