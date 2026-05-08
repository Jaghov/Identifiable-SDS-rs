//! CNN encoder/decoder pair — Burn port of `CNNFastEncoder` / `CNNFastDecoder`
//! from `identifiable-SDS/models/modules.py`.
//!
//! Both modules expose a 2-D `(batch, features)` interface so they slot into the
//! existing `VariationalSnlds` flow without leaking the 4-D image tensor up to
//! the caller. The CNN reshapes internally to `[B, 3, res, res]` (NCHW).
//!
//! Spatial-shape contract for `n_layers = log2(res / 8)`:
//! - Encoder: `res → res/2 → res/4 → … → 4` after `n_layers + 1` strided convs,
//!   producing a `hidden × 4 × 4` feature map regardless of `res ≥ 16`.
//! - Decoder: starts from `hidden × 8 × 8`, doubles spatial dims `n_layers`
//!   times via `ConvTranspose(stride=2)` + trailing `(0, 1, 0, 1)` pad, ending
//!   at `res × res × 3` after the final 3-channel conv.

use crate::mlp::{Mlp, MlpConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{
        activation::{Activation, ActivationConfig},
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        LeakyReluConfig, PaddingConfig2d,
    },
    tensor::{activation::relu, backend::Backend, Tensor},
};

/// Negative slope used by every CNN activation; matches the Python reference.
const LEAKY_NEGATIVE_SLOPE: f64 = 0.2;
/// Encoder bottleneck spatial side; output of the final strided conv is always
/// `4×4` regardless of input `res`.
const ENCODER_BOTTLENECK_SPATIAL: usize = 4;
/// Decoder input spatial side; the projection MLP feeds an `8×8` feature map.
const DECODER_INPUT_SPATIAL: usize = 8;

/// Number of strided/non-strided conv pairs after the input/output stem so the
/// CNN spatial chain reaches `res`. Mirrors `n_layers = log2(res / 8)` from
/// Python `CNNFastEncoder` / `CNNFastDecoder`.
fn n_layers_for_res(res: usize) -> usize {
    debug_assert!(res >= 16 && res.is_power_of_two());
    res.trailing_zeros() as usize - 3
}

/// Validate a CNN resolution: power of 2 and `≥ 16` (Python parity bottoms out
/// at `n_layers = 1` for `res = 16`).
pub fn validate_cnn_res(res: usize) -> Result<(), String> {
    if !(res >= 16 && res.is_power_of_two()) {
        return Err(format!(
            "EncoderKind::Cnn requires res >= 16 and a power of 2 (got {res})"
        ));
    }
    Ok(())
}

/// Configuration for [`CnnEncoder`].
#[derive(Config, Debug)]
pub struct CnnEncoderConfig {
    /// Input frame side length (also conditions `n_layers`).
    pub res: usize,
    /// Number of channels emitted by the projection MLP (`2 * latent_dim`).
    pub output_dim: usize,
    /// Channel width carried through every conv block.
    pub hidden_dim: usize,
}

impl CnnEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnEncoder<B> {
        validate_cnn_res(self.res).expect("callers must call validate_cnn_res(res) before init");
        let n_layers = n_layers_for_res(self.res);

        let in_conv = Conv2dConfig::new([3, self.hidden_dim], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let mut strided_convs = Vec::with_capacity(n_layers);
        let mut refine_convs = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            strided_convs.push(
                Conv2dConfig::new([self.hidden_dim, self.hidden_dim], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            );
            refine_convs.push(
                Conv2dConfig::new([self.hidden_dim, self.hidden_dim], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            );
        }

        let projection_input_dim =
            self.hidden_dim * ENCODER_BOTTLENECK_SPATIAL * ENCODER_BOTTLENECK_SPATIAL;
        let projection =
            MlpConfig::leaky_relu(projection_input_dim, self.output_dim, self.hidden_dim)
                .init(device);

        let activation = ActivationConfig::LeakyRelu(LeakyReluConfig {
            negative_slope: LEAKY_NEGATIVE_SLOPE,
        })
        .init(device);

        CnnEncoder {
            in_conv,
            strided_convs,
            refine_convs,
            projection,
            activation,
            res: self.res,
            hidden_dim: self.hidden_dim,
        }
    }
}

/// CNN encoder: `[B, 3 * res * res] → [B, output_dim]` with internal NCHW
/// reshape to `[B, 3, res, res]`.
#[derive(Module, Debug)]
pub struct CnnEncoder<B: Backend> {
    pub in_conv: Conv2d<B>,
    pub strided_convs: Vec<Conv2d<B>>,
    pub refine_convs: Vec<Conv2d<B>>,
    pub projection: Mlp<B>,
    pub activation: Activation<B>,
    res: usize,
    hidden_dim: usize,
}

impl<B: Backend> CnnEncoder<B> {
    /// Forward: input shape `[B, 3 * res * res]`, output `[B, output_dim]`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, flat_dim] = input.dims();
        debug_assert_eq!(
            flat_dim,
            3 * self.res * self.res,
            "CnnEncoder expected obs_dim = 3*res*res"
        );

        // NHWC flat → NCHW. The data crate writes pixels in NHWC row-major, so
        // we reshape to [B, res, res, 3] and permute to [B, 3, res, res].
        let nhwc = input.reshape([batch_size, self.res, self.res, 3]);
        let mut feature_map = nhwc.permute([0, 3, 1, 2]);

        // Python parity: `f.relu` on the input conv, `f.leaky_relu(0.2)` on the
        // hidden convs (`identifiable-SDS/models/modules.py:265-268`).
        feature_map = relu(self.in_conv.forward(feature_map));
        for (strided, refine) in self.strided_convs.iter().zip(self.refine_convs.iter()) {
            feature_map = self.activation.forward(strided.forward(feature_map));
            feature_map = self.activation.forward(refine.forward(feature_map));
        }

        let bottleneck_dim =
            self.hidden_dim * ENCODER_BOTTLENECK_SPATIAL * ENCODER_BOTTLENECK_SPATIAL;
        let flattened = feature_map.reshape([batch_size, bottleneck_dim]);
        self.projection.forward(flattened)
    }
}

/// Configuration for [`CnnDecoder`].
#[derive(Config, Debug)]
pub struct CnnDecoderConfig {
    /// Output frame side length (also conditions `n_layers`).
    pub res: usize,
    /// Latent dimensionality consumed by the projection MLP.
    pub input_dim: usize,
    /// Channel width carried through every conv block.
    pub hidden_dim: usize,
}

impl CnnDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnDecoder<B> {
        validate_cnn_res(self.res).expect("callers must call validate_cnn_res(res) before init");
        let n_layers = n_layers_for_res(self.res);

        let projection_output_dim = self.hidden_dim * DECODER_INPUT_SPATIAL * DECODER_INPUT_SPATIAL;
        let projection =
            MlpConfig::leaky_relu(self.input_dim, projection_output_dim, self.hidden_dim)
                .init(device);

        let mut transposed_convs = Vec::with_capacity(n_layers);
        let mut refine_convs = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            transposed_convs.push(
                ConvTranspose2dConfig::new([self.hidden_dim, self.hidden_dim], [3, 3])
                    .with_stride([2, 2])
                    .with_padding([1, 1])
                    .init(device),
            );
            refine_convs.push(
                Conv2dConfig::new([self.hidden_dim, self.hidden_dim], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            );
        }

        let final_conv = Conv2dConfig::new([self.hidden_dim, 3], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let activation = ActivationConfig::LeakyRelu(LeakyReluConfig {
            negative_slope: LEAKY_NEGATIVE_SLOPE,
        })
        .init(device);

        CnnDecoder {
            projection,
            transposed_convs,
            refine_convs,
            final_conv,
            activation,
            res: self.res,
            hidden_dim: self.hidden_dim,
        }
    }
}

/// CNN decoder: `[B, input_dim] → [B, 3 * res * res]`. Returns raw `final_conv`
/// output (no nonlinearity), matching Python `CNNFastDecoder.forward` which
/// returns `self.out_conv(x)` unwrapped.
#[derive(Module, Debug)]
pub struct CnnDecoder<B: Backend> {
    pub projection: Mlp<B>,
    pub transposed_convs: Vec<ConvTranspose2d<B>>,
    pub refine_convs: Vec<Conv2d<B>>,
    pub final_conv: Conv2d<B>,
    pub activation: Activation<B>,
    res: usize,
    hidden_dim: usize,
}

impl<B: Backend> CnnDecoder<B> {
    /// Forward: input shape `[B, input_dim]`, output `[B, 3 * res * res]`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, _] = input.dims();
        let projected = self.projection.forward(input);
        let mut feature_map = projected.reshape([
            batch_size,
            self.hidden_dim,
            DECODER_INPUT_SPATIAL,
            DECODER_INPUT_SPATIAL,
        ]);

        // ConvTranspose(k=3, s=2, p=1) maps H → 2H − 1. Pad (0,1,0,1) on
        // (left, right, top, bottom) recovers the missing right/bottom pixel
        // so the next refinement conv sees an even 2H spatial extent — same
        // as Python `F.pad(x, (0, 1, 0, 1))`.
        for (transposed, refine) in self.transposed_convs.iter().zip(self.refine_convs.iter()) {
            feature_map = self.activation.forward(transposed.forward(feature_map));
            feature_map = feature_map.pad((0, 1, 0, 1), 0.0);
            feature_map = self.activation.forward(refine.forward(feature_map));
        }

        // Output is unbounded (no sigmoid) — Python parity. The Gaussian
        // reconstruction term in the ELBO measures squared distance; downstream
        // consumers that need pixels in `[0, 1]` (e.g. visualisation) must
        // clamp themselves.
        let pixels = self.final_conv.forward(feature_map);
        // NCHW → NHWC then flatten so the layout matches `draw_sequence` output
        // and the data-crate flattening order.
        let nhwc = pixels.permute([0, 2, 3, 1]);
        nhwc.reshape([batch_size, 3 * self.res * self.res])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Backend = NdArray<f32>;

    #[test]
    fn n_layers_matches_python_reference() {
        assert_eq!(n_layers_for_res(16), 1);
        assert_eq!(n_layers_for_res(32), 2);
        assert_eq!(n_layers_for_res(64), 3);
    }

    #[test]
    fn validate_rejects_non_power_of_two_or_too_small() {
        assert!(validate_cnn_res(8).is_err());
        assert!(validate_cnn_res(24).is_err());
        assert!(validate_cnn_res(16).is_ok());
        assert!(validate_cnn_res(32).is_ok());
    }

    #[test]
    fn encoder_output_shape_matches_python() {
        let device = Default::default();
        let res = 32usize;
        let hidden = 8usize;
        let output_dim = 4usize;
        let encoder = CnnEncoderConfig {
            res,
            output_dim,
            hidden_dim: hidden,
        }
        .init::<Backend>(&device);
        let input = Tensor::<Backend, 2>::zeros([2, 3 * res * res], &device);
        let output = encoder.forward(input);
        assert_eq!(output.dims(), [2, output_dim]);
    }

    #[test]
    fn decoder_output_shape_matches_python() {
        let device = Default::default();
        let res = 32usize;
        let hidden = 8usize;
        let input_dim = 2usize;
        let decoder = CnnDecoderConfig {
            res,
            input_dim,
            hidden_dim: hidden,
        }
        .init::<Backend>(&device);
        let input = Tensor::<Backend, 2>::zeros([3, input_dim], &device);
        let output = decoder.forward(input);
        assert_eq!(output.dims(), [3, 3 * res * res]);
    }

    #[test]
    fn encoder_decoder_compose_at_res16() {
        let device = Default::default();
        let res = 16usize;
        let hidden = 4usize;
        let latent_dim = 2usize;
        let encoder = CnnEncoderConfig {
            res,
            output_dim: 2 * latent_dim,
            hidden_dim: hidden,
        }
        .init::<Backend>(&device);
        let decoder = CnnDecoderConfig {
            res,
            input_dim: latent_dim,
            hidden_dim: hidden,
        }
        .init::<Backend>(&device);
        let obs = Tensor::<Backend, 2>::zeros([2, 3 * res * res], &device);
        let encoded = encoder.forward(obs);
        // Take the first `latent_dim` columns (mean) and feed back through the decoder.
        let latent_mean = encoded.slice([0..2, 0..latent_dim]);
        let recon = decoder.forward(latent_mean);
        assert_eq!(recon.dims(), [2, 3 * res * res]);
    }
}
