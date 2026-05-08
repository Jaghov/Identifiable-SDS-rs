//! Isotropic Gaussian prior for the residual latent `z_r`.

use burn::prelude::Backend;
use burn::tensor::Tensor;
use std::f32::consts::TAU;

/// Log-density of `z` under a standard isotropic Gaussian `N(0, I)`.
///
/// `z` has shape `[B, D]`. Returns `[B]`:
/// `-0.5 * (||z||^2 + D * ln(2π))`.
pub fn log_p_z_isotropic<B: Backend>(z: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, d] = z.dims();
    let const_term = -0.5 * d as f32 * TAU.ln();
    let z_sq_sum: Tensor<B, 1> = (z.clone() * z).sum_dim(1).reshape([batch]);
    z_sq_sum.mul_scalar(-0.5_f32).add_scalar(const_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn log_p_isotropic_matches_hand_computed() {
        let device = Default::default();
        let z = Tensor::<B, 2>::from_floats([[1.0_f32, 2.0]], &device);
        let result: f32 = log_p_z_isotropic::<B>(z).into_scalar();
        let expected = -0.5 * 5.0 - TAU.ln();
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn log_p_isotropic_zero_is_normalising_constant() {
        let device = Default::default();
        let z = Tensor::<B, 2>::zeros([1, 3], &device);
        let result: f32 = log_p_z_isotropic::<B>(z).into_scalar();
        let expected = -1.5 * TAU.ln();
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }
}
