//! Learned Householder reflector stack: volume-preserving orthogonal map on `R^D`.

use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;

const HH_EPS: f32 = 1e-8;

/// `K` successive Householder maps `H_K ... H_1` with unit vectors `v_k ∈ R^D`.
///
/// Forward: `z ← z - 2 v (v·z)` for each `k` in order. Inverse applies the same reflections in
/// reverse order. Each step is orthogonal, so **log |det J| = 0** for this block.
#[derive(Module, Debug)]
pub struct HouseholderStack<B: Backend> {
    /// Unnormalized directions, shape `[K, D]`. Rows are normalized on the forward path.
    pub raw: Param<Tensor<B, 2>>,
}

impl<B: Backend> HouseholderStack<B> {
    pub fn new(num_reflectors: usize, dim: usize, device: &B::Device) -> Self {
        assert!(num_reflectors > 0, "HouseholderStack requires K >= 1");
        assert!(dim > 0, "HouseholderStack requires D >= 1");
        let raw = Tensor::random(
            [num_reflectors, dim],
            burn::tensor::Distribution::Normal(0.0, 0.01),
            device,
        );
        Self {
            raw: Param::from_tensor(raw),
        }
    }

    /// `z`: `[B, D]` → `[B, D]`.
    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.apply_stack(z, false)
    }

    pub fn inverse(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.apply_stack(z, true)
    }

    fn apply_stack(&self, mut z: Tensor<B, 2>, reverse: bool) -> Tensor<B, 2> {
        let [b_sz, d] = z.dims();
        let raw = self.raw.val();
        let [k, d_raw] = raw.dims();
        debug_assert_eq!(d_raw, d);

        let sq_norm = (raw.clone() * raw.clone())
            .sum_dim(1)
            .clamp_min(HH_EPS * HH_EPS);
        let norms = sq_norm.sqrt();
        let v = raw / norms;

        let indices: Vec<usize> = if reverse {
            (0..k).rev().collect()
        } else {
            (0..k).collect()
        };

        for idx in indices {
            let vk = v.clone().slice([idx..idx + 1, 0..d]);
            let vk_rows = vk.repeat_dim(0, b_sz);
            let dot_col = (z.clone() * vk_rows.clone()).sum_dim(1);
            z = z - dot_col.mul_scalar(2.0) * vk_rows;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Param;

    #[test]
    fn householder_single_vector_matches_hand_expectation() {
        type B = NdArray<f32>;
        let device = Default::default();
        let mut hh = HouseholderStack::<B>::new(1, 3, &device);
        hh.raw = Param::from_tensor(Tensor::from_data([[1.0f32, 0.0, 0.0]], &device));
        let z = Tensor::from_data([[1.0f32, 2.0, 3.0]], &device);
        let y = hh.forward(z);
        let expect = Tensor::from_data([[-1.0f32, 2.0, 3.0]], &device);
        let err: f32 = (y - expect).abs().max().into_scalar();
        assert!(err < 1e-5, "err={err}");
    }

    #[test]
    fn householder_two_batch_second_row_unchanged_when_orthogonal_to_v() {
        type B = NdArray<f32>;
        let device = Default::default();
        let mut hh = HouseholderStack::<B>::new(1, 3, &device);
        hh.raw = Param::from_tensor(Tensor::from_data([[1.0f32, 0.0, 0.0]], &device));
        let z = Tensor::from_data([[1.0f32, 0.0, 0.0], [0.0f32, 1.0, 0.0]], &device);
        let y = hh.forward(z);
        let expect = Tensor::from_data([[-1.0f32, 0.0, 0.0], [0.0f32, 1.0, 0.0]], &device);
        let err: f32 = (y - expect).abs().max().into_scalar();
        assert!(err < 1e-5, "err={err}");
    }

    #[test]
    fn householder_round_trip_identity() {
        type B = NdArray<f32>;
        let device = Default::default();
        let k = 5usize;
        let d = 12usize;
        let hh = HouseholderStack::<B>::new(k, d, &device);
        let z = Tensor::<B, 2>::random(
            [7, d],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z2 = hh.forward(z.clone());
        let z_back = hh.inverse(z2);
        let err: f32 = (z.clone() - z_back).abs().max().into_scalar();
        assert!(
            err < 1e-4,
            "inverse(forward(z)) should match z, max abs err={err}"
        );
    }

    #[test]
    fn householder_preserves_norm() {
        type B = NdArray<f32>;
        let device = Default::default();
        let hh = HouseholderStack::<B>::new(8, 16, &device);
        let z = Tensor::<B, 2>::random(
            [4, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z2 = hh.forward(z.clone());
        let n0: f32 = (z.clone() * z).sum_dim(1).sqrt().mean().into_scalar();
        let n1: f32 = (z2.clone() * z2).sum_dim(1).sqrt().mean().into_scalar();
        assert!(
            (n0 - n1).abs() < 1e-3,
            "norm should be preserved, n0={n0} n1={n1}"
        );
    }
}
