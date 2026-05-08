//! Flatten / unflatten multi-scale Glow latents `Vec<Tensor<B,4>>` ↔ `Tensor<B,2>`.

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Flatten per-level `[B, C, H, W]` tensors into a single `[B, D]` where `D = sum(C*H*W)`.
///
/// Returns the flat tensor and the per-level shapes needed by [`unflatten_zs`].
pub fn flatten_zs<B: Backend>(zs: &[Tensor<B, 4>]) -> (Tensor<B, 2>, Vec<[usize; 4]>) {
    let shapes: Vec<[usize; 4]> = zs.iter().map(|z| z.dims()).collect();
    let batch = shapes[0][0];
    let parts: Vec<Tensor<B, 2>> = zs
        .iter()
        .map(|z| {
            let [b, c, h, w] = z.dims();
            debug_assert_eq!(b, batch);
            z.clone().reshape([b, c * h * w])
        })
        .collect();
    let flat = Tensor::cat(parts, 1);
    (flat, shapes)
}

/// Inverse of [`flatten_zs`]: split `[B, D]` back into per-level `[B, C, H, W]` tensors.
pub fn unflatten_zs<B: Backend>(flat: Tensor<B, 2>, shapes: &[[usize; 4]]) -> Vec<Tensor<B, 4>> {
    let batch = flat.dims()[0];
    let mut offset = 0usize;
    shapes
        .iter()
        .map(|&[_b, c, h, w]| {
            let d = c * h * w;
            let slice = flat.clone().slice([0..batch, offset..offset + d]);
            offset += d;
            slice.reshape([batch, c, h, w])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray;

    #[test]
    fn flatten_unflatten_round_trip() {
        let device = Default::default();
        let z0 = Tensor::<B, 4>::random([2, 6, 4, 4], Distribution::Normal(0.0, 1.0), &device);
        let z1 = Tensor::<B, 4>::random([2, 24, 2, 2], Distribution::Normal(0.0, 1.0), &device);
        let zs = vec![z0.clone(), z1.clone()];

        let (flat, shapes) = flatten_zs(&zs);
        assert_eq!(flat.dims(), [2, 6 * 4 * 4 + 24 * 2 * 2]);
        assert_eq!(shapes.len(), 2);

        let recovered = unflatten_zs(flat, &shapes);
        assert_eq!(recovered.len(), 2);
        assert!(recovered[0].clone().all_close(z0, Some(1e-6), Some(1e-6)));
        assert!(recovered[1].clone().all_close(z1, Some(1e-6), Some(1e-6)));
    }
}
