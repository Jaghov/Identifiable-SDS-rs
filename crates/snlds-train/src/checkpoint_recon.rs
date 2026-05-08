//! Mean squared error between observations and reconstructions at checkpoint time.
//!
//! `recon_mse` is the mean over all elements of `(x - x_hat)²`; checkpoint logs also report
//! `recon_rmse = sqrt(recon_mse)` in the same units as the observations.

use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

pub(crate) fn tensor_mean_mse<B: AutodiffBackend + Backend<FloatElem = f32>, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
) -> f32 {
    let d = a.detach() - b.detach();
    (d.clone() * d).mean().into_scalar()
}
