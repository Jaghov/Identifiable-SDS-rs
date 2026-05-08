//! PCA-specific BatchNorm: learnable scale `alpha`, no beta (forced to 0).
//!
//! In minibatch mode (`stats` is [`None`]), variance uses **Bessel's correction**
//! (unbiased sample variance): `var_pop * B / (B - 1)` when `B > 1`, matching
//! common BatchNorm implementations. For `B == 1` the population variance is used.

use burn::module::{Module, Param};
use burn::prelude::{Backend, ToElement};
use burn::tensor::Tensor;

const BN_EPS: f32 = 1e-5;

/// PCA BatchNorm layer with `beta = 0`.
#[derive(Module, Debug)]
pub struct PcaBatchNorm<B: Backend> {
    pub alpha: Param<Tensor<B, 1>>,
    /// mean and (unbiase with bessel's correction) var pair
    pub stats: Option<(Param<Tensor<B, 1>>, Param<Tensor<B, 1>>)>,
}

/// Output of [`PcaBatchNorm::forward`].
pub struct BnOutput<B: Backend> {
    pub z_bn: Tensor<B, 2>,
    pub log_det: Tensor<B, 1>,
    pub batch_stats: (Tensor<B, 1>, Tensor<B, 1>),
}

impl<B: Backend> PcaBatchNorm<B> {
    pub fn new(d: usize, device: &B::Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::ones([d], device)),
            stats: Option::None,
        }
    }

    /// Forward pass. `z` has shape `[B, D]`.
    pub fn forward(&self, z: Tensor<B, 2>) -> BnOutput<B> {
        let [batch, _d] = z.dims();
        let device = z.device();

        let (mean, var) = match self.stats {
            None => {
                let mean: Tensor<B, 2> = z.clone().mean_dim(0);
                let diff = z.clone() - mean.clone();
                let var_pop: Tensor<B, 2> = (diff.clone() * diff).mean_dim(0);
                let var = if batch > 1 {
                    var_pop.mul_scalar(batch as f32 / (batch - 1) as f32)
                } else {
                    var_pop
                };
                // Li & Hooi 2022 Remark 3 / Appendix B: detach batch stats so
                // each sample's gradient flows only through its own z value.
                (mean.detach(), var.detach())
            }
            Some((ref mean, ref var)) => {
                let mean: Tensor<B, 2> = mean.val().unsqueeze_dim(0);
                let var: Tensor<B, 2> = var.val().unsqueeze_dim(0);
                (mean.detach(), var.detach())
            }
        };
        let batch_stats = (mean.clone().squeeze(), var.clone().squeeze());

        let eps_t = Tensor::<B, 1>::from_floats([BN_EPS], &device);
        let std_inv = (var.clone() + eps_t.clone().unsqueeze_dim(0))
            .sqrt()
            .recip();
        let z_norm = (z - mean) * std_inv;
        let alpha_2d: Tensor<B, 2> = self.alpha.val().unsqueeze_dim(0);
        let z_bn = z_norm * alpha_2d;

        let log_alpha_sum = scalar_sum(self.alpha.val().clone().abs().log());
        let var_1d: Tensor<B, 1> = var.squeeze();
        let log_var_sum = scalar_sum((var_1d + eps_t).log());
        let log_det_scalar = log_alpha_sum - 0.5 * log_var_sum;
        let log_det = Tensor::<B, 1>::zeros([batch], &device)
            .add_scalar(log_det_scalar)
            .detach();

        BnOutput {
            z_bn,
            log_det,
            batch_stats,
        }
    }

    /// Post training, after rotation frozen. run to set average batchnorm stats
    pub fn set_stats(&mut self, (mean, var): (Tensor<B, 2>, Tensor<B, 2>)) {
        let mean = mean.mean_dim(0).squeeze();
        let var = var.mean_dim(0).squeeze();
        let _ = self.stats.insert((
            Param::from_tensor(mean).no_grad(),
            Param::from_tensor(var).no_grad(),
        ));
    }

    pub fn inverse(
        &self,
        z_bn: Tensor<B, 2>,
        (batch_mean, batch_var): (Tensor<B, 1>, Tensor<B, 1>),
    ) -> Tensor<B, 2> {
        let device = z_bn.device();
        let alpha: Tensor<B, 2> = self.alpha.val().unsqueeze_dim(0);
        let (mean, var) = match &self.stats {
            // If stats have been computed, use stored average stats
            Some((mean, var)) => (mean.val().unsqueeze_dim(0), var.val().unsqueeze_dim(0)),
            // otherwise use the batch stats
            None => (batch_mean.unsqueeze_dim(0), batch_var.unsqueeze_dim(0)),
        };

        let eps = Tensor::<B, 1>::from_floats([BN_EPS], &device).unsqueeze_dim(0);
        let std = (var + eps).sqrt();
        z_bn / alpha * std + mean
    }
}
#[inline]
fn scalar_sum<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.sum().into_scalar().to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray;

    #[test]
    fn bn_forward_output_is_finite() {
        let device = Default::default();
        let bn = PcaBatchNorm::<B>::new(4, &device);
        let z = Tensor::<B, 2>::random([8, 4], Distribution::Normal(5.0, 2.0), &device);
        let out = bn.forward(z);
        let vals: Vec<f32> = out.z_bn.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
        let ld: Vec<f32> = out.log_det.into_data().to_vec().unwrap();
        assert!(ld.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bn_inverse_recovers_input_in_eval_mode() {
        let device = Default::default();
        let bn = PcaBatchNorm::<B>::new(4, &device);
        for _ in 0..20 {
            let z = Tensor::<B, 2>::random([32, 4], Distribution::Normal(3.0, 2.0), &device);
            let _ = bn.forward(z);
        }

        let z = Tensor::<B, 2>::random([8, 4], Distribution::Normal(3.0, 2.0), &device);
        let out = bn.forward(z.clone());
        let recovered = bn.inverse(out.z_bn, out.batch_stats);
        let max_err: f32 = (recovered - z).abs().max().into_scalar();
        assert!(
            max_err < 0.5,
            "inverse should approximately recover input in eval mode; max_err={max_err}"
        );
    }
    #[test]
    fn bn_forward_inverse_round_trip_after_learned_stats() {
        let device = Default::default();
        let mut bn = PcaBatchNorm::<B>::new(4, &device);
        // Fake “per-batch” statistics from M forward passes, each row [D].
        let batch_means = Tensor::<B, 2>::from_floats(
            [
                [1.0_f32, 0.0, -0.5, 2.0],
                [1.2, 0.1, -0.4, 1.9],
                [0.8, -0.1, -0.6, 2.1],
            ],
            &device,
        );
        let batch_vars = Tensor::<B, 2>::from_floats(
            [
                [0.5_f32, 1.0, 0.25, 2.0],
                [0.6, 1.1, 0.30, 1.8],
                [0.4, 0.9, 0.20, 2.2],
            ],
            &device,
        );
        bn.set_stats((batch_means, batch_vars));
        let z = Tensor::<B, 2>::random([8, 4], Distribution::Normal(0.0, 1.0), &device);
        let out = bn.forward(z.clone());
        let z_back = bn.inverse(out.z_bn, out.batch_stats);
        let max_err: f32 = (z_back - z).abs().max().into_scalar();
        assert!(
            max_err < 1e-4,
            "forward/inverse should agree after frozen stats; max_err={max_err}"
        );
    }

    #[test]
    fn bn_minibatch_mode_matches_closed_form() {
        let device = Default::default();
        let bn = PcaBatchNorm::<B>::new(3, &device);
        assert!(bn.stats.is_none());
        let z = Tensor::<B, 2>::from_floats(
            [
                [1.0_f32, 2.0, 4.0],
                [3.0_f32, 4.0, 10.0],
                [5.0_f32, 6.0, 4.0],
            ],
            &device,
        );
        let out = bn.forward(z.clone());

        let mean: Tensor<B, 2> = z.clone().mean_dim(0);
        let diff = z.clone() - mean.clone();
        let var_pop: Tensor<B, 2> = (diff.clone() * diff.clone()).mean_dim(0);
        let b = 3.0_f32;
        let var = var_pop.clone().mul_scalar(b / (b - 1.0));
        let eps_t = Tensor::<B, 1>::from_floats([BN_EPS], &device);
        let expected = (z - mean) * (var + eps_t.clone().unsqueeze_dim(0)).sqrt().recip();

        let max_err: f32 = (out.z_bn - expected).abs().max().into_scalar();
        assert!(
            max_err < 1e-5,
            "minibatch BN z_bn should match closed form; max_err={max_err}"
        );
    }

    #[test]
    fn bn_minibatch_stats_in_output_match_batch_statistics() {
        let device = Default::default();
        let bn = PcaBatchNorm::<B>::new(2, &device);
        assert!(bn.stats.is_none());
        let z = Tensor::<B, 2>::from_floats(
            [[10.0_f32, -2.0_f32], [4.0_f32, 6.0_f32], [0.0_f32, 4.0_f32]],
            &device,
        );
        let out = bn.forward(z.clone());

        let [b, d] = z.dims();
        assert!(b >= 2);
        let bias_corr = b as f32 / (b - 1) as f32;
        let mean_2d: Tensor<B, 2> = z.clone().mean_dim(0);
        let diff = z.clone() - mean_2d.clone();
        let exp_mean: Tensor<B, 1> = mean_2d.reshape([d]);
        let exp_var: Tensor<B, 1> = (diff.clone() * diff)
            .mean_dim(0)
            .reshape([d])
            .mul_scalar(bias_corr);

        let mean_err: f32 = (out.batch_stats.0.clone() - exp_mean)
            .abs()
            .max()
            .into_scalar();
        let var_err: f32 = (out.batch_stats.1.clone() - exp_var)
            .abs()
            .max()
            .into_scalar();
        assert!(
            mean_err < 1e-5,
            "batch_stats mean should match batch; err={mean_err}"
        );
        assert!(
            var_err < 1e-5,
            "batch_stats var should be unbiased (Bessel) sample variance; err={var_err}"
        );
    }

    #[test]
    fn bn_minibatch_batch_size_one_skips_bessel() {
        let device = Default::default();
        let bn = PcaBatchNorm::<B>::new(2, &device);
        let z = Tensor::<B, 2>::from_floats([[1.0_f32, -1.0_f32]], &device);
        let out = bn.forward(z.clone());
        assert!(out
            .z_bn
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|f| f.is_finite()));
    }

    #[test]
    fn bn_forward_uses_stored_stats_not_batch_stats() {
        let device = Default::default();
        let mut bn = PcaBatchNorm::<B>::new(2, &device);
        // Frozen stats: μ=0, σ²=1 so forward is almost identity when α=1.
        let frozen_mean = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0_f32]], &device);
        let frozen_var = Tensor::<B, 2>::from_floats([[1.0_f32, 1.0_f32]], &device);
        bn.set_stats((frozen_mean, frozen_var));
        // Same values on every row, but a huge “wrong” batch would change batch mean/var if used.
        let z = Tensor::<B, 2>::from_floats([[3.0_f32, -1.0_f32]; 4], &device);
        let out = bn.forward(z.clone());
        let expected = (z - 0.0) / (1.0 + BN_EPS).sqrt(); // alpha = 1
        let max_err: f32 = (out.z_bn - expected).abs().max().into_scalar();
        assert!(
            max_err < 1e-4,
            "should normalize with frozen μ=0, σ²=1, not batch stats; max_err={max_err}"
        );
    }
}
