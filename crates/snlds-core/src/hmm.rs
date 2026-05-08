use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Numerically stable logsumexp along `dim`, keeping that dimension (size 1).
fn logsumexp_dim<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let max = x.clone().max_dim(dim);
    (x - max.clone()).exp().sum_dim(dim).log() + max
}

/// HMM forward pass in log-domain.
///
/// Returns `(log_alpha, log_z)` where:
/// - `log_alpha[n, t, k]` = normalised log forward variable
/// - `log_z[n, t]` = log normaliser at each step (for ELBO)
///
/// # Arguments
/// - `log_local_evidence` — `[N, T, K]`
/// - `log_pi` — `[K]` initial log-distribution
/// - `log_trans` — `[K, K]`; `[i, j]` = log p(s_t=j | s_{t-1}=i), already log-softmax'd
pub fn log_forward<B: Backend>(
    log_local_evidence: Tensor<B, 3>,
    log_pi: Tensor<B, 1>,
    log_trans: Tensor<B, 2>,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    let [n, t_len, k] = log_local_evidence.dims();

    let mut log_alphas: Vec<Tensor<B, 2>> = Vec::with_capacity(t_len);
    let mut log_zs: Vec<Tensor<B, 1>> = Vec::with_capacity(t_len);

    // t = 0
    let evidence_0 = log_local_evidence
        .clone()
        .slice([0..n, 0..1, 0..k])
        .reshape([n, k]);
    let log_pi_exp = log_pi.clone().unsqueeze::<2>().expand([n, k]); // [1, K] → [N, K]
    let unnorm_0 = log_pi_exp + evidence_0; // [N, K]
    let log_z0 = logsumexp_dim::<B, 2>(unnorm_0.clone(), 1).reshape([n]); // [N]
    let log_alpha_0 = unnorm_0 - log_z0.clone().unsqueeze_dim::<2>(1).expand([n, k]);
    log_alphas.push(log_alpha_0);
    log_zs.push(log_z0);

    // t = 1..T
    for t in 1..t_len {
        // prev: [N, K]
        let prev = log_alphas
            .last()
            .expect("log_alphas non-empty: pushed init step before loop")
            .clone();
        // [N, K, 1] + [N, K, K] broadcast sum-over-K_from → [N, K_to]
        let prev_3d = prev.unsqueeze_dim::<3>(2); // [N, K_from, 1]
        let log_trans_exp = log_trans.clone().unsqueeze::<3>().expand([n, k, k]); // [N, K_from, K_to]
        let log_pred = logsumexp_dim::<B, 3>(prev_3d + log_trans_exp, 1).reshape([n, k]); // [N, K_to]

        let evidence_t = log_local_evidence
            .clone()
            .slice([0..n, t..t + 1, 0..k])
            .reshape([n, k]);
        let unnorm_t = log_pred + evidence_t;
        let log_zt = logsumexp_dim::<B, 2>(unnorm_t.clone(), 1).reshape([n]);
        let log_alpha_t = unnorm_t - log_zt.clone().unsqueeze_dim::<2>(1).expand([n, k]);
        log_alphas.push(log_alpha_t);
        log_zs.push(log_zt);
    }

    let log_alpha = Tensor::stack(log_alphas, 1); // [N, T, K]
    let log_z = Tensor::stack(log_zs, 1); // [N, T]
    (log_alpha, log_z)
}

/// HMM backward pass in log-domain.
///
/// Returns `log_beta[n, t, k]` — normalised log backward variable.
///
/// # Arguments
/// - `log_local_evidence` — `[N, T, K]`
/// - `log_trans` — `[K, K]` same convention as `log_forward`
/// - `log_z` — `[N, T]` normalisers from `log_forward`
pub fn log_backward<B: Backend>(
    log_local_evidence: Tensor<B, 3>,
    log_trans: Tensor<B, 2>,
    log_z: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [n, t_len, k] = log_local_evidence.dims();
    let device = log_local_evidence.device();

    let mut log_betas: Vec<Tensor<B, 2>> = vec![Tensor::zeros([n, k], &device); t_len];

    for t in (0..t_len - 1).rev() {
        let next_beta = log_betas[t + 1].clone(); // [N, K_to]
        let evidence_next = log_local_evidence
            .clone()
            .slice([0..n, t + 1..t + 2, 0..k])
            .reshape([n, k]); // [N, K_to]
        let log_z_next = log_z.clone().slice([0..n, t + 1..t + 2]).reshape([n]); // [N]

        // Combine: beta + evidence - log_z, shape [N, K_to]
        let combined = next_beta + evidence_next - log_z_next.unsqueeze_dim::<2>(1).expand([n, k]);

        // [N, 1, K_to] + [N, K_from, K_to] → logsumexp over K_to → [N, K_from]
        let combined_3d = combined.unsqueeze_dim::<3>(1); // [N, 1, K_to]
        let log_trans_exp = log_trans.clone().unsqueeze::<3>().expand([n, k, k]); // [N, K_from, K_to]
        let log_beta_t = logsumexp_dim::<B, 3>(combined_3d + log_trans_exp, 2).reshape([n, k]);

        log_betas[t] = log_beta_t;
    }

    Tensor::stack(log_betas, 1) // [N, T, K]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type B = NdArray<f32>;

    fn dev() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn uniform_log(k: usize, device: &NdArrayDevice) -> Tensor<B, 1> {
        Tensor::full([k], -(k as f32).ln(), device)
    }

    fn uniform_log_trans(k: usize, device: &NdArrayDevice) -> Tensor<B, 2> {
        Tensor::full([k, k], -(k as f32).ln(), device)
    }

    #[test]
    fn forward_shapes_and_finite() {
        let dev = dev();
        let (n, t, k) = (2, 4, 3);
        let evidence = Tensor::<B, 3>::zeros([n, t, k], &dev);
        let (log_alpha, log_z) =
            log_forward::<B>(evidence, uniform_log(k, &dev), uniform_log_trans(k, &dev));
        assert_eq!(log_alpha.dims(), [n, t, k]);
        assert_eq!(log_z.dims(), [n, t]);
        assert!(log_alpha
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        assert!(log_z
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn backward_shapes_and_finite() {
        let dev = dev();
        let (n, t, k) = (2, 4, 3);
        let evidence = Tensor::<B, 3>::zeros([n, t, k], &dev);
        let (_, log_z) = log_forward::<B>(
            evidence.clone(),
            uniform_log(k, &dev),
            uniform_log_trans(k, &dev),
        );
        let log_beta = log_backward::<B>(evidence, uniform_log_trans(k, &dev), log_z);
        assert_eq!(log_beta.dims(), [n, t, k]);
        assert!(log_beta
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn forward_uniform_alpha_rows_sum_to_one() {
        let dev = dev();
        let (n, t, k) = (1, 3, 2);
        let evidence = Tensor::<B, 3>::zeros([n, t, k], &dev); // log(1)=0 → uniform
        let (log_alpha, _) =
            log_forward::<B>(evidence, uniform_log(k, &dev), uniform_log_trans(k, &dev));
        let alpha = log_alpha.exp().into_data().to_vec::<f32>().unwrap();
        for chunk in alpha.chunks(k) {
            let s: f32 = chunk.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "alpha row sum {s} != 1");
        }
    }

    #[test]
    fn alpha_beta_posterior_sums_to_one() {
        let dev = dev();
        let (n, t, k) = (2, 5, 3);
        let evidence = Tensor::<B, 3>::zeros([n, t, k], &dev);
        let log_trans = uniform_log_trans(k, &dev);
        let (log_alpha, log_z) =
            log_forward::<B>(evidence.clone(), uniform_log(k, &dev), log_trans.clone());
        let log_beta = log_backward::<B>(evidence, log_trans, log_z);
        let posterior = (log_alpha + log_beta)
            .exp()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for chunk in posterior.chunks(k) {
            let s: f32 = chunk.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "posterior row sum {s} != 1");
        }
    }

    #[test]
    fn deterministic_spike_dominates() {
        let dev = dev();
        let (n, t, k) = (1, 3, 3);
        let mut ev_data = vec![-1000.0_f32; n * t * k];
        for step in 0..t {
            ev_data[step * k] = 0.0; // state 0 has log evidence = 0; rest ≈ -∞
        }
        let evidence =
            Tensor::<B, 3>::from_floats(burn::tensor::TensorData::new(ev_data, [n, t, k]), &dev);
        let (log_alpha, _) =
            log_forward::<B>(evidence, uniform_log(k, &dev), uniform_log_trans(k, &dev));
        let alpha = log_alpha.exp().into_data().to_vec::<f32>().unwrap();
        for step in 0..t {
            let row = &alpha[step * k..(step + 1) * k];
            assert!(
                row[0] > 0.99,
                "state 0 should dominate at t={step}, got {}",
                row[0]
            );
        }
    }
}
