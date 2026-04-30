#[cfg(test)]
mod model_tests {
    use crate::model::{EncoderKind, SnldsConfig};
    use burn::backend::cpu::CpuDevice;
    use burn::backend::{Autodiff, Cpu};
    use burn::tensor::Tensor;

    type CpuBackend = Cpu<f32>;
    type AutodiffBackend = Autodiff<Cpu<f32>>;

    fn small_config() -> SnldsConfig {
        SnldsConfig::new(4, 2, 8, 3)
    }

    fn cpu_device() -> CpuDevice {
        CpuDevice
    }

    #[test]
    fn forward_output_shapes() {
        let device = cpu_device();
        let model = small_config().init::<CpuBackend>(&device);

        let batch_size = 2;
        let seq_len = 5;
        let obs_dim = 4;
        let latent_dim = 2;
        let num_states = 3;

        let obs = Tensor::<CpuBackend, 3>::random(
            [batch_size, seq_len, obs_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(obs, 1.0, 5e-4, 1.0);

        assert_eq!(
            output.obs_reconstructed.dims(),
            [batch_size, seq_len, obs_dim]
        );
        assert_eq!(
            output.latent_samples.dims(),
            [batch_size, seq_len, latent_dim]
        );
        assert_eq!(output.elbo.dims(), [1]);
        assert_eq!(output.recon_loss.dims(), [1]);
        assert_eq!(output.entropy_q.dims(), [1]);
        assert_eq!(output.msm_loss.dims(), [1]);

        let posteriors = output
            .state_posteriors
            .expect("state_posteriors should be Some when beta=1.0");
        assert_eq!(posteriors.dims(), [batch_size, seq_len, num_states]);
    }

    #[test]
    fn forward_loss_finite() {
        let device = cpu_device();
        let model = small_config().init::<CpuBackend>(&device);

        let obs = Tensor::<CpuBackend, 3>::random(
            [2, 5, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(obs, 1.0, 5e-4, 1.0);

        let elbo_val = output.elbo.into_data().to_vec::<f32>().unwrap()[0];
        let recon_val = output.recon_loss.into_data().to_vec::<f32>().unwrap()[0];
        let entropy_val = output.entropy_q.into_data().to_vec::<f32>().unwrap()[0];
        let msm_val = output.msm_loss.into_data().to_vec::<f32>().unwrap()[0];

        assert!(elbo_val.is_finite(), "ELBO is not finite: {elbo_val}");
        assert!(
            recon_val.is_finite(),
            "recon_loss is not finite: {recon_val}"
        );
        assert!(
            entropy_val.is_finite(),
            "entropy_q is not finite: {entropy_val}"
        );
        assert!(msm_val.is_finite(), "msm_loss is not finite: {msm_val}");
    }

    #[test]
    fn forward_posteriors_sum_to_one() {
        let device = cpu_device();
        let model = small_config().init::<CpuBackend>(&device);

        let obs = Tensor::<CpuBackend, 3>::random(
            [2, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(obs, 1.0, 5e-4, 1.0);
        let posteriors = output
            .state_posteriors
            .expect("posteriors should be Some when beta>0");

        let row_sums = posteriors.sum_dim(2).into_data().to_vec::<f32>().unwrap();
        for sum in &row_sums {
            assert!((sum - 1.0_f32).abs() < 1e-4, "posterior row sum {sum} != 1");
        }
    }

    #[test]
    fn beta_zero_disables_msm() {
        let device = cpu_device();
        let model = small_config().init::<CpuBackend>(&device);
        let obs = Tensor::<CpuBackend, 3>::random(
            [2, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(obs, 0.0, 5e-4, 1.0);
        let msm_val = output.msm_loss.into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(msm_val, 0.0, "msm_loss should be 0 when beta=0");
        assert!(
            output.state_posteriors.is_none(),
            "posteriors should be None when beta=0"
        );
    }

    /// Exercises the early-return branch in `compute_local_evidence` where `seq_len == 1`.
    /// If that branch is removed, slicing `0..0` on the time axis would produce wrong shapes.
    #[test]
    fn seq_len_one_exercises_init_only_branch() {
        let device = cpu_device();
        let model = small_config().init::<CpuBackend>(&device);

        let obs = Tensor::<CpuBackend, 3>::random(
            [2, 1, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(obs, 1.0, 5e-4, 1.0);

        assert_eq!(output.obs_reconstructed.dims(), [2, 1, 4]);
        assert_eq!(output.latent_samples.dims(), [2, 1, 2]);

        let elbo_val = output.elbo.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            elbo_val.is_finite(),
            "ELBO should be finite for seq_len=1, got {elbo_val}"
        );
    }

    #[test]
    fn autodiff_gradient_smoke() {
        let device = cpu_device();
        let model = small_config().init::<AutodiffBackend>(&device);

        let obs = Tensor::<AutodiffBackend, 3>::random(
            [2, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(obs, 1.0, 5e-4, 1.0);
        // Negate ELBO to get loss, then backward
        let loss = output.elbo.neg().sum();
        let gradients = loss.backward();

        // Check that gradients exist and are finite for the decoder's first linear weight.
        // Default config uses EncoderKind::Mlp, so the decoder is the MLP variant.
        let decoder_mlp = match &model.decoder {
            crate::model::SnldsDecoder::Mlp(mlp) => mlp,
            crate::model::SnldsDecoder::Cnn(_) => {
                panic!("small_config uses EncoderKind::Mlp; CNN decoder unexpected")
            }
        };
        let decoder_first_weight_grad = decoder_mlp
            .first_linear
            .weight
            .grad(&gradients)
            .expect("gradient for decoder first_linear.weight should exist");

        let grad_data = decoder_first_weight_grad
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            grad_data.iter().all(|value| value.is_finite()),
            "decoder gradients contain non-finite values"
        );
    }

    /// End-to-end forward through `VariationalSnlds` with `EncoderKind::Cnn`.
    /// Asserts that image-shaped observations round-trip to themselves and
    /// that the ELBO is finite. Kept minimal (`res=16, hidden=4, T=2`) because
    /// the CPU backend's conv kernels are slow.
    #[test]
    fn forward_cnn_shapes_and_elbo_finite() {
        let device = cpu_device();
        let res = 16usize;
        let obs_dim = 3 * res * res;
        let latent_dim = 2usize;
        let hidden_dim = 4usize;
        let num_states = 3usize;

        let config = SnldsConfig::new(obs_dim, latent_dim, hidden_dim, num_states)
            .with_kind(EncoderKind::Cnn { res });
        let model = config.init::<CpuBackend>(&device);

        let batch_size = 2;
        let seq_len = 2;
        let obs = Tensor::<CpuBackend, 3>::random(
            [batch_size, seq_len, obs_dim],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let output = model.forward(obs, 1.0, 5e-4, 1.0);

        assert_eq!(
            output.obs_reconstructed.dims(),
            [batch_size, seq_len, obs_dim]
        );
        assert_eq!(
            output.latent_samples.dims(),
            [batch_size, seq_len, latent_dim]
        );

        let elbo_val = output.elbo.into_data().to_vec::<f32>().unwrap()[0];
        assert!(elbo_val.is_finite(), "CNN ELBO not finite: {elbo_val}");
    }
}
