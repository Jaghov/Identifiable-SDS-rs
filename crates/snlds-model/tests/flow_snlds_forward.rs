//! FlowSNLDS forward smoke test.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use snlds_model::FlowSnldsConfig;

type B = NdArray<f32>;

#[test]
fn flow_snlds_forward_shapes_joint_finite() {
    let device = Default::default();
    let res = 16usize;
    let obs_dim = 3 * res * res;
    let latent_dim = 2usize;
    let hidden_dim = 4usize;
    let num_states = 3usize;

    let cfg = FlowSnldsConfig::new(obs_dim, latent_dim, hidden_dim, num_states, res, 2, 2, 8)
        .with_pixel_depth(8);
    let model = cfg.init::<B>(&device);

    let batch_size = 2;
    let seq_len = 2;
    let obs = Tensor::<B, 3>::random(
        [batch_size, seq_len, obs_dim],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let out = model.forward(obs, 1.0, 1.0, 1.0, false);

    assert_eq!(out.latent_samples.dims(), [batch_size, seq_len, latent_dim]);
    assert_eq!(out.npca_logprob_frames.dims(), [batch_size, seq_len]);
    assert!(out.msm_loglik.into_data().to_vec::<f32>().unwrap()[0].is_finite());
    assert!(out.npca_loglik.into_data().to_vec::<f32>().unwrap()[0].is_finite());
    let j = out.joint_objective.into_data().to_vec::<f32>().unwrap()[0];
    assert!(j.is_finite(), "joint_objective {j}");
    let posteriors = out.state_posteriors.expect("w_msm>0");
    assert_eq!(posteriors.dims(), [batch_size, seq_len, num_states]);
}

#[test]
fn flow_snlds_backward_transition_grad_nonzero() {
    type AD = Autodiff<NdArray<f32>>;

    let device = Default::default();
    let res = 16usize;
    let obs_dim = 3 * res * res;
    let cfg = FlowSnldsConfig::new(obs_dim, 2, 4, 3, res, 2, 2, 8).with_pixel_depth(8);
    let model = cfg.init::<AD>(&device);
    let obs = Tensor::<AD, 3>::random(
        [1, 2, obs_dim],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let out = model.forward(obs, 1.0, 1.0, 1.0, false);
    let loss = out.loss.sum();
    let grads = loss.backward();
    let ts = model.transition_nets[0]
        .first_linear
        .weight
        .grad(&grads)
        .expect("transition net grad");
    let v = ts.into_data().to_vec::<f32>().unwrap();
    assert!(
        v.iter().any(|x| *x != 0.0),
        "expected non-zero transition grad"
    );
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn flow_decode_matches_obs_shape() {
    let device = Default::default();
    let res = 16usize;
    let obs_dim = 3 * res * res;
    let cfg = FlowSnldsConfig::new(obs_dim, 2, 4, 3, res, 2, 2, 8).with_pixel_depth(8);
    let model = cfg.init::<B>(&device);
    let obs = Tensor::<B, 3>::random(
        [1, 2, obs_dim],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let out = model.forward(obs.clone(), 1.0, 1.0, 1.0, false);
    let x_hat = model.decode_observations(
        out.npca_output.z_pca.clone(),
        out.npca_output.z_prefix.clone(),
        &out.npca_output.latent_shapes,
        out.npca_output.batch_stats.clone(),
        (1, 2),
    );
    assert_eq!(x_hat.dims(), [1, 2, obs_dim]);
}
