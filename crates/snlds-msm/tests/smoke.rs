//! Smoke tests for the M5 NeuralMSM warm-start path.

use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};
use ndarray::Array3;
use snlds_model::SnldsConfig;
use snlds_msm::{pca_fit_transform, transfer_into_snlds, NeuralMsmConfig};

type TestBackend = Autodiff<NdArray<f32>>;

fn deterministic_obs(num_sequences: usize, seq_length: usize, obs_dim: usize) -> Array3<f32> {
    let mut data = Vec::with_capacity(num_sequences * seq_length * obs_dim);
    for seq_idx in 0..num_sequences {
        for time_idx in 0..seq_length {
            for dim_idx in 0..obs_dim {
                // Use distinct frequencies per dimension so that all axes carry
                // independent variance — otherwise linfa-reduction's PCA drops
                // numerically-degenerate components and returns fewer than
                // n_components columns.
                let phase = (seq_idx as f32) * 0.7 + (time_idx as f32) * 0.31;
                let frequency = 1.0 + dim_idx as f32 * 0.9;
                let value = (phase * frequency).sin() + (dim_idx as f32) * 0.1;
                data.push(value);
            }
        }
    }
    Array3::from_shape_vec((num_sequences, seq_length, obs_dim), data).unwrap()
}

#[test]
fn pca_reduces_to_target_components() {
    let obs = deterministic_obs(8, 6, 5);
    let reduced = pca_fit_transform(&obs, 3).expect("pca");
    assert_eq!(reduced.dim(), (8, 6, 3));
    assert!(reduced.iter().all(|value| value.is_finite()));
}

#[test]
fn pca_rejects_invalid_components() {
    let obs = deterministic_obs(2, 3, 4);
    assert!(pca_fit_transform(&obs, 0).is_err());
    assert!(pca_fit_transform(&obs, 5).is_err());
}

#[test]
fn neural_msm_fit_one_step_finite() {
    let device = NdArrayDevice::default();
    let obs_array = deterministic_obs(4, 5, 2);
    let (raw, _offset) = obs_array.into_raw_vec_and_offset();
    let obs_tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(raw, [4, 5, 2]), &device);

    let msm = NeuralMsmConfig::new(2, 3)
        .with_hidden_dim(4)
        .init::<TestBackend>(&device);
    let (_fitted, history) = msm.fit(obs_tensor, 2, 2, 1e-3);
    assert_eq!(history.len(), 2);
    for (epoch_idx, value) in history.iter().enumerate() {
        assert!(
            value.is_finite(),
            "epoch {epoch_idx} log-likelihood not finite: {value}"
        );
    }
}

#[test]
fn transfer_replaces_warm_start_params() {
    let device = NdArrayDevice::default();
    let obs_array = deterministic_obs(4, 5, 4);
    let reduced = pca_fit_transform(&obs_array, 2).expect("pca");
    let (raw, _offset) = reduced.into_raw_vec_and_offset();
    let reduced_tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(raw, [4, 5, 2]), &device);

    let msm = NeuralMsmConfig::new(2, 3)
        .with_hidden_dim(4)
        .init::<TestBackend>(&device);
    let (fitted, _history) = msm.fit(reduced_tensor, 1, 2, 1e-3);

    let snlds = SnldsConfig::new(4, 2, 8, 3).init::<TestBackend>(&device);
    let warm = transfer_into_snlds(fitted, snlds).expect("transfer");

    let elbo = warm
        .forward(
            Tensor::<TestBackend, 3>::random(
                [2, 5, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            1.0,
            5e-4,
            1.0,
        )
        .elbo
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(elbo.is_finite(), "warm-started ELBO not finite: {elbo}");
}
