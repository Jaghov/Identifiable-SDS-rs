//! Smoke tests for `snlds-train`.

use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use snlds_data::{
    generate_train_test, save_train_test, GenConfig, Manifest, SimulatorKind,
    MANIFEST_SCHEMA_VERSION,
};
use snlds_train::{
    build_model_config, load_train_obs, load_train_obs_array, run_warm_start, train,
    train_with_model, MsmWarmStartConfig, TrainConfig, TrainSnapshot, TRAIN_SNAPSHOT_FILENAME,
};
use std::process::Command;

type TrainBackend = Autodiff<NdArray<f32>>;

fn tiny_data(dir: &std::path::Path) -> Manifest {
    let cfg = GenConfig {
        seed: 7,
        num_states: 3,
        dim_obs: 4,
        dim_latent: 2,
        seq_length: 5,
        num_samples: 4,
        sparsity_prob: 0.0,
        kind: SimulatorKind::Poly,
        poly_degree: 2,
    };
    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        seed: cfg.seed,
        num_states: cfg.num_states,
        dim_obs: cfg.dim_obs,
        dim_latent: cfg.dim_latent,
        seq_length: cfg.seq_length,
        num_samples: cfg.num_samples,
        sparsity_prob: cfg.sparsity_prob,
        data_type: "poly".into(),
        degree: Some(cfg.poly_degree),
    };
    let train_test = generate_train_test(&cfg);
    save_train_test(dir, &train_test, &manifest).expect("save tiny data");
    manifest
}

#[test]
fn train_one_step_no_panic() {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let output_dir = tempfile::tempdir().expect("output tempdir");
    tiny_data(data_dir.path());

    let device = NdArrayDevice::default();
    let obs_tensor =
        load_train_obs::<TrainBackend>(data_dir.path(), &device).expect("load obs_train");

    let config = TrainConfig {
        data_dir: data_dir.path().into(),
        output_dir: output_dir.path().into(),
        epochs: 1,
        batch_size: 2,
        learning_rate: 3e-4,
        beta: 1.0,
        temperature: 1.0,
        grad_clip: 1.0,
        checkpoint_every: 0,
        hidden_dim: 8,
        obs_noise_var: 5e-4,
        seed: 0,
        resume_from: None,
    };

    let history = train::<TrainBackend>(&config, obs_tensor, &device).expect("train");
    assert_eq!(history.len(), 1);
    assert!(
        history[0].mean_loss.is_finite(),
        "mean loss not finite: {}",
        history[0].mean_loss
    );
}

#[test]
fn snapshot_persisted_next_to_checkpoint() {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let output_dir = tempfile::tempdir().expect("output tempdir");
    tiny_data(data_dir.path());

    let device = NdArrayDevice::default();
    let obs_tensor =
        load_train_obs::<TrainBackend>(data_dir.path(), &device).expect("load obs_train");

    let config = TrainConfig {
        data_dir: data_dir.path().into(),
        output_dir: output_dir.path().into(),
        epochs: 1,
        batch_size: 2,
        learning_rate: 3e-4,
        beta: 0.9,
        temperature: 0.7,
        grad_clip: 1.0,
        checkpoint_every: 1,
        hidden_dim: 12,
        obs_noise_var: 7e-4,
        seed: 0,
        resume_from: None,
    };
    train::<TrainBackend>(&config, obs_tensor, &device).expect("train");

    let snapshot_path = output_dir.path().join(TRAIN_SNAPSHOT_FILENAME);
    assert!(
        snapshot_path.exists(),
        "expected snapshot at {snapshot_path:?}"
    );
    let loaded = TrainSnapshot::load_from_dir(output_dir.path()).expect("load snapshot");
    assert_eq!(loaded.hidden_dim, 12);
    assert!((loaded.beta - 0.9).abs() < 1e-6);
    assert!((loaded.temperature - 0.7).abs() < 1e-6);
    assert!((loaded.obs_noise_var - 7e-4).abs() < 1e-9);

    let checkpoint_path = output_dir.path().join("checkpoint_0000.mpk");
    let from_checkpoint =
        TrainSnapshot::load_for_checkpoint(&checkpoint_path).expect("load via checkpoint path");
    assert_eq!(from_checkpoint, loaded);
}

#[test]
fn checkpoint_round_trip() {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let output_dir = tempfile::tempdir().expect("output tempdir");
    tiny_data(data_dir.path());

    let device = NdArrayDevice::default();
    let obs_tensor =
        load_train_obs::<TrainBackend>(data_dir.path(), &device).expect("load obs_train");

    let config = TrainConfig {
        data_dir: data_dir.path().into(),
        output_dir: output_dir.path().into(),
        epochs: 1,
        batch_size: 2,
        learning_rate: 3e-4,
        beta: 1.0,
        temperature: 1.0,
        grad_clip: 1.0,
        checkpoint_every: 1,
        hidden_dim: 8,
        obs_noise_var: 5e-4,
        seed: 0,
        resume_from: None,
    };
    train::<TrainBackend>(&config, obs_tensor, &device).expect("train");

    let checkpoint_path = output_dir.path().join("checkpoint_0000.mpk");
    assert!(
        checkpoint_path.exists(),
        "expected checkpoint at {:?}",
        checkpoint_path
    );

    let obs_tensor_resume =
        load_train_obs::<TrainBackend>(data_dir.path(), &device).expect("reload obs_train");
    let resume_config = TrainConfig {
        epochs: 1,
        resume_from: Some(checkpoint_path),
        ..config
    };
    train::<TrainBackend>(&resume_config, obs_tensor_resume, &device).expect("resume train");
}

#[test]
fn warm_start_then_train_no_panic() {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let output_dir = tempfile::tempdir().expect("output tempdir");
    tiny_data(data_dir.path());

    let device = NdArrayDevice::default();
    let obs_tensor =
        load_train_obs::<TrainBackend>(data_dir.path(), &device).expect("load obs_train");
    let (obs_array, _manifest) = load_train_obs_array(data_dir.path()).expect("load obs array");

    let config = TrainConfig {
        data_dir: data_dir.path().into(),
        output_dir: output_dir.path().into(),
        epochs: 1,
        batch_size: 2,
        learning_rate: 3e-4,
        beta: 1.0,
        temperature: 1.0,
        grad_clip: 1.0,
        checkpoint_every: 0,
        hidden_dim: 8,
        obs_noise_var: 5e-4,
        seed: 0,
        resume_from: None,
    };
    let snlds_config = build_model_config(&config, &obs_tensor.manifest);

    let warm_config = MsmWarmStartConfig {
        restarts: 1,
        epochs: 1,
        batch_size: 2,
        learning_rate: 1e-3,
        hidden_dim: 4,
    };
    let warm_started =
        run_warm_start::<TrainBackend>(&warm_config, &snlds_config, &obs_array, &device)
            .expect("warm-start");

    let history = train_with_model::<TrainBackend>(&config, warm_started, obs_tensor, &device)
        .expect("train post warm-start");
    assert_eq!(history.len(), 1);
    assert!(history[0].mean_loss.is_finite());
}

#[test]
fn cli_help_exits_zero() {
    let status = Command::new(env!("CARGO_BIN_EXE_snlds-train"))
        .arg("--help")
        .status()
        .expect("spawn snlds-train --help");
    assert!(status.success(), "snlds-train --help should exit 0");
}
