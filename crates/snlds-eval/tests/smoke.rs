//! End-to-end smoke test: tiny dataset → 1 train epoch → snlds-eval.

use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use snlds_data::{
    generate_train_test, save_train_test, GenConfig, Manifest, SimulatorKind,
    MANIFEST_SCHEMA_VERSION,
};
use snlds_eval::{run_eval, EvalConfig};
use snlds_train::{load_train_obs, train, TrainConfig};
use std::process::Command;

type TrainBackend = Autodiff<NdArray<f32>>;
type EvalBackend = NdArray<f32>;

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
fn eval_after_one_epoch_writes_rrd() {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let train_output_dir = tempfile::tempdir().expect("train output tempdir");
    tiny_data(data_dir.path());

    let device = NdArrayDevice::default();
    let obs_tensor = load_train_obs::<TrainBackend>(data_dir.path(), &device)
        .expect("load obs_train (autodiff)");

    let train_config = TrainConfig {
        data_dir: data_dir.path().into(),
        output_dir: train_output_dir.path().into(),
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
    train::<TrainBackend>(&train_config, obs_tensor, &device).expect("train");

    let checkpoint_path = train_output_dir.path().join("checkpoint_0000.mpk");
    assert!(
        checkpoint_path.exists(),
        "expected checkpoint at {:?}",
        checkpoint_path
    );

    let rrd_dir = tempfile::tempdir().expect("rrd tempdir");
    let rrd_path = rrd_dir.path().join("inferred.rrd");
    let eval_config = EvalConfig {
        data_dir: data_dir.path().into(),
        checkpoint: checkpoint_path,
        output: rrd_path.clone(),
        spawn: false,
        sequences: 2,
        hidden_dim_override: None,
        temperature_override: None,
        obs_noise_var_override: None,
        beta_override: None,
    };
    let eval_device = NdArrayDevice::default();
    run_eval::<EvalBackend>(&eval_config, &eval_device).expect("run_eval");

    let meta = std::fs::metadata(&rrd_path).expect("rrd file exists");
    assert!(meta.len() > 0, "rrd file is empty");
}

#[test]
fn cli_help_exits_zero() {
    let status = Command::new(env!("CARGO_BIN_EXE_snlds-eval"))
        .arg("--help")
        .status()
        .expect("spawn snlds-eval --help");
    assert!(status.success(), "snlds-eval --help should exit 0");
}
