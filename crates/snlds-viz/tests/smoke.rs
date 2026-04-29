use ndarray::Array2;
use snlds_data::{
    generate_train_test, save_train_test, GenConfig, Manifest, SimulatorKind,
    MANIFEST_SCHEMA_VERSION,
};
use snlds_viz::{
    log_gamma_heatmap, log_posteriors, log_reconstructions, log_state_strip, log_train_scalars,
    log_transition_matrix,
};
use std::process::Command;

fn tiny_cfg() -> (GenConfig, Manifest) {
    let cfg = GenConfig {
        seed: 0,
        num_states: 3,
        dim_obs: 2,
        dim_latent: 2,
        seq_length: 4,
        num_samples: 3,
        sparsity_prob: 0.0,
        kind: SimulatorKind::Cosine,
        poly_degree: 3,
        ..GenConfig::default()
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
        data_type: "cosine".to_string(),
        degree: None,
        init_noise_std: cfg.init_noise_std,
        init_mean_std: cfg.init_mean_std,
        transition_step_var: cfg.transition_step_var,
        emission_hidden_dim: cfg.emission_hidden_dim,
    };
    (cfg, manifest)
}

#[test]
fn log_posteriors_no_panic() {
    let (rec, _storage) = rerun::RecordingStreamBuilder::new("test_posteriors")
        .memory()
        .unwrap();
    // [T=3, K=2] uniform posteriors — each row sums to 1
    let gamma = Array2::from_shape_vec((3, 2), vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5]).unwrap();
    log_posteriors(&rec, 0, gamma.view()).expect("log_posteriors should not fail");
}

#[test]
fn log_train_scalars_no_panic() {
    let (rec, _storage) = rerun::RecordingStreamBuilder::new("test_train_scalars")
        .memory()
        .unwrap();
    log_train_scalars(&rec, 0, -1.5, 0.02, 1.0).expect("log_train_scalars should not fail");
}

#[test]
fn log_reconstructions_no_panic() {
    let (rec, _storage) = rerun::RecordingStreamBuilder::new("test_reconstructions")
        .memory()
        .unwrap();
    // obs_dim == 2: takes the LineStrips2D path
    let x_hat = Array2::from_shape_vec((3, 2), vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    log_reconstructions(&rec, 0, x_hat.view()).expect("log_reconstructions should not fail");
}

#[test]
fn log_transition_matrix_no_panic() {
    let (rec, _storage) = rerun::RecordingStreamBuilder::new("test_q")
        .memory()
        .unwrap();
    let q = Array2::from_shape_vec(
        (3, 3),
        vec![0.7_f32, 0.2, 0.1, 0.1, 0.8, 0.1, 0.0, 0.0, 1.0],
    )
    .unwrap();
    log_transition_matrix(&rec, "snlds/markov/q_true", q.view())
        .expect("log_transition_matrix should not fail");
}

#[test]
fn log_state_strip_no_panic() {
    let (rec, _storage) = rerun::RecordingStreamBuilder::new("test_strip")
        .memory()
        .unwrap();
    let states = vec![0_i32, 1, 1, 2, 0, 2];
    log_state_strip(&rec, "snlds/state/strip_true", &states)
        .expect("log_state_strip should not fail");
}

#[test]
fn log_gamma_heatmap_no_panic() {
    let (rec, _storage) = rerun::RecordingStreamBuilder::new("test_gamma_heatmap")
        .memory()
        .unwrap();
    let gamma = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0_f32, 0.0, 0.0, 0.5, 0.5, 0.0, 0.2, 0.3, 0.5, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    log_gamma_heatmap(&rec, "snlds/state/gamma", gamma.view())
        .expect("log_gamma_heatmap should not fail");
}

#[test]
fn cli_smoke_writes_rrd() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input_dir = dir.path().join("dataset");
    let rrd_path = dir.path().join("out.rrd");

    let (cfg, manifest) = tiny_cfg();
    let tt = generate_train_test(&cfg).expect("generate dataset");
    save_train_test(&input_dir, &tt, &manifest).expect("save dataset");

    let bin = env!("CARGO_BIN_EXE_snlds-viz");
    let status = Command::new(bin)
        .args([
            "--input",
            input_dir.to_str().unwrap(),
            "--sequences",
            "2",
            "--output",
            rrd_path.to_str().unwrap(),
        ])
        .status()
        .expect("run snlds-viz");

    assert!(status.success(), "snlds-viz exited with {status}");
    let meta = std::fs::metadata(&rrd_path).expect("rrd file exists");
    assert!(meta.len() > 0, "rrd file is empty");
}

#[test]
fn cli_render_flag_with_dim_latent_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input_dir = dir.path().join("dataset");
    let rrd_path = dir.path().join("out_render.rrd");

    let (cfg, manifest) = tiny_cfg();
    let tt = generate_train_test(&cfg).expect("generate dataset");
    save_train_test(&input_dir, &tt, &manifest).expect("save dataset");

    let bin = env!("CARGO_BIN_EXE_snlds-viz");
    let status = Command::new(bin)
        .args([
            "--input",
            input_dir.to_str().unwrap(),
            "--sequences",
            "1",
            "--render",
            "--output",
            rrd_path.to_str().unwrap(),
        ])
        .status()
        .expect("run snlds-viz --render");

    assert!(status.success(), "snlds-viz --render exited with {status}");
    let meta = std::fs::metadata(&rrd_path).expect("rrd file exists");
    assert!(meta.len() > 0, "rrd file is empty");
}
