use snlds_data::{
    generate_train_test, save_train_test, GenConfig, Manifest, SimulatorKind,
    MANIFEST_SCHEMA_VERSION,
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
    };
    (cfg, manifest)
}

#[test]
fn cli_smoke_writes_rrd() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input_dir = dir.path().join("dataset");
    let rrd_path = dir.path().join("out.rrd");

    let (cfg, manifest) = tiny_cfg();
    let tt = generate_train_test(&cfg);
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
    let tt = generate_train_test(&cfg);
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
