//! Deterministic generation + SafeTensors round-trip (M1 gates).

use ndarray::array;
use safetensors::SafeTensors;
use snlds_data::io::MANIFEST_SCHEMA_VERSION;
use snlds_data::{
    generate_train_test, load_manifest, load_tensor_f32, load_tensor_i32, save_train_test,
    transitions::get_trans_mat, GenConfig, Manifest, ObservationKind, SimulatorKind, TrainTest,
    TransitionPattern,
};
use std::fs;
use tempfile::tempdir;

fn cosine_tiny_cfg() -> GenConfig {
    GenConfig {
        seed: 4242,
        seq_length: 8,
        num_samples: 4,
        kind: SimulatorKind::Cosine,
        ..GenConfig::default()
    }
}

fn poly_tiny_cfg() -> GenConfig {
    GenConfig {
        seed: 9191,
        seq_length: 8,
        num_samples: 4,
        kind: SimulatorKind::Poly,
        ..GenConfig::default()
    }
}

fn assert_all_finite_f32(tt: &TrainTest) {
    assert!(tt.latents_train.iter().all(|x| x.is_finite()));
    assert!(tt.latents_test.iter().all(|x| x.is_finite()));
    assert!(tt.obs_train.iter().all(|x| x.is_finite()));
    assert!(tt.obs_test.iter().all(|x| x.is_finite()));
}

fn assert_state_range(tt: &TrainTest, k: usize) {
    let ok = |s: i32| (0..k as i32).contains(&s);
    assert!(tt.states_train.iter().copied().all(ok));
    assert!(tt.states_test.iter().copied().all(ok));
}

fn assert_train_test_shapes(cfg: &GenConfig, tt: &TrainTest) {
    let n_train = cfg.num_samples;
    let n_test = (cfg.num_samples / 10).max(1);
    let t = cfg.seq_length;
    let dl = cfg.dim_latent;
    let d_obs = cfg.dim_obs;

    assert_eq!(tt.latents_train.dim(), (n_train, t, dl));
    assert_eq!(tt.obs_train.dim(), (n_train, t, d_obs));
    assert_eq!(tt.states_train.dim(), (n_train, t));

    assert_eq!(tt.latents_test.dim(), (n_test, t, dl));
    assert_eq!(tt.obs_test.dim(), (n_test, t, d_obs));
    assert_eq!(tt.states_test.dim(), (n_test, t));
}

#[test]
fn generate_train_test_shapes_ranges_and_finite() {
    let cfg = cosine_tiny_cfg();
    let tt = generate_train_test(&cfg).unwrap();
    assert_train_test_shapes(&cfg, &tt);
    assert_all_finite_f32(&tt);
    assert_state_range(&tt, cfg.num_states);
}

#[test]
fn generate_train_test_poly_shapes_ranges_and_finite() {
    let cfg = poly_tiny_cfg();
    let tt = generate_train_test(&cfg).unwrap();
    assert_train_test_shapes(&cfg, &tt);
    assert_all_finite_f32(&tt);
    assert_state_range(&tt, cfg.num_states);
}

#[test]
fn generation_is_deterministic() {
    let cfg = GenConfig {
        seed: 7,
        num_states: 2,
        seq_length: 5,
        num_samples: 3,
        kind: SimulatorKind::Poly,
        poly_degree: 2,
        ..GenConfig::default()
    };
    let a = generate_train_test(&cfg).unwrap();
    let b = generate_train_test(&cfg).unwrap();
    assert_eq!(a.latents_train, b.latents_train);
    assert_eq!(a.obs_train, b.obs_train);
    assert_eq!(a.states_train, b.states_train);
    assert_eq!(a.latents_test, b.latents_test);
    assert_eq!(a.obs_test, b.obs_test);
    assert_eq!(a.states_test, b.states_test);
}

fn cosine_sparse_tiny_cfg() -> GenConfig {
    GenConfig {
        sparsity_prob: 0.5,
        ..cosine_tiny_cfg()
    }
}

#[test]
fn generate_train_test_cosine_sparse_shapes_ranges_and_finite() {
    let cfg = cosine_sparse_tiny_cfg();
    let tt = generate_train_test(&cfg).unwrap();
    assert_train_test_shapes(&cfg, &tt);
    assert_all_finite_f32(&tt);
    assert_state_range(&tt, cfg.num_states);
}

#[test]
fn generation_is_deterministic_cosine_sparse() {
    let cfg = cosine_sparse_tiny_cfg();
    let a = generate_train_test(&cfg).unwrap();
    let b = generate_train_test(&cfg).unwrap();
    assert_eq!(a.latents_train, b.latents_train);
    assert_eq!(a.obs_train, b.obs_train);
    assert_eq!(a.states_train, b.states_train);
    assert_eq!(a.latents_test, b.latents_test);
    assert_eq!(a.obs_test, b.obs_test);
    assert_eq!(a.states_test, b.states_test);
}

#[test]
fn safetensors_roundtrip_all_tensors_and_metadata() {
    let cfg = cosine_tiny_cfg();
    let tt = generate_train_test(&cfg).unwrap();
    let dir = tempdir().unwrap();
    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        seed: cfg.seed,
        num_states: cfg.num_states,
        dim_obs: cfg.dim_obs,
        dim_latent: cfg.dim_latent,
        seq_length: cfg.seq_length,
        num_samples: cfg.num_samples,
        sparsity_prob: cfg.sparsity_prob,
        data_type: "cosine".into(),
        degree: None,
        init_noise_std: cfg.init_noise_std,
        init_mean_std: cfg.init_mean_std,
        transition_step_var: cfg.transition_step_var,
        emission_hidden_dim: cfg.emission_hidden_dim,
    };

    save_train_test(dir.path(), &tt, &manifest).unwrap();

    let meta_path = dir.path().join("metadata.json");
    let loaded = load_manifest(&meta_path).unwrap();
    assert_eq!(loaded, manifest);

    let st_path = dir.path().join("sequences.safetensors");
    let bytes = fs::read(&st_path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    // Schema v3: 6 sequence tensors + q_true + pi_true.
    assert_eq!(st.len(), 8);

    let assert_f32_eq = |name: &str, got: &[f32]| {
        let loaded = load_tensor_f32(&st_path, name).unwrap();
        assert_eq!(loaded, got);
    };
    assert_f32_eq(
        "latents_train",
        &tt.latents_train.iter().copied().collect::<Vec<_>>(),
    );
    assert_f32_eq(
        "latents_test",
        &tt.latents_test.iter().copied().collect::<Vec<_>>(),
    );
    assert_f32_eq(
        "obs_train",
        &tt.obs_train.iter().copied().collect::<Vec<_>>(),
    );
    assert_f32_eq("obs_test", &tt.obs_test.iter().copied().collect::<Vec<_>>());

    assert_eq!(
        load_tensor_i32(&st_path, "states_train").unwrap(),
        tt.states_train.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        load_tensor_i32(&st_path, "states_test").unwrap(),
        tt.states_test.iter().copied().collect::<Vec<_>>()
    );
    assert_f32_eq("q_true", &tt.q_true.iter().copied().collect::<Vec<_>>());
    assert_f32_eq("pi_true", &tt.pi_true.iter().copied().collect::<Vec<_>>());
}

#[test]
fn safetensors_persists_q_and_pi_true() {
    let cfg = cosine_tiny_cfg();
    let tt = generate_train_test(&cfg).unwrap();
    let dir = tempdir().unwrap();
    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        seed: cfg.seed,
        num_states: cfg.num_states,
        dim_obs: cfg.dim_obs,
        dim_latent: cfg.dim_latent,
        seq_length: cfg.seq_length,
        num_samples: cfg.num_samples,
        sparsity_prob: cfg.sparsity_prob,
        data_type: "cosine".into(),
        degree: None,
        init_noise_std: cfg.init_noise_std,
        init_mean_std: cfg.init_mean_std,
        transition_step_var: cfg.transition_step_var,
        emission_hidden_dim: cfg.emission_hidden_dim,
    };
    save_train_test(dir.path(), &tt, &manifest).unwrap();
    let st_path = dir.path().join("sequences.safetensors");

    // q_true matches the deterministic cyclic constructor.
    let q_loaded = load_tensor_f32(&st_path, "q_true").unwrap();
    let q_expected: Vec<f32> = get_trans_mat(&cfg.transition, cfg.num_states)
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(q_loaded, q_expected);

    // pi_true is uniform 1/K.
    let pi_loaded = load_tensor_f32(&st_path, "pi_true").unwrap();
    let expected_value = 1.0_f32 / cfg.num_states as f32;
    assert!(pi_loaded
        .iter()
        .all(|value| (value - expected_value).abs() < 1e-6));
    assert_eq!(pi_loaded.len(), cfg.num_states);
}

#[test]
fn transition_matrix_three_states() {
    let q = get_trans_mat(&TransitionPattern::default(), 3).unwrap();
    assert_eq!(q.shape(), [3, 3]);
    assert!((q.sum() - 3.0).abs() < 1e-6);
}

#[test]
fn cyclic_self_prob_threads_through_to_q_true() {
    let cfg = GenConfig {
        transition: TransitionPattern::Cyclic { self_prob: 0.75 },
        ..cosine_tiny_cfg()
    };
    let tt = generate_train_test(&cfg).unwrap();
    let expected = get_trans_mat(&cfg.transition, cfg.num_states).unwrap();
    assert_eq!(tt.q_true, expected);
}

#[test]
fn provided_transition_matrix_threads_through_to_q_true() {
    let q = array![[0.4f32, 0.6], [0.2f32, 0.8]];
    let cfg = GenConfig {
        num_states: 2,
        dim_obs: 2,
        dim_latent: 2,
        transition: TransitionPattern::Provided(q.clone()),
        ..cosine_tiny_cfg()
    };
    let tt = generate_train_test(&cfg).unwrap();
    assert_eq!(tt.q_true, q);
}

#[test]
fn single_state_cyclic_q_true_is_identity() {
    let cfg = GenConfig {
        num_states: 1,
        dim_obs: 2,
        dim_latent: 2,
        seq_length: 4,
        num_samples: 8,
        transition: TransitionPattern::Cyclic { self_prob: 0.9 },
        ..GenConfig::default()
    };
    let tt = generate_train_test(&cfg).unwrap();
    assert_eq!(tt.q_true.shape(), [1, 1]);
    assert!((tt.q_true[[0, 0]] - 1.0).abs() < 1e-6);
    assert_state_range(&tt, 1);
}

#[test]
fn num_states_zero_errors_before_rollout() {
    let cfg = GenConfig {
        num_states: 0,
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(err.contains("num_states"), "{err}");
}

#[test]
fn manifest_v4_simulator_hparams_round_trip() {
    let cfg = cosine_tiny_cfg();
    let tt = generate_train_test(&cfg).unwrap();
    let dir = tempdir().unwrap();
    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        seed: cfg.seed,
        num_states: cfg.num_states,
        dim_obs: cfg.dim_obs,
        dim_latent: cfg.dim_latent,
        seq_length: cfg.seq_length,
        num_samples: cfg.num_samples,
        sparsity_prob: cfg.sparsity_prob,
        data_type: "cosine".into(),
        degree: None,
        init_noise_std: 0.123,
        init_mean_std: 0.456,
        transition_step_var: 0.0789,
        emission_hidden_dim: 16,
    };
    save_train_test(dir.path(), &tt, &manifest).unwrap();
    let loaded = load_manifest(dir.path().join("metadata.json")).unwrap();
    assert_eq!(loaded.schema_version, 4);
    assert!((loaded.init_noise_std - 0.123).abs() < 1e-7);
    assert!((loaded.init_mean_std - 0.456).abs() < 1e-7);
    assert!((loaded.transition_step_var - 0.0789).abs() < 1e-7);
    assert_eq!(loaded.emission_hidden_dim, 16);
    assert_eq!(loaded, manifest);
}

#[test]
fn manifest_v3_legacy_json_loads_with_default_simulator_hparams() {
    // Literal v3 manifest (schema_version: 3, none of the v4 fields present).
    let v3_json = r#"{
        "schema_version": 3,
        "seed": 4242,
        "num_states": 3,
        "dim_obs": 2,
        "dim_latent": 2,
        "seq_length": 8,
        "num_samples": 4,
        "sparsity_prob": 0.0,
        "data_type": "cosine"
    }"#;
    let m: Manifest = serde_json::from_str(v3_json).expect("v3 JSON must still parse");
    assert_eq!(m.schema_version, 3);
    assert!((m.init_noise_std - 0.1).abs() < 1e-7);
    assert!((m.init_mean_std - 0.7).abs() < 1e-7);
    assert!((m.transition_step_var - 0.05).abs() < 1e-7);
    assert_eq!(m.emission_hidden_dim, 8);
}

#[test]
fn non_uniform_initial_distribution_drives_first_state_frequencies() {
    let cfg = GenConfig {
        seed: 1234,
        seq_length: 2,
        num_samples: 200,
        initial_distribution: Some(vec![0.9, 0.05, 0.05]),
        ..GenConfig::default()
    };
    let tt = generate_train_test(&cfg).unwrap();
    let n = tt.states_train.shape()[0];
    let mut counts = [0u32; 3];
    for ni in 0..n {
        counts[tt.states_train[[ni, 0]] as usize] += 1;
    }
    let freqs = [
        counts[0] as f32 / n as f32,
        counts[1] as f32 / n as f32,
        counts[2] as f32 / n as f32,
    ];
    assert!(
        (freqs[0] - 0.9).abs() < 0.05,
        "expected ~0.9 in state 0, got {}",
        freqs[0]
    );
    // p=0.05 → σ ≈ 0.015 over n=200; <0.1 is ~3σ slack and still rules out uniform (~0.33).
    assert!(freqs[1] < 0.1, "state 1 freq too high: {}", freqs[1]);
    assert!(freqs[2] < 0.1, "state 2 freq too high: {}", freqs[2]);

    // pi_true persisted should match the supplied distribution exactly.
    assert_eq!(tt.pi_true.as_slice().unwrap(), &[0.9, 0.05, 0.05]);
}

#[test]
fn initial_distribution_length_mismatch_errors() {
    let cfg = GenConfig {
        num_samples: 2,
        seq_length: 2,
        initial_distribution: Some(vec![0.5, 0.5]),
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(err.contains("length"), "error should mention length: {err}");
}

#[test]
fn initial_distribution_does_not_sum_to_one_errors() {
    let cfg = GenConfig {
        num_samples: 2,
        seq_length: 2,
        initial_distribution: Some(vec![0.5, 0.5, 0.5]),
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(err.contains("sum"), "error should mention sum: {err}");
}

#[test]
fn initial_distribution_negative_entry_errors() {
    let cfg = GenConfig {
        num_samples: 2,
        seq_length: 2,
        // length 3, sums to 1.0 → must hit the "finite and non-negative" arm, not length/sum.
        initial_distribution: Some(vec![-0.1, 0.55, 0.55]),
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(
        err.contains("finite") && err.contains("non-negative"),
        "error should mention finite + non-negative: {err}"
    );
}

#[test]
fn initial_distribution_nan_entry_errors() {
    let cfg = GenConfig {
        num_samples: 2,
        seq_length: 2,
        // NaN fails `is_finite`; the other entries are valid so length and sum arms can't fire.
        initial_distribution: Some(vec![f32::NAN, 0.5, 0.5]),
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(
        err.contains("finite") && err.contains("non-negative"),
        "error should mention finite + non-negative: {err}"
    );
}

#[test]
fn image_observation_shapes_and_pixel_range() {
    let res = 16usize;
    let cfg = GenConfig {
        seed: 7,
        num_samples: 2,
        seq_length: 4,
        dim_latent: 2,
        dim_obs: res * res * 3,
        observation: ObservationKind::Image { res },
        ..GenConfig::default()
    };
    let tt = generate_train_test(&cfg).expect("image gen succeeds");
    let n_train = cfg.num_samples;
    let n_test = (cfg.num_samples / 10).max(1);
    assert_eq!(
        tt.obs_train.shape(),
        &[n_train, cfg.seq_length, res * res * 3]
    );
    assert_eq!(
        tt.obs_test.shape(),
        &[n_test, cfg.seq_length, res * res * 3]
    );
    for pixel in tt.obs_train.iter().chain(tt.obs_test.iter()) {
        assert!(
            pixel.is_finite() && (0.0..=1.0).contains(pixel),
            "pixel {pixel} out of [0,1] or non-finite"
        );
    }
}

#[test]
fn image_observation_rejects_wrong_obs_dim() {
    let res = 16usize;
    let cfg = GenConfig {
        seed: 7,
        num_samples: 1,
        seq_length: 2,
        dim_latent: 2,
        dim_obs: res * res * 3 + 1, // off by one
        observation: ObservationKind::Image { res },
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(
        err.contains("dim_obs") && err.contains(&format!("{}", res * res * 3)),
        "error should reference required dim_obs: {err}"
    );
}

#[test]
fn image_observation_rejects_wrong_dim_latent() {
    let res = 16usize;
    let cfg = GenConfig {
        seed: 7,
        num_samples: 1,
        seq_length: 2,
        dim_latent: 3, // must be 2 for image rendering
        dim_obs: res * res * 3,
        observation: ObservationKind::Image { res },
        ..GenConfig::default()
    };
    let err = generate_train_test(&cfg).unwrap_err().to_string();
    assert!(
        err.contains("dim_latent"),
        "error should mention dim_latent: {err}"
    );
}
