//! Deterministic generation + SafeTensors round-trip (M1 gates).

use ndarray::array;
use safetensors::SafeTensors;
use snlds_data::io::MANIFEST_SCHEMA_VERSION;
use snlds_data::{
    generate_shard, generate_train_test, load_manifest, load_tensor_f32, load_tensor_i32,
    save_train_test, transitions::get_trans_mat, GenConfig, Manifest, ObservationKind,
    SimulatorKind, TrainTest, TransitionPattern,
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
        num_samples_eval: 0,
    };

    save_train_test(dir.path(), &tt, &manifest).unwrap();

    let meta_path = dir.path().join("metadata.json");
    let loaded = load_manifest(&meta_path).unwrap();
    assert_eq!(loaded, manifest);

    let st_path = dir.path().join("sequences.safetensors");
    let bytes = fs::read(&st_path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    // Schema v5: 9 sequence tensors (3 splits × {latents, obs, states}) + q_true + pi_true.
    // `*_eval` tensors are always written; zero-row when eval_fraction == 0.
    assert_eq!(st.len(), 11);

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
        num_samples_eval: 0,
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
        num_samples_eval: 0,
    };
    save_train_test(dir.path(), &tt, &manifest).unwrap();
    let loaded = load_manifest(dir.path().join("metadata.json")).unwrap();
    assert_eq!(loaded.schema_version, MANIFEST_SCHEMA_VERSION);
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

#[test]
fn eval_fraction_default_produces_zero_row_eval_tensors() {
    // With eval_fraction = 0 (the default), the in-memory eval arrays are
    // zero-row and the writer persists them as zero-row tensors (matching
    // the existing test-split convention so the sharded loader can iterate
    // every shard without "tensor not found"). The manifest reports
    // num_samples_eval = 0 so consumers can distinguish "no eval split"
    // from "split exists but lives in another shard".
    let cfg = cosine_tiny_cfg();
    let tt = generate_train_test(&cfg).expect("default cfg has eval_fraction=0");
    assert_eq!(tt.obs_eval.shape()[0], 0);
    assert_eq!(tt.states_eval.shape()[0], 0);
    assert_eq!(tt.latents_eval.shape()[0], 0);

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
        num_samples_eval: 0,
    };
    save_train_test(dir.path(), &tt, &manifest).unwrap();
    let st_path = dir.path().join("sequences.safetensors");
    let bytes = fs::read(&st_path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let tensor_names: Vec<String> = st.names().into_iter().cloned().collect();
    for required in ["latents_eval", "obs_eval", "states_eval"] {
        assert!(
            tensor_names.iter().any(|name| name == required),
            "v5 always writes {required} (zero-row when eval_fraction=0); got {tensor_names:?}",
        );
        let tv = st.tensor(required).unwrap();
        assert_eq!(
            tv.shape()[0],
            0,
            "{required} must be zero-row when eval_fraction=0",
        );
    }

    let loaded = load_manifest(dir.path().join("metadata.json")).unwrap();
    assert_eq!(loaded.num_samples_eval, 0);
}

#[test]
fn eval_fraction_threads_through_to_obs_eval_shape() {
    let cfg = GenConfig {
        seed: 7,
        seq_length: 5,
        num_samples: 10,
        eval_fraction: 0.2,
        ..GenConfig::default()
    };
    let tt = generate_train_test(&cfg).expect("eval_fraction=0.2 is valid");
    // round(10 * 0.2) = 2
    assert_eq!(tt.obs_eval.shape(), &[2, 5, cfg.dim_obs]);
    assert_eq!(tt.states_eval.shape(), &[2, 5]);
    assert_eq!(tt.latents_eval.shape(), &[2, 5, cfg.dim_latent]);
    // train/test untouched
    assert_eq!(tt.obs_train.shape(), &[10, 5, cfg.dim_obs]);
    assert_eq!(tt.obs_test.shape(), &[1, 5, cfg.dim_obs]);
}

#[test]
fn eval_fraction_zero_preserves_train_test_bytes() {
    // Two regens with the same seed and `eval_fraction = 0` produce
    // bit-identical train+test tensors. (The third roll_sequences call for
    // eval still advances the RNG, but only after train+test have been
    // sampled, so train and test bytes are unaffected.)
    let cfg = cosine_tiny_cfg();
    let first = generate_train_test(&cfg).unwrap();
    let second = generate_train_test(&cfg).unwrap();
    assert_eq!(
        first.obs_train.iter().copied().collect::<Vec<_>>(),
        second.obs_train.iter().copied().collect::<Vec<_>>(),
    );
    assert_eq!(
        first.obs_test.iter().copied().collect::<Vec<_>>(),
        second.obs_test.iter().copied().collect::<Vec<_>>(),
    );
    assert_eq!(
        first.states_test.iter().copied().collect::<Vec<_>>(),
        second.states_test.iter().copied().collect::<Vec<_>>(),
    );
}

#[test]
fn eval_fraction_addition_preserves_existing_train_test_bytes() {
    // Adding an eval split to a seed must NOT perturb the bytes drawn for
    // train and test before it. This is the load-bearing invariant: existing
    // checkpoints trained on (train, test) of a given seed still see the
    // same data when the user regenerates with the same seed + an eval
    // fraction added.
    //
    // The guard is structural: `generate_split` rolls train, then test, then
    // eval. The RNG state when train + test are sampled is therefore
    // independent of `eval_fraction`, so the train and test bytes must
    // match for the same seed across any two `eval_fraction` values. This
    // test pins that ordering; a future refactor that moves the eval roll
    // earlier would break it loudly.
    let mut cfg_no_eval = cosine_tiny_cfg();
    cfg_no_eval.eval_fraction = 0.0;
    let mut cfg_with_eval = cosine_tiny_cfg();
    cfg_with_eval.eval_fraction = 0.25;

    let no_eval = generate_train_test(&cfg_no_eval).unwrap();
    let with_eval = generate_train_test(&cfg_with_eval).unwrap();

    assert_eq!(
        no_eval.obs_train.iter().copied().collect::<Vec<_>>(),
        with_eval.obs_train.iter().copied().collect::<Vec<_>>(),
        "train bytes must match across eval_fraction values for the same seed",
    );
    assert_eq!(
        no_eval.obs_test.iter().copied().collect::<Vec<_>>(),
        with_eval.obs_test.iter().copied().collect::<Vec<_>>(),
        "test bytes must match across eval_fraction values for the same seed",
    );
    assert_eq!(
        no_eval.q_true, with_eval.q_true,
        "q_true must match across eval_fraction values (simulator parameters depend on seed alone)",
    );
}

#[test]
fn eval_fraction_rejects_out_of_range() {
    let mut cfg = cosine_tiny_cfg();
    cfg.eval_fraction = 1.5;
    let err = generate_train_test(&cfg)
        .expect_err("eval_fraction=1.5 is out of [0, 1]")
        .to_string();
    assert!(
        err.contains("eval_fraction"),
        "error should mention eval_fraction: {err}"
    );
}

#[test]
fn eval_fraction_rejects_nan() {
    let mut cfg = cosine_tiny_cfg();
    cfg.eval_fraction = f32::NAN;
    let err = generate_train_test(&cfg)
        .expect_err("NaN eval_fraction must fail")
        .to_string();
    assert!(
        err.contains("eval_fraction"),
        "error should mention eval_fraction: {err}"
    );
}

#[test]
fn manifest_v4_loads_as_v5_with_zero_eval() {
    // A literal v4 manifest (no `num_samples_eval` key) on disk must
    // deserialize cleanly under v5 code with `num_samples_eval = 0`.
    let dir = tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let v4_json = r#"{
        "schema_version": 4,
        "seed": 42,
        "num_states": 3,
        "dim_obs": 2,
        "dim_latent": 2,
        "seq_length": 50,
        "num_samples": 100,
        "sparsity_prob": 0.0,
        "data_type": "cosine",
        "init_noise_std": 0.1,
        "init_mean_std": 0.7,
        "transition_step_var": 0.05,
        "emission_hidden_dim": 8
    }"#;
    fs::write(&path, v4_json).unwrap();
    let loaded = load_manifest(&path).expect("v4 manifest loads under v5 code");
    assert_eq!(loaded.schema_version, 4);
    assert_eq!(loaded.num_samples_eval, 0);
}

#[test]
fn eval_round_trip_through_safetensors() {
    // When eval_fraction > 0, the three *_eval tensors are persisted and
    // load back with the expected shapes.
    let cfg = GenConfig {
        seed: 13,
        seq_length: 6,
        num_samples: 10,
        eval_fraction: 0.3,
        ..GenConfig::default()
    };
    let tt = generate_train_test(&cfg).unwrap();
    // round(10 * 0.3) = 3
    assert_eq!(tt.obs_eval.shape()[0], 3);

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
        num_samples_eval: 3,
    };
    save_train_test(dir.path(), &tt, &manifest).unwrap();

    let loaded = load_manifest(dir.path().join("metadata.json")).unwrap();
    assert_eq!(loaded.num_samples_eval, 3);

    let st_path = dir.path().join("sequences.safetensors");
    let obs_eval_loaded = load_tensor_f32(&st_path, "obs_eval").unwrap();
    assert_eq!(
        obs_eval_loaded.len(),
        3 * cfg.seq_length * cfg.dim_obs,
        "obs_eval should round-trip with the expected element count",
    );
    let states_eval_loaded = load_tensor_i32(&st_path, "states_eval").unwrap();
    assert_eq!(states_eval_loaded.len(), 3 * cfg.seq_length);
}

#[test]
fn sharded_dataset_writes_eval_tensors_in_every_shard() {
    // Regression guard for the "sharded eval split silently dropped" bug:
    // shards that do not carry the eval batch (i.e., shard index > 0) must
    // still persist zero-row `*_eval` tensors so the multi-shard loader can
    // iterate every shard without "tensor not found". `num_samples_eval` in
    // each shard's manifest is the per-shard count (non-zero on shard 0 only).
    let num_shards = 3;
    let cfg = GenConfig {
        seed: 11,
        seq_length: 4,
        num_samples: 12,
        eval_fraction: 0.25,
        ..GenConfig::default()
    };
    let expected_eval_total = (cfg.num_samples as f32 * cfg.eval_fraction).round() as usize;
    assert_eq!(expected_eval_total, 3, "sanity: round(12 * 0.25) = 3");

    let root = tempdir().unwrap();
    let mut shard_eval_counts = Vec::with_capacity(num_shards);
    for shard_idx in 0..num_shards {
        let tt = generate_shard(&cfg, shard_idx, num_shards).unwrap();
        let n_eval = tt.obs_eval.shape()[0];
        shard_eval_counts.push(n_eval);
        let shard_dir = root.path().join(format!("shard_{shard_idx:03}"));
        let manifest = Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            seed: cfg.seed,
            num_states: cfg.num_states,
            dim_obs: cfg.dim_obs,
            dim_latent: cfg.dim_latent,
            seq_length: cfg.seq_length,
            num_samples: tt.obs_train.shape()[0],
            sparsity_prob: cfg.sparsity_prob,
            data_type: "cosine".into(),
            degree: None,
            init_noise_std: cfg.init_noise_std,
            init_mean_std: cfg.init_mean_std,
            transition_step_var: cfg.transition_step_var,
            emission_hidden_dim: cfg.emission_hidden_dim,
            num_samples_eval: n_eval,
        };
        save_train_test(&shard_dir, &tt, &manifest).unwrap();
    }
    assert_eq!(
        shard_eval_counts,
        vec![expected_eval_total, 0, 0],
        "all eval samples must land in shard 0",
    );

    // Every shard's safetensors must contain `obs_eval` so the multi-shard
    // loader can iterate them; shards > 0 carry zero-row tensors.
    let mut total_eval_rows = 0usize;
    for shard_idx in 0..num_shards {
        let shard_dir = root.path().join(format!("shard_{shard_idx:03}"));
        let st_path = shard_dir.join("sequences.safetensors");
        let bytes = std::fs::read(&st_path).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let tv = st
            .tensor("obs_eval")
            .unwrap_or_else(|_| panic!("shard {shard_idx} must contain obs_eval (zero-row OK)"));
        total_eval_rows += tv.shape()[0];
        let manifest = load_manifest(shard_dir.join("metadata.json")).unwrap();
        assert_eq!(
            manifest.num_samples_eval,
            tv.shape()[0],
            "manifest.num_samples_eval must mirror obs_eval first axis in shard {shard_idx}",
        );
    }
    assert_eq!(
        total_eval_rows, expected_eval_total,
        "summing eval rows across shards must recover the dataset-wide count",
    );
}
