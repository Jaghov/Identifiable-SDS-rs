//! Deterministic generation + SafeTensors round-trip (M1 gates).

use safetensors::SafeTensors;
use snlds_data::io::MANIFEST_SCHEMA_VERSION;
use snlds_data::{
    generate_train_test, load_manifest, load_tensor_f32, load_tensor_i32, save_train_test,
    transitions::get_trans_mat, GenConfig, Manifest, SimulatorKind, TrainTest,
};
use std::fs;
use tempfile::tempdir;

fn cosine_tiny_cfg() -> GenConfig {
    GenConfig {
        seed: 4242,
        num_states: 3,
        dim_obs: 2,
        dim_latent: 2,
        seq_length: 8,
        num_samples: 4,
        sparsity_prob: 0.0,
        kind: SimulatorKind::Cosine,
        poly_degree: 3,
    }
}

fn poly_tiny_cfg() -> GenConfig {
    GenConfig {
        seed: 9191,
        num_states: 3,
        dim_obs: 2,
        dim_latent: 2,
        seq_length: 8,
        num_samples: 4,
        sparsity_prob: 0.0,
        kind: SimulatorKind::Poly,
        poly_degree: 3,
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
    let tt = generate_train_test(&cfg);
    assert_train_test_shapes(&cfg, &tt);
    assert_all_finite_f32(&tt);
    assert_state_range(&tt, cfg.num_states);
}

#[test]
fn generate_train_test_poly_shapes_ranges_and_finite() {
    let cfg = poly_tiny_cfg();
    let tt = generate_train_test(&cfg);
    assert_train_test_shapes(&cfg, &tt);
    assert_all_finite_f32(&tt);
    assert_state_range(&tt, cfg.num_states);
}

#[test]
fn generation_is_deterministic() {
    let cfg = GenConfig {
        seed: 7,
        num_states: 2,
        dim_obs: 2,
        dim_latent: 2,
        seq_length: 5,
        num_samples: 3,
        sparsity_prob: 0.0,
        kind: SimulatorKind::Poly,
        poly_degree: 2,
    };
    let a = generate_train_test(&cfg);
    let b = generate_train_test(&cfg);
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
    let tt = generate_train_test(&cfg);
    assert_train_test_shapes(&cfg, &tt);
    assert_all_finite_f32(&tt);
    assert_state_range(&tt, cfg.num_states);
}

#[test]
fn generation_is_deterministic_cosine_sparse() {
    let cfg = cosine_sparse_tiny_cfg();
    let a = generate_train_test(&cfg);
    let b = generate_train_test(&cfg);
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
    let tt = generate_train_test(&cfg);
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
    };
    save_train_test(dir.path(), &tt, &manifest).unwrap();

    let meta_path = dir.path().join("metadata.json");
    let loaded = load_manifest(&meta_path).unwrap();
    assert_eq!(loaded, manifest);

    let st_path = dir.path().join("sequences.safetensors");
    let bytes = fs::read(&st_path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(st.len(), 6);

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
}

#[test]
fn transition_matrix_three_states() {
    let q = get_trans_mat(3);
    assert_eq!(q.shape(), [3, 3]);
    assert!((q.sum() - 3.0).abs() < 1e-4);
}
