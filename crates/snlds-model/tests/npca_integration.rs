use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};
use glow_flow::prelude::GlowConfig;
use snlds_model::{glow_flattened_latent_dim, glow_last_split_dim, NeuralPcaConfig};

type B = NdArray;

fn test_glow(in_ch: usize, levels: usize) -> GlowConfig {
    GlowConfig::new(in_ch)
        .with_num_levels(levels)
        .with_num_steps(2)
        .with_hidden_features(16)
}

fn test_npca_config(in_ch: usize, levels: usize, h: usize, w: usize) -> NeuralPcaConfig {
    let d = glow_flattened_latent_dim(in_ch, levels, h, w);
    let ld = glow_last_split_dim(in_ch, levels, h, w);
    NeuralPcaConfig::new(test_glow(in_ch, levels), d, ld)
}

#[test]
fn neural_pca_forward_produces_finite_outputs() {
    let device = Default::default();
    let in_ch = 3;
    let levels = 2;
    let h = 8;
    let w = 8;
    let ld = glow_last_split_dim(in_ch, levels, h, w);

    let model = test_npca_config(in_ch, levels, h, w).init::<B>(&device);

    let x = Tensor::<B, 4>::random([4, in_ch, h, w], Distribution::Normal(0.0, 1.0), &device);
    let out = model.forward(x);

    assert_eq!(out.z_pca.dims(), [4, ld]);
    assert!(out
        .z_pca
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(out
        .log_det
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(out.v_matrix.is_some());
}

#[test]
fn neural_pca_freeze_and_eval_forward() {
    let device = Default::default();
    let in_ch = 3;
    let levels = 2;
    let mut model = test_npca_config(in_ch, levels, 8, 8).init::<B>(&device);

    let mut vs = Vec::new();
    let mut bn_means = Vec::new();
    let mut bn_vars = Vec::new();
    for _ in 0..5 {
        let x = Tensor::<B, 4>::random([4, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let out = model.forward(x);
        if let Some(v) = out.v_matrix {
            vs.push(v);
        }
        bn_means.push(out.batch_stats.0);
        bn_vars.push(out.batch_stats.1);
    }
    assert_eq!(vs.len(), 5, "training forwards should yield V each time");
    model.freeze_stats(
        Tensor::stack(vs, 0),
        Tensor::stack(bn_means, 0),
        Tensor::stack(bn_vars, 0),
        &device,
    );

    let x = Tensor::<B, 4>::random([4, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
    let out = model.forward(x);
    assert!(out.v_matrix.is_none());
}

#[test]
fn neural_pca_freeze_sets_bn_and_rotation() {
    let device = Default::default();
    let in_ch = 3;
    let levels = 2;
    let ld = glow_last_split_dim(in_ch, levels, 8, 8);
    let mut model = test_npca_config(in_ch, levels, 8, 8).init::<B>(&device);
    assert!(model.batchnorm.stats.is_none());

    let m = 4usize;
    let mut vs = Vec::new();
    let mut bn_means = Vec::new();
    let mut bn_vars = Vec::new();
    for _ in 0..m {
        let x = Tensor::<B, 4>::random([4, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let out = model.forward(x);
        if let Some(v) = out.v_matrix {
            vs.push(v);
        }
        bn_means.push(out.batch_stats.0);
        bn_vars.push(out.batch_stats.1);
    }

    let stacked_means = Tensor::stack(bn_means.clone(), 0);
    let stacked_vars = Tensor::stack(bn_vars.clone(), 0);
    model.freeze_stats(
        Tensor::stack(vs, 0),
        stacked_means.clone(),
        stacked_vars.clone(),
        &device,
    );

    assert!(model.batchnorm.stats.is_some());
    let exp_mu: Tensor<B, 1> = stacked_means.mean_dim(0).reshape([ld]);
    let exp_var: Tensor<B, 1> = stacked_vars.mean_dim(0).reshape([ld]);
    let (ref mu_p, ref var_p) = model.batchnorm.stats.as_ref().unwrap();
    let max_mu: f32 = (mu_p.val().clone() - exp_mu).abs().max().into_scalar();
    let max_var: f32 = (var_p.val().clone() - exp_var).abs().max().into_scalar();
    assert!(
        max_mu < 1e-5 && max_var < 1e-5,
        "frozen BN should equal mean of per-forward batch stats; mu_err={max_mu} var_err={max_var}"
    );

    let x = Tensor::<B, 4>::random([2, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
    assert!(model.forward(x).v_matrix.is_none());
}

#[test]
fn neural_pca_inverse_round_trip() {
    let device = Default::default();
    let in_ch = 3;
    let levels = 2;
    let mut model = test_npca_config(in_ch, levels, 8, 8).init::<B>(&device);

    let mut vs = Vec::new();
    let mut bn_means = Vec::new();
    let mut bn_vars = Vec::new();
    for _ in 0..10 {
        let x = Tensor::<B, 4>::random([8, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let out = model.forward(x);
        if let Some(v) = out.v_matrix {
            vs.push(v);
        }
        bn_means.push(out.batch_stats.0);
        bn_vars.push(out.batch_stats.1);
    }
    assert_eq!(
        vs.len(),
        10,
        "every training forward should yield a V matrix"
    );
    model.freeze_stats(
        Tensor::stack(vs, 0),
        Tensor::stack(bn_means, 0),
        Tensor::stack(bn_vars, 0),
        &device,
    );

    let x = Tensor::<B, 4>::random([2, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
    let out = model.forward(x.clone());
    let x_recon = model.inverse(out.z_pca, out.z_prefix, &out.latent_shapes, out.batch_stats);
    let max_err: f32 = (x_recon - x).abs().max().into_scalar();
    assert!(max_err < 5.0, "max_err={max_err}");
}

#[test]
fn neural_pca_householder_forward_no_v_matrix() {
    let device = Default::default();
    let in_ch = 3;
    let levels = 2;
    let h = 8;
    let w = 8;
    let ld = glow_last_split_dim(in_ch, levels, h, w);
    let cfg = test_npca_config(in_ch, levels, h, w)
        .with_householder_rotation(true)
        .with_householder_reflectors(8);
    let model = cfg.init::<B>(&device);
    let x = Tensor::<B, 4>::random([2, in_ch, h, w], Distribution::Normal(0.0, 1.0), &device);
    let out = model.forward(x);
    assert!(out.v_matrix.is_none());
    assert_eq!(out.z_pca.dims(), [2, ld]);
}

#[test]
fn neural_pca_householder_inverse_round_trip() {
    let device = Default::default();
    let in_ch = 3;
    let levels = 2;
    let cfg = test_npca_config(in_ch, levels, 8, 8)
        .with_householder_rotation(true)
        .with_householder_reflectors(6);
    let model = cfg.init::<B>(&device);

    let x = Tensor::<B, 4>::random([2, in_ch, 8, 8], Distribution::Normal(0.0, 1.0), &device);
    let out = model.forward(x.clone());
    let x_recon = model.inverse(out.z_pca, out.z_prefix, &out.latent_shapes, out.batch_stats);
    let max_err: f32 = (x_recon - x).abs().max().into_scalar();
    assert!(max_err < 5.0, "max_err={max_err}");
}
