//! `snlds-train` — Adam minibatch training CLI for `VariationalSnlds` and FlowSNLDS (`--flow-snlds`).
//!
//! Optional `--config train.toml` supplies defaults; every other flag overrides the file when set.

use anyhow::Result;
use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use clap::Parser;
use snlds_train::config_file::{
    load_train_config_file, resolve_encoder_kind, resolve_train, EncoderCli, NpcaRotationFile,
    ResolvedMode, TrainCli,
};
use snlds_train::{
    build_model_config, load_train_obs, load_train_obs_array, run_warm_start,
    train_flow_from_dataset, train_with_model, FlowTrainConfig, MsmWarmStartConfig,
    SequenceDataset, TrainConfig,
};

type TrainBackend = Autodiff<LibTorch<f32>>;

fn main() -> Result<()> {
    // Disable TF32 before any CUDA tensors are allocated; flow round-trip invertibility
    // through deep InvConv1x1 stacks is sensitive to the 10-bit-mantissa matmul path.
    glow_flow::disable_tf32();

    let cli = TrainCli::parse();
    let file = cli
        .config
        .as_deref()
        .map(load_train_config_file)
        .transpose()?;
    let r = resolve_train(file.as_ref(), &cli.args)?;

    let device = LibTorchDevice::Cuda(0);

    match r.mode {
        ResolvedMode::FlowSnlds => {
            if cli.args.msm_init {
                anyhow::bail!(
                    "--flow-snlds cannot be used with --msm-init (use flow training only)"
                );
            }
            let res = r.res.ok_or_else(|| {
                anyhow::anyhow!(
                    "flow: set `res` in TOML or pass `--res` (image side, power-of-two ≥ 16)"
                )
            })?;
            if r.encoder != EncoderCli::Gen {
                anyhow::bail!("flow: use `encoder = \"cnn\"` in TOML or `--encoder cnn`");
            }
            snlds_model::validate_cnn_res(res).map_err(|e| anyhow::anyhow!(e))?;
            let flow_cfg = FlowTrainConfig {
                data_dir: r.data_dir.clone(),
                output_dir: r.output_dir.clone(),
                epochs: r.epochs,
                batch_size: r.batch_size,
                learning_rate: r.learning_rate,
                temperature: r.temperature,
                grad_clip: r.grad_clip,
                checkpoint_every: r.checkpoint_every,
                hidden_dim: r.hidden_dim,
                obs_noise_var: r.obs_noise_var,
                seed: r.seed,
                resume_from: r.resume.clone(),
                res,
                w_msm: r.w_msm,
                w_npca: r.w_npca,
                glow_levels: r.npca_glow_levels,
                glow_steps: r.npca_glow_steps,
                glow_hidden_features: r.npca_glow_hidden,
                glow_coupling: r.npca_glow_coupling,
                log_every_batch: r.log_every_batch,
                weight_decay: r.weight_decay,
                transition_log_every_batches: r.transition_log_every_batches,
                npca_householder: matches!(r.npca_rotation, NpcaRotationFile::Householder),
                npca_householder_reflectors: r.npca_householder_reflectors,
            };
            let dataset = SequenceDataset::open(&r.data_dir)?;
            let val_dataset = SequenceDataset::open_val(&r.data_dir)?;
            train_flow_from_dataset::<TrainBackend>(&flow_cfg, dataset, val_dataset, &device)?;
            return Ok(());
        }
        ResolvedMode::Variational => {}
    }

    let obs_tensor = load_train_obs::<TrainBackend>(&r.data_dir, &device)?;

    let kind = resolve_encoder_kind(r.encoder, r.res)?;

    let config = TrainConfig {
        data_dir: r.data_dir.clone(),
        output_dir: r.output_dir,
        epochs: r.epochs,
        batch_size: r.batch_size,
        learning_rate: r.learning_rate,
        beta: r.beta,
        temperature: r.temperature,
        grad_clip: r.grad_clip,
        checkpoint_every: r.checkpoint_every,
        hidden_dim: r.hidden_dim,
        obs_noise_var: r.obs_noise_var,
        seed: r.seed,
        resume_from: r.resume,
        kind,
        log_every_batch: r.log_every_batch,
        transition_log_every_batches: r.transition_log_every_batches,
    };

    let snlds_config = build_model_config(&config, &obs_tensor.manifest);
    let initial_model = if r.msm_init {
        let warm_config = MsmWarmStartConfig {
            restarts: r.msm_restarts,
            epochs: r.msm_epochs,
            batch_size: r.msm_batch_size,
            learning_rate: r.msm_lr,
            hidden_dim: r.msm_hidden_dim,
        };
        let (obs_array, _manifest) = load_train_obs_array(&r.data_dir)?;
        run_warm_start::<TrainBackend>(&warm_config, &snlds_config, &obs_array, &device)?
    } else {
        snlds_config.init::<TrainBackend>(&device)
    };

    train_with_model::<TrainBackend>(&config, initial_model, obs_tensor, &device)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use snlds_model::EncoderKind;
    use snlds_train::config_file::EncoderCli;

    #[test]
    fn resolve_encoder_kind_rejects_non_power_of_two_res() {
        let err = resolve_encoder_kind(EncoderCli::Gen, Some(24)).expect_err("res=24 must fail");
        assert!(
            format!("{err:#}").contains("power of 2"),
            "error should explain validation rule, got: {err:#}"
        );
    }

    #[test]
    fn resolve_encoder_kind_rejects_too_small_res() {
        let err = resolve_encoder_kind(EncoderCli::Gen, Some(8)).expect_err("res=8 must fail");
        assert!(format!("{err:#}").contains(">= 16"));
    }

    #[test]
    fn resolve_encoder_kind_rejects_mlp_with_res() {
        let err = resolve_encoder_kind(EncoderCli::Factored, Some(32)).expect_err("must reject");
        assert!(format!("{err:#}").contains("--res"));
    }

    #[test]
    fn resolve_encoder_kind_rejects_cnn_without_res() {
        let err = resolve_encoder_kind(EncoderCli::Gen, None).expect_err("must reject");
        assert!(format!("{err:#}").contains("--res"));
    }

    #[test]
    fn resolve_encoder_kind_accepts_valid_pairings() {
        assert_eq!(
            resolve_encoder_kind(EncoderCli::Factored, None).unwrap(),
            EncoderKind::Mlp
        );
        assert_eq!(
            resolve_encoder_kind(EncoderCli::Gen, Some(16)).unwrap(),
            EncoderKind::Cnn { res: 16 }
        );
    }
}
