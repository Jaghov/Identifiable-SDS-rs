//! Optional TOML defaults for `snlds-train` CLI values; CLI flags override when set.
//!
//! Load with [`load_train_config_file`], then [`resolve_train`] merges file + parsed args.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use snlds_model::{validate_cnn_res, CouplingType, EncoderKind};
use std::path::{Path, PathBuf};

use crate::DEFAULT_OBS_NOISE_VAR;

/// Top-level `[..]` TOML schema (flat table). Unknown keys are rejected.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfigFile {
    pub data_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub mode: TrainModeFile,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    #[serde(rename = "lr")]
    pub learning_rate: Option<f64>,
    pub beta: Option<f32>,
    pub temperature: Option<f32>,
    pub grad_clip: Option<f32>,
    pub checkpoint_every: Option<usize>,
    /// Minibatch log cadence: print every N batches per epoch (`0` = off; default 1).
    pub log_every_batch: Option<usize>,
    /// Variational / Flow: print learned `Q` every N minibatches (`0` = epoch end only).
    pub transition_log_every_batches: Option<usize>,
    /// AdamW decoupled weight decay for FlowSNLDS and Neural PCA (ignored for variational SNLDS, which uses Adam).
    pub weight_decay: Option<f32>,
    pub hidden_dim: Option<usize>,
    pub obs_noise_var: Option<f32>,
    pub seed: Option<u64>,
    pub resume: Option<PathBuf>,
    #[serde(default)]
    pub msm_init: bool,
    pub msm_restarts: Option<usize>,
    pub msm_epochs: Option<usize>,
    pub msm_batch_size: Option<usize>,
    pub msm_lr: Option<f64>,
    pub msm_hidden_dim: Option<usize>,
    #[serde(default)]
    pub encoder: EncoderFile,
    pub res: Option<usize>,
    pub w_msm: Option<f32>,
    pub w_npca: Option<f32>,
    pub npca_glow_levels: Option<usize>,
    pub npca_glow_steps: Option<usize>,
    pub npca_glow_hidden: Option<usize>,
    pub npca_glow_coupling: Option<GlowCouplingFile>,
    /// Neural PCA post-BN rotation: `svd` (default, Li & Hooi) or `householder`.
    #[serde(default)]
    pub npca_rotation: NpcaRotationFile,
    /// Number of Householder reflectors when `npca_rotation = "householder"`. Defaults to 32 if unset.
    pub npca_householder_reflectors: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainModeFile {
    #[default]
    Variational,
    FlowSnlds,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EncoderFile {
    #[default]
    Mlp,
    Factored,
    Cnn,
    Gen,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GlowCouplingFile {
    Affine,
    Additive,
}

impl From<GlowCouplingFile> for CouplingType {
    fn from(v: GlowCouplingFile) -> Self {
        match v {
            GlowCouplingFile::Affine => CouplingType::Affine,
            GlowCouplingFile::Additive => CouplingType::Additive,
        }
    }
}

/// [`TrainConfigFile::npca_rotation`] / [`ResolvedTrain::npca_rotation`].
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NpcaRotationFile {
    #[default]
    Svd,
    Householder,
}

/// One combined `clap` struct (`--config` + all training flags as optional overrides).
#[derive(Parser, Debug)]
#[command(
    name = "snlds-train",
    about = "Train VariationalSnlds, FlowSNLDS, or Neural PCA on M1 SafeTensors."
)]
pub struct TrainCli {
    /// TOML file with defaults. Every CLI flag below overrides the file when you pass it.
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
    #[command(flatten)]
    pub args: TrainArgs,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EncoderCli {
    Factored,
    Gen,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum GlowCouplingCli {
    #[default]
    Affine,
    Additive,
}

impl From<GlowCouplingCli> for CouplingType {
    fn from(c: GlowCouplingCli) -> Self {
        match c {
            GlowCouplingCli::Affine => CouplingType::Affine,
            GlowCouplingCli::Additive => CouplingType::Additive,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum NpcaRotationCli {
    #[default]
    Svd,
    Householder,
}

#[derive(Parser, Debug)]
pub struct TrainArgs {
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    #[arg(long)]
    pub epochs: Option<usize>,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long = "lr")]
    pub learning_rate: Option<f64>,
    #[arg(long)]
    pub beta: Option<f32>,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub grad_clip: Option<f32>,
    #[arg(long)]
    pub checkpoint_every: Option<usize>,
    /// Log every N minibatches per epoch (0 = off; default 1).
    #[arg(long = "log-every")]
    pub log_every_batch: Option<usize>,
    /// Print learned Markov Q every N minibatches for variational / Flow (`0` = epoch end only).
    #[arg(long = "transition-log-every")]
    pub transition_log_every_batches: Option<usize>,
    /// AdamW weight decay for FlowSNLDS / Neural PCA (0 = off; default 1e-4).
    #[arg(long)]
    pub weight_decay: Option<f32>,
    #[arg(long)]
    pub hidden_dim: Option<usize>,
    #[arg(long)]
    pub obs_noise_var: Option<f32>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub resume: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub msm_init: bool,
    #[arg(long)]
    pub msm_restarts: Option<usize>,
    #[arg(long)]
    pub msm_epochs: Option<usize>,
    #[arg(long)]
    pub msm_batch_size: Option<usize>,
    #[arg(long)]
    pub msm_lr: Option<f64>,
    #[arg(long)]
    pub msm_hidden_dim: Option<usize>,

    #[arg(long, value_enum)]
    pub encoder: Option<EncoderCli>,
    #[arg(long)]
    pub res: Option<usize>,

    #[arg(long, default_value_t = false)]
    pub flow_snlds: bool,
    #[arg(long)]
    pub w_msm: Option<f32>,
    #[arg(long)]
    pub w_npca: Option<f32>,
    #[arg(long)]
    pub npca_glow_levels: Option<usize>,
    #[arg(long)]
    pub npca_glow_steps: Option<usize>,
    #[arg(long)]
    pub npca_glow_hidden: Option<usize>,
    #[arg(long, value_enum)]
    pub npca_glow_coupling: Option<GlowCouplingCli>,

    #[arg(long, value_enum)]
    pub npca_rotation: Option<NpcaRotationCli>,

    #[arg(long)]
    pub npca_householder_reflectors: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum ResolvedMode {
    Variational,
    FlowSnlds,
}

#[derive(Debug, Clone)]
pub struct ResolvedTrain {
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub mode: ResolvedMode,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub beta: f32,
    pub temperature: f32,
    pub grad_clip: f32,
    pub checkpoint_every: usize,
    pub log_every_batch: usize,
    pub weight_decay: f32,
    pub transition_log_every_batches: usize,
    pub hidden_dim: usize,
    pub obs_noise_var: f32,
    pub seed: u64,
    pub resume: Option<PathBuf>,
    pub msm_init: bool,
    pub msm_restarts: usize,
    pub msm_epochs: usize,
    pub msm_batch_size: usize,
    pub msm_lr: f64,
    pub msm_hidden_dim: usize,
    pub encoder: EncoderCli,
    pub res: Option<usize>,
    pub w_msm: f32,
    pub w_npca: f32,
    pub npca_glow_levels: usize,
    pub npca_glow_steps: usize,
    pub npca_glow_hidden: usize,
    pub npca_glow_coupling: CouplingType,
    pub npca_rotation: NpcaRotationFile,
    pub npca_householder_reflectors: usize,
}

fn parse_train_config_toml(raw: &str) -> Result<TrainConfigFile> {
    toml::from_str(raw).context("parse training config TOML")
}

pub fn load_train_config_file(path: &Path) -> Result<TrainConfigFile> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read training config TOML {:?}", path))?;
    parse_train_config_toml(&raw).with_context(|| format!("in {:?}", path))
}

fn file_encoder_to_cli(e: &EncoderFile) -> EncoderCli {
    match e {
        EncoderFile::Mlp | EncoderFile::Factored => EncoderCli::Factored,
        EncoderFile::Cnn | EncoderFile::Gen => EncoderCli::Gen,
    }
}

fn pick_opt<T: Clone>(cli: Option<T>, file: Option<T>, hard: T) -> T {
    cli.or(file).unwrap_or(hard)
}

/// Merge optional TOML layer with CLI. CLI `Option::Some` wins; otherwise use file; otherwise `hard_*`.
pub fn resolve_train(file: Option<&TrainConfigFile>, args: &TrainArgs) -> Result<ResolvedTrain> {
    let data_dir = args
        .data_dir
        .clone()
        .or_else(|| file.and_then(|f| f.data_dir.clone()))
        .ok_or_else(|| {
            anyhow::anyhow!("data_dir: set `data_dir` in the TOML or pass `--data-dir`")
        })?;
    let output_dir = args
        .output_dir
        .clone()
        .or_else(|| file.and_then(|f| f.output_dir.clone()))
        .ok_or_else(|| {
            anyhow::anyhow!("output_dir: set `output_dir` in the TOML or pass `--output-dir`")
        })?;

    let encoder = args
        .encoder
        .or_else(|| file.map(|f| file_encoder_to_cli(&f.encoder)))
        .unwrap_or(EncoderCli::Factored);

    let mut want_flow = args.flow_snlds;
    if !want_flow {
        if let Some(f) = file {
            match f.mode {
                TrainModeFile::FlowSnlds => want_flow = true,
                TrainModeFile::Variational => {}
            }
        }
    }
    let mode = if want_flow {
        ResolvedMode::FlowSnlds
    } else {
        ResolvedMode::Variational
    };

    let f_epochs = file.and_then(|x| x.epochs);
    let f_batch = file.and_then(|x| x.batch_size);
    let f_lr = file.and_then(|x| x.learning_rate);
    let f_beta = file.and_then(|x| x.beta);
    let f_temp = file.and_then(|x| x.temperature);
    let f_gc = file.and_then(|x| x.grad_clip);
    let f_ckpt = file.and_then(|x| x.checkpoint_every);
    let f_log_batch = file.and_then(|x| x.log_every_batch);
    let f_transition_log = file.and_then(|x| x.transition_log_every_batches);
    let f_weight_decay = file.and_then(|x| x.weight_decay);
    let f_hidden = file.and_then(|x| x.hidden_dim);
    let f_obs = file.and_then(|x| x.obs_noise_var);
    let f_seed = file.and_then(|x| x.seed);
    let f_res = file.and_then(|x| x.res);
    let f_msm_r = file.and_then(|x| x.msm_restarts);
    let f_msm_e = file.and_then(|x| x.msm_epochs);
    let f_msm_bs = file.and_then(|x| x.msm_batch_size);
    let f_msm_lr = file.and_then(|x| x.msm_lr);
    let f_msm_h = file.and_then(|x| x.msm_hidden_dim);

    let msm_init = args.msm_init || file.map(|f| f.msm_init).unwrap_or(false);

    let (f_w_msm, f_w_npca, f_gl, f_gs, f_gh, f_gcpl, f_npca_rot, f_npca_hh_k) =
        if let Some(f) = file {
            (
                f.w_msm,
                f.w_npca,
                f.npca_glow_levels,
                f.npca_glow_steps,
                f.npca_glow_hidden,
                f.npca_glow_coupling,
                Some(f.npca_rotation),
                f.npca_householder_reflectors,
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

    let npca_rotation = match args.npca_rotation {
        Some(NpcaRotationCli::Svd) => NpcaRotationFile::Svd,
        Some(NpcaRotationCli::Householder) => NpcaRotationFile::Householder,
        None => f_npca_rot.unwrap_or_default(),
    };
    let npca_householder_reflectors =
        pick_opt(args.npca_householder_reflectors, f_npca_hh_k, 32usize);

    let npca_glow_coupling = args
        .npca_glow_coupling
        .map(Into::into)
        .or_else(|| f_gcpl.map(Into::into))
        .unwrap_or(CouplingType::Affine);

    Ok(ResolvedTrain {
        data_dir,
        output_dir,
        mode,
        epochs: pick_opt(args.epochs, f_epochs, 100),
        batch_size: pick_opt(args.batch_size, f_batch, 32),
        learning_rate: pick_opt(args.learning_rate, f_lr, 3e-4),
        beta: pick_opt(args.beta, f_beta, 1.0),
        temperature: pick_opt(args.temperature, f_temp, 1.0),
        grad_clip: pick_opt(args.grad_clip, f_gc, 1.0),
        checkpoint_every: pick_opt(args.checkpoint_every, f_ckpt, 10),
        log_every_batch: pick_opt(args.log_every_batch, f_log_batch, 1),
        weight_decay: pick_opt(args.weight_decay, f_weight_decay, 1e-4),
        transition_log_every_batches: pick_opt(
            args.transition_log_every_batches,
            f_transition_log,
            0,
        ),
        hidden_dim: pick_opt(args.hidden_dim, f_hidden, 64),
        obs_noise_var: pick_opt(args.obs_noise_var, f_obs, DEFAULT_OBS_NOISE_VAR),
        seed: pick_opt(args.seed, f_seed, 0),
        resume: args
            .resume
            .clone()
            .or_else(|| file.and_then(|f| f.resume.clone())),
        msm_init,
        msm_restarts: pick_opt(args.msm_restarts, f_msm_r, 3),
        msm_epochs: pick_opt(args.msm_epochs, f_msm_e, 30),
        msm_batch_size: pick_opt(args.msm_batch_size, f_msm_bs, 32),
        msm_lr: pick_opt(args.msm_lr, f_msm_lr, 7e-3),
        msm_hidden_dim: pick_opt(args.msm_hidden_dim, f_msm_h, 16),
        encoder,
        res: args.res.or(f_res),
        w_msm: pick_opt(args.w_msm, f_w_msm, 3.0),
        w_npca: pick_opt(args.w_npca, f_w_npca, 1.0),
        npca_glow_levels: pick_opt(args.npca_glow_levels, f_gl, 2),
        npca_glow_steps: pick_opt(args.npca_glow_steps, f_gs, 2),
        npca_glow_hidden: pick_opt(args.npca_glow_hidden, f_gh, 16),
        npca_glow_coupling,
        npca_rotation,
        npca_householder_reflectors,
    })
}

pub fn resolve_encoder_kind(encoder: EncoderCli, res: Option<usize>) -> Result<EncoderKind> {
    match (encoder, res) {
        (EncoderCli::Factored, None) => Ok(EncoderKind::Mlp),
        (EncoderCli::Factored, Some(res)) => {
            anyhow::bail!("--res {res} given with --encoder mlp; --res is only valid for cnn")
        }
        (EncoderCli::Gen, Some(res)) => {
            validate_cnn_res(res).map_err(|err| anyhow::anyhow!(err))?;
            Ok(EncoderKind::Cnn { res })
        }
        (EncoderCli::Gen, None) => {
            anyhow::bail!("--encoder cnn requires --res <usize> (e.g. 16, 32)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_overridden_by_cli_numbers() {
        let file = TrainConfigFile {
            epochs: Some(10),
            batch_size: Some(7),
            learning_rate: Some(1e-2),
            ..minimal_file()
        };
        let args = TrainArgs {
            data_dir: Some("x".into()),
            output_dir: Some("y".into()),
            epochs: Some(99),
            batch_size: None,
            ..minimal_args()
        };
        let r = resolve_train(Some(&file), &args).expect("resolve");
        assert_eq!(r.epochs, 99);
        assert_eq!(r.batch_size, 7);
        assert!((r.learning_rate - 1e-2).abs() < 1e-9);
    }

    #[test]
    fn weight_decay_from_toml() {
        let file = TrainConfigFile {
            weight_decay: Some(0.02),
            ..minimal_file()
        };
        let args = TrainArgs {
            data_dir: Some("d".into()),
            output_dir: Some("o".into()),
            ..minimal_args()
        };
        let r = resolve_train(Some(&file), &args).expect("resolve");
        assert!((r.weight_decay - 0.02).abs() < 1e-6);

        let args_cli = TrainArgs {
            data_dir: Some("d".into()),
            output_dir: Some("o".into()),
            weight_decay: Some(0.0),
            ..minimal_args()
        };
        let r2 = resolve_train(Some(&file), &args_cli).expect("resolve");
        assert_eq!(r2.weight_decay, 0.0);
    }

    #[test]
    fn transition_log_every_batches_from_toml() {
        let file = TrainConfigFile {
            transition_log_every_batches: Some(50),
            ..minimal_file()
        };
        let args = TrainArgs {
            data_dir: Some("d".into()),
            output_dir: Some("o".into()),
            ..minimal_args()
        };
        let r = resolve_train(Some(&file), &args).expect("resolve");
        assert_eq!(r.transition_log_every_batches, 50);
    }

    #[test]
    fn transition_log_every_batches_cli_overrides_toml() {
        let file = TrainConfigFile {
            transition_log_every_batches: Some(50),
            ..minimal_file()
        };
        let args = TrainArgs {
            data_dir: Some("d".into()),
            output_dir: Some("o".into()),
            transition_log_every_batches: Some(3),
            ..minimal_args()
        };
        let r = resolve_train(Some(&file), &args).expect("resolve");
        assert_eq!(r.transition_log_every_batches, 3);
    }

    #[test]
    fn example_toml_parses() {
        let raw = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/train.example.toml"));
        let f = parse_train_config_toml(raw).expect("parse train.example.toml");
        assert!(f.data_dir.is_some());
        assert_eq!(f.epochs, Some(100));
    }

    #[test]
    fn paths_required_without_toml() {
        let args = minimal_args();
        let err = resolve_train(None, &args).unwrap_err().to_string();
        assert!(err.contains("data_dir"), "{err}");
    }

    fn minimal_file() -> TrainConfigFile {
        TrainConfigFile {
            data_dir: Some("d".into()),
            output_dir: Some("o".into()),
            mode: TrainModeFile::Variational,
            epochs: None,
            batch_size: None,
            learning_rate: None,
            beta: None,
            temperature: None,
            grad_clip: None,
            checkpoint_every: None,
            log_every_batch: None,
            transition_log_every_batches: None,
            weight_decay: None,
            hidden_dim: None,
            obs_noise_var: None,
            seed: None,
            resume: None,
            msm_init: false,
            msm_restarts: None,
            msm_epochs: None,
            msm_batch_size: None,
            msm_lr: None,
            msm_hidden_dim: None,
            encoder: EncoderFile::Mlp,
            res: None,
            w_msm: None,
            w_npca: None,
            npca_glow_levels: None,
            npca_glow_steps: None,
            npca_glow_hidden: None,
            npca_glow_coupling: None,
            npca_rotation: NpcaRotationFile::Svd,
            npca_householder_reflectors: None,
        }
    }

    fn minimal_args() -> TrainArgs {
        TrainArgs {
            data_dir: None,
            output_dir: None,
            epochs: None,
            batch_size: None,
            learning_rate: None,
            beta: None,
            temperature: None,
            grad_clip: None,
            checkpoint_every: None,
            log_every_batch: None,
            transition_log_every_batches: None,
            weight_decay: None,
            hidden_dim: None,
            obs_noise_var: None,
            seed: None,
            resume: None,
            msm_init: false,
            msm_restarts: None,
            msm_epochs: None,
            msm_batch_size: None,
            msm_lr: None,
            msm_hidden_dim: None,
            encoder: None,
            res: None,
            flow_snlds: false,
            w_msm: None,
            w_npca: None,
            npca_glow_levels: None,
            npca_glow_steps: None,
            npca_glow_hidden: None,
            npca_glow_coupling: None,
            npca_rotation: None,
            npca_householder_reflectors: None,
        }
    }
}
