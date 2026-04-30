//! `snlds-train` — Adam minibatch training CLI for `VariationalSnlds`.

use anyhow::Result;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use clap::{Parser, ValueEnum};
use snlds_model::EncoderKind;
use snlds_train::{
    build_model_config, load_train_obs, load_train_obs_array, run_warm_start, train_with_model,
    MsmWarmStartConfig, TrainConfig, DEFAULT_OBS_NOISE_VAR,
};
use std::path::PathBuf;

/// CLI surface for [`EncoderKind`]. Kept separate so the model crate stays free
/// of `clap` derive details and we can validate the (`encoder`, `res`) pair as
/// one unit before constructing the real enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EncoderCli {
    /// Two-hidden-layer leaky-ReLU MLP pair (Python `factored` parity, default).
    Mlp,
    /// CNN encoder/decoder pair for flat RGB image observations. Requires `--res`.
    Cnn,
}

type TrainBackend = Autodiff<NdArray<f32>>;

#[derive(Parser, Debug)]
#[command(about = "Train a VariationalSnlds model on M1 SafeTensors data.")]
struct Cli {
    /// Directory containing `sequences.safetensors` and `metadata.json`.
    #[arg(long)]
    data_dir: PathBuf,

    /// Directory to write checkpoint files to.
    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 100)]
    epochs: usize,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long = "lr", default_value_t = 3e-4)]
    learning_rate: f64,

    #[arg(long, default_value_t = 1.0)]
    beta: f32,

    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    #[arg(long, default_value_t = 10)]
    checkpoint_every: usize,

    #[arg(long, default_value_t = 64)]
    hidden_dim: usize,

    /// Observation noise variance used in the ELBO reconstruction term.
    /// Persisted in `<output_dir>/train_config.json`.
    #[arg(long, default_value_t = DEFAULT_OBS_NOISE_VAR)]
    obs_noise_var: f32,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Optional checkpoint file to resume training from.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Enable M5 NeuralMSM warm-start (PCA → MSM fit → parameter transfer).
    #[arg(long, default_value_t = false)]
    msm_init: bool,

    /// Number of MSM random restarts (best by mean log-likelihood is kept).
    #[arg(long, default_value_t = 3)]
    msm_restarts: usize,

    /// MSM training epochs per restart.
    #[arg(long, default_value_t = 30)]
    msm_epochs: usize,

    /// MSM minibatch size.
    #[arg(long, default_value_t = 32)]
    msm_batch_size: usize,

    /// MSM learning rate.
    #[arg(long, default_value_t = 7e-3)]
    msm_lr: f64,

    /// Hidden dim for the MSM transition MLPs.
    #[arg(long, default_value_t = 16)]
    msm_hidden_dim: usize,

    /// Encoder/decoder family. Default `mlp` (Python `factored` parity); `cnn`
    /// requires `--res` and assumes flat-RGB image observations of length
    /// `3 * res * res`.
    #[arg(long, value_enum, default_value_t = EncoderCli::Mlp)]
    encoder: EncoderCli,

    /// Image resolution for `--encoder cnn`. Required when `encoder=cnn`;
    /// must be a power of two `≥ 16`.
    #[arg(long)]
    res: Option<usize>,
}

/// Resolve the `(encoder, res)` CLI pair into an [`EncoderKind`], surfacing
/// the "missing `--res`" and "stray `--res`" cases as user-facing errors.
fn resolve_encoder_kind(encoder: EncoderCli, res: Option<usize>) -> Result<EncoderKind> {
    match (encoder, res) {
        (EncoderCli::Mlp, None) => Ok(EncoderKind::Mlp),
        (EncoderCli::Mlp, Some(res)) => {
            anyhow::bail!("--res {res} given with --encoder mlp; --res is only valid for cnn")
        }
        (EncoderCli::Cnn, Some(res)) => Ok(EncoderKind::Cnn { res }),
        (EncoderCli::Cnn, None) => {
            anyhow::bail!("--encoder cnn requires --res <usize> (e.g. 16, 32)")
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let device = NdArrayDevice::default();
    let obs_tensor = load_train_obs::<TrainBackend>(&cli.data_dir, &device)?;
    let kind = resolve_encoder_kind(cli.encoder, cli.res)?;

    let config = TrainConfig {
        data_dir: cli.data_dir.clone(),
        output_dir: cli.output_dir,
        epochs: cli.epochs,
        batch_size: cli.batch_size,
        learning_rate: cli.learning_rate,
        beta: cli.beta,
        temperature: cli.temperature,
        grad_clip: cli.grad_clip,
        checkpoint_every: cli.checkpoint_every,
        hidden_dim: cli.hidden_dim,
        obs_noise_var: cli.obs_noise_var,
        seed: cli.seed,
        resume_from: cli.resume,
        kind,
    };

    let snlds_config = build_model_config(&config, &obs_tensor.manifest);
    let initial_model = if cli.msm_init {
        let warm_config = MsmWarmStartConfig {
            restarts: cli.msm_restarts,
            epochs: cli.msm_epochs,
            batch_size: cli.msm_batch_size,
            learning_rate: cli.msm_lr,
            hidden_dim: cli.msm_hidden_dim,
        };
        let (obs_array, _manifest) = load_train_obs_array(&cli.data_dir)?;
        run_warm_start::<TrainBackend>(&warm_config, &snlds_config, &obs_array, &device)?
    } else {
        snlds_config.init::<TrainBackend>(&device)
    };

    train_with_model::<TrainBackend>(&config, initial_model, obs_tensor, &device)?;
    Ok(())
}
