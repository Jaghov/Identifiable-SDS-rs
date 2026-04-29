use anyhow::Context;
use burn::backend::NdArray;
use clap::Parser;
use snlds_eval::{run_eval, EvalConfig};
use std::path::PathBuf;

type Backend = NdArray<f32>;

#[derive(Parser)]
#[command(
    name = "snlds-eval",
    about = "Run a trained SNLDS checkpoint and log inferred Markov chain + posteriors to Rerun"
)]
struct Cli {
    /// Directory containing `sequences.safetensors` + `metadata.json`.
    #[arg(long)]
    data_dir: PathBuf,

    /// Path to a checkpoint produced by `snlds-train`.
    /// `train_config.json` (written next to the checkpoint by `snlds-train`) is
    /// loaded automatically; CLI overrides below take precedence when set.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Output `.rrd` path (ignored when `--spawn` is set).
    #[arg(long, default_value = "snlds_inferred.rrd")]
    output: PathBuf,

    /// Spawn the Rerun viewer instead of writing to `--output`.
    #[arg(long)]
    spawn: bool,

    /// Number of training sequences to log.
    #[arg(long, default_value_t = 5)]
    sequences: usize,

    /// Override the snapshot's MLP hidden dimension (must match the checkpoint).
    #[arg(long)]
    hidden_dim: Option<usize>,

    /// Override the snapshot's softmax temperature for `q_logits` / `pi_logits`.
    #[arg(long)]
    temperature: Option<f32>,

    /// Override the snapshot's observation noise variance (ELBO reconstruction term).
    #[arg(long)]
    obs_noise_var: Option<f32>,

    /// Override the snapshot's `beta` weight on the discrete-state ELBO term (must be > 0).
    #[arg(long)]
    beta: Option<f32>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let device = Default::default();
    let config = EvalConfig {
        data_dir: cli.data_dir,
        checkpoint: cli.checkpoint,
        output: cli.output,
        spawn: cli.spawn,
        sequences: cli.sequences,
        hidden_dim_override: cli.hidden_dim,
        temperature_override: cli.temperature,
        obs_noise_var_override: cli.obs_noise_var,
        beta_override: cli.beta,
    };
    run_eval::<Backend>(&config, &device).context("run snlds-eval")
}
