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

    /// Hidden MLP dimension used during training (must match checkpoint).
    #[arg(long, default_value_t = 64)]
    hidden_dim: usize,

    /// Softmax temperature applied to `q_logits` / `pi_logits` during inference.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
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
        hidden_dim: cli.hidden_dim,
        temperature: cli.temperature,
    };
    run_eval::<Backend>(&config, &device).context("run snlds-eval")
}
