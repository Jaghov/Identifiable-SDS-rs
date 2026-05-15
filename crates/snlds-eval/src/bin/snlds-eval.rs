use anyhow::Context;
use burn::backend::LibTorch;
use clap::{Parser, ValueEnum};
use snlds_eval::{run_eval, DataSplit, EvalConfig};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SplitArg {
    Train,
    Test,
    /// Held-out evaluation split (schema v5+). Requires the dataset to be
    /// generated with `snlds-gen --eval-fraction > 0`.
    Eval,
}

impl From<SplitArg> for DataSplit {
    fn from(value: SplitArg) -> Self {
        match value {
            SplitArg::Train => DataSplit::Train,
            SplitArg::Test => DataSplit::Test,
            SplitArg::Eval => DataSplit::Eval,
        }
    }
}

type Backend = LibTorch<f32>;

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

    /// Dataset split to evaluate against (loaded via Burn's `Dataset` API).
    #[arg(long, value_enum, default_value_t = SplitArg::Train)]
    split: SplitArg,

    /// Number of sequences from the chosen split to log.
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

    /// Override the snapshot's `beta` weight on the discrete-state ELBO term (VariationalSnlds; must be > 0).
    #[arg(long)]
    beta: Option<f32>,

    /// Override FlowSNLDS `w_msm`.
    #[arg(long)]
    w_msm: Option<f32>,

    /// Override FlowSNLDS `w_npca`.
    #[arg(long)]
    w_npca: Option<f32>,

    /// Load the matching `states_*` tensor for the chosen `--split` and
    /// print a Hungarian-aligned accuracy report comparing argmax(γ) against
    /// the ground-truth state sequence.
    #[arg(long)]
    report_accuracy: bool,
}

fn main() -> anyhow::Result<()> {
    // Match the training-side TF32 setting: flow inverse round-trips need the full
    // f32 mantissa to stay numerically tight.
    glow_flow::disable_tf32();

    let cli = Cli::parse();
    let device = Default::default();
    let config = EvalConfig {
        data_dir: cli.data_dir,
        checkpoint: cli.checkpoint,
        output: cli.output,
        spawn: cli.spawn,
        split: cli.split.into(),
        sequences: cli.sequences,
        hidden_dim_override: cli.hidden_dim,
        temperature_override: cli.temperature,
        obs_noise_var_override: cli.obs_noise_var,
        beta_override: cli.beta,
        w_msm_override: cli.w_msm,
        w_npca_override: cli.w_npca,
        report_accuracy: cli.report_accuracy,
    };
    run_eval::<Backend>(&config, &device).context("run snlds-eval")
}
