//! `snlds-train` — Adam minibatch training CLI for `VariationalSnlds`.

use anyhow::Result;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use clap::Parser;
use snlds_train::{load_train_obs, train, TrainConfig};
use std::path::PathBuf;

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

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Optional checkpoint file to resume training from.
    #[arg(long)]
    resume: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let device = NdArrayDevice::default();
    let obs_tensor = load_train_obs::<TrainBackend>(&cli.data_dir, &device)?;

    let config = TrainConfig {
        data_dir: cli.data_dir,
        output_dir: cli.output_dir,
        epochs: cli.epochs,
        batch_size: cli.batch_size,
        learning_rate: cli.learning_rate,
        beta: cli.beta,
        temperature: cli.temperature,
        grad_clip: cli.grad_clip,
        checkpoint_every: cli.checkpoint_every,
        hidden_dim: cli.hidden_dim,
        seed: cli.seed,
        resume_from: cli.resume,
    };

    train::<TrainBackend>(&config, obs_tensor, &device)?;
    Ok(())
}
