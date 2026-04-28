//! CLI: synthetic data + SafeTensors export (subset of Python `generate_data_and_train_snlds.py`).
//!
//! ```text
//! snlds-gen --seed 42 --dim-obs 2 --dim-latent 2 --num-states 3 --seq-length 32 --num-samples 16 --data-type cosine --out ./out/run1
//! ```

use anyhow::Result;
use clap::{Parser, ValueEnum};
use snlds_data::io::MANIFEST_SCHEMA_VERSION;
use snlds_data::{generate_train_test, save_train_test, GenConfig, Manifest, SimulatorKind};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DataCli {
    Cosine,
    Poly,
}

#[derive(Parser, Debug)]
#[command(
    name = "snlds-gen",
    about = "Generate synthetic SDS data (Rust M1)",
    version
)]
struct Args {
    #[arg(long, default_value_t = 24)]
    seed: u64,
    #[arg(long, default_value_t = 2)]
    dim_obs: usize,
    #[arg(long, default_value_t = 2)]
    dim_latent: usize,
    #[arg(long, default_value_t = 3)]
    num_states: usize,
    #[arg(long, default_value_t = 200)]
    seq_length: usize,
    #[arg(long, default_value_t = 5000)]
    num_samples: usize,
    #[arg(long, default_value_t = 0.0)]
    sparsity_prob: f32,
    #[arg(long, value_enum, default_value_t = DataCli::Cosine)]
    data_type: DataCli,
    #[arg(long, default_value_t = 3)]
    degree: usize,
    #[arg(short, long, default_value = "./snlds-gen-out")]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kind = match args.data_type {
        DataCli::Cosine => SimulatorKind::Cosine,
        DataCli::Poly => SimulatorKind::Poly,
    };

    let cfg = GenConfig {
        seed: args.seed,
        num_states: args.num_states,
        dim_obs: args.dim_obs,
        dim_latent: args.dim_latent,
        seq_length: args.seq_length,
        num_samples: args.num_samples,
        sparsity_prob: args.sparsity_prob,
        kind,
        poly_degree: args.degree,
    };

    let tt = generate_train_test(&cfg);
    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        seed: cfg.seed,
        num_states: cfg.num_states,
        dim_obs: cfg.dim_obs,
        dim_latent: cfg.dim_latent,
        seq_length: cfg.seq_length,
        num_samples: cfg.num_samples,
        sparsity_prob: cfg.sparsity_prob,
        data_type: match cfg.kind {
            SimulatorKind::Cosine => "cosine".into(),
            SimulatorKind::Poly => "poly".into(),
        },
        degree: match cfg.kind {
            SimulatorKind::Poly => Some(cfg.poly_degree),
            _ => None,
        },
    };
    save_train_test(&args.out, &tt, &manifest)?;
    eprintln!(
        "Wrote sequences.safetensors + metadata.json under {:?}",
        args.out
    );
    Ok(())
}
