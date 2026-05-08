//! CLI: synthetic data + SafeTensors export (subset of Python `generate_data_and_train_snlds.py`).
//!
//! ```text
//! snlds-gen --seed 42 --dim-obs 2 --dim-latent 2 --num-states 3 --seq-length 32 --num-samples 16 --data-type cosine --out ./out/run1
//! ```
//!
//! Bouncing-ball style RGB frames (2-D latent rendered with `draw_sequence`):
//!
//! ```text
//! snlds-gen --observation image --res 32 --seq-length 64 --num-samples 256 --out ./out/ball
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use snlds_data::io::MANIFEST_SCHEMA_VERSION;
use snlds_data::{
    generate_shard, generate_train_test, save_train_test, GenConfig, Manifest, ObservationKind,
    SimulatorKind,
};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DataCli {
    Cosine,
    Poly,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum ObservationCli {
    /// Leaky-ReLU emission MLP to `dim-obs` (default).
    #[default]
    Vector,
    /// Flat RGB `[res*res*3]` from rendered 2-D latents (`dim-latent` forced to 2).
    Image,
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
    /// With `--observation image`, frame side length (`dim-obs` becomes `3*res*res`, `dim-latent` = 2).
    #[arg(long)]
    res: Option<usize>,
    #[arg(long, value_enum, default_value_t = ObservationCli::Vector)]
    observation: ObservationCli,
    /// Split generation into N shards written to `<out>/shard_000/`, etc.
    /// Each shard holds `num_samples / num_shards` sequences (remainder goes
    /// to the last shard). Keeps peak memory proportional to shard size.
    #[arg(long, default_value_t = 1)]
    num_shards: usize,
    #[arg(short, long, default_value = "./snlds-gen-out")]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kind = match args.data_type {
        DataCli::Cosine => SimulatorKind::Cosine,
        DataCli::Poly => SimulatorKind::Poly,
    };

    if matches!(args.observation, ObservationCli::Vector) && args.res.is_some() {
        anyhow::bail!("--res is only valid with --observation image");
    }

    let (observation, dim_obs, dim_latent) = match args.observation {
        ObservationCli::Vector => (ObservationKind::Vector, args.dim_obs, args.dim_latent),
        ObservationCli::Image => {
            let res = args
                .res
                .context("--observation image requires --res (e.g. 16, 32)")?;
            anyhow::ensure!(res > 0, "--res must be > 0");
            let expected = res * res * 3;
            (ObservationKind::Image { res }, expected, 2usize)
        }
    };

    // TODO(M1+): expose --init-noise-std, --init-mean-std, --transition-step-var,
    // --emission-hidden-dim, and --initial-distribution as CLI flags. Currently
    // pinned to GenConfig::default() — see docs/CLEANUP-hardcoded-values.md.
    let cfg = GenConfig {
        seed: args.seed,
        num_states: args.num_states,
        dim_obs,
        dim_latent,
        seq_length: args.seq_length,
        num_samples: args.num_samples,
        sparsity_prob: args.sparsity_prob,
        kind,
        poly_degree: args.degree,
        observation,
        ..GenConfig::default()
    };

    let make_manifest = |n_samples: usize| Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        seed: cfg.seed,
        num_states: cfg.num_states,
        dim_obs: cfg.dim_obs,
        dim_latent: cfg.dim_latent,
        seq_length: cfg.seq_length,
        num_samples: n_samples,
        sparsity_prob: cfg.sparsity_prob,
        data_type: match cfg.kind {
            SimulatorKind::Cosine => "cosine".into(),
            SimulatorKind::Poly => "poly".into(),
        },
        degree: match cfg.kind {
            SimulatorKind::Poly => Some(cfg.poly_degree),
            _ => None,
        },
        init_noise_std: cfg.init_noise_std,
        init_mean_std: cfg.init_mean_std,
        transition_step_var: cfg.transition_step_var,
        emission_hidden_dim: cfg.emission_hidden_dim,
    };

    if args.num_shards <= 1 {
        let tt = generate_train_test(&cfg)?;
        let manifest = make_manifest(cfg.num_samples);
        save_train_test(&args.out, &tt, &manifest)?;
        eprintln!(
            "Wrote sequences.safetensors + metadata.json under {:?}",
            args.out
        );
    } else {
        for shard in 0..args.num_shards {
            eprintln!(
                "Generating shard {}/{} ...",
                shard + 1,
                args.num_shards
            );
            let tt = generate_shard(&cfg, shard, args.num_shards)?;
            let n = tt.obs_train.shape()[0];
            let manifest = make_manifest(n);
            let shard_dir = args.out.join(format!("shard_{:03}", shard));
            save_train_test(&shard_dir, &tt, &manifest)?;
            eprintln!("  → {:?} ({} sequences)", shard_dir, n);
        }
        eprintln!(
            "Wrote {} shards under {:?}",
            args.num_shards, args.out
        );
    }
    Ok(())
}
