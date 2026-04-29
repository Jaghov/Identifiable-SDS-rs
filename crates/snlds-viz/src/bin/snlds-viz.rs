use anyhow::Context;
use clap::Parser;
use ndarray::Array2;
use snlds_data::{load_manifest, load_tensor_f32, load_tensor_i32};
use snlds_viz::{log, render};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "snlds-viz",
    about = "Visualise SNLDS ground-truth sequences in Rerun"
)]
struct Args {
    /// Directory containing sequences.safetensors + metadata.json
    #[arg(long)]
    input: PathBuf,

    /// Number of sequences to log
    #[arg(long, default_value_t = 5)]
    sequences: usize,

    /// Which split to visualise
    #[arg(long, default_value = "train")]
    split: String,

    /// Output .rrd file path
    #[arg(long, default_value = "snlds_gt.rrd")]
    output: PathBuf,

    /// Enable _draw image frame rendering (requires dim_latent == 2)
    #[arg(long)]
    render: bool,

    /// Spawn live Rerun viewer
    #[arg(long)]
    spawn: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let manifest_path = args.input.join("metadata.json");
    let st_path = args.input.join("sequences.safetensors");

    let manifest = load_manifest(&manifest_path)
        .with_context(|| format!("load manifest from {manifest_path:?}"))?;

    let (latent_key, obs_key, state_key) = match args.split.as_str() {
        "train" => ("latents_train", "obs_train", "states_train"),
        "test" => ("latents_test", "obs_test", "states_test"),
        other => anyhow::bail!("unknown split {:?}; use train or test", other),
    };

    let n = manifest.num_samples;
    let t = manifest.seq_length;
    let d_lat = manifest.dim_latent;
    let d_obs = manifest.dim_obs;
    let num_states = manifest.num_states;

    if args.render && d_lat != 2 {
        anyhow::bail!("--render requires dim_latent == 2 (dataset has {d_lat})");
    }

    let latents =
        load_tensor_f32(&st_path, latent_key).with_context(|| format!("load {latent_key}"))?;
    let obs = load_tensor_f32(&st_path, obs_key).with_context(|| format!("load {obs_key}"))?;
    let states =
        load_tensor_i32(&st_path, state_key).with_context(|| format!("load {state_key}"))?;

    let q_true_flat = load_tensor_f32(&st_path, "q_true").context("load q_true")?;
    let q_true =
        Array2::from_shape_vec([num_states, num_states], q_true_flat).context("reshape q_true")?;

    let num_seqs = args.sequences.min(n);

    let rec = if args.spawn {
        rerun::RecordingStreamBuilder::new("snlds-viz")
            .spawn()
            .context("spawn Rerun viewer")?
    } else {
        rerun::RecordingStreamBuilder::new("snlds-viz")
            .save(&args.output)
            .with_context(|| format!("open output {:?}", args.output))?
    };

    // Log the ground-truth Markov chain once (sequence-independent).
    log::log_transition_matrix(&rec, "snlds/markov/q_true", q_true.view())?;

    for seq in 0..num_seqs {
        let seq_states = &states[seq * t..(seq + 1) * t];
        log::log_state_s(&rec, seq as i64, seq_states)?;
        rec.set_time_sequence("sequence", seq as i64);
        log::log_state_strip(&rec, "snlds/state/strip_true", seq_states)?;

        if d_lat == 2 {
            let lat_arr = Array2::from_shape_vec(
                [t, 2],
                latents[seq * t * d_lat..(seq + 1) * t * d_lat].to_vec(),
            )
            .context("reshape latents")?;
            log::log_latent_z(&rec, seq as i64, lat_arr.view())?;

            if args.render {
                let frames = render::draw_sequence(lat_arr.view(), 64);
                log::log_render_frames(&rec, seq as i64, frames.view())?;
            }
        }

        if d_obs == 2 {
            let obs_arr = Array2::from_shape_vec(
                [t, 2],
                obs[seq * t * d_obs..(seq + 1) * t * d_obs].to_vec(),
            )
            .context("reshape obs")?;
            log::log_obs_x(&rec, seq as i64, obs_arr.view())?;
        }
    }

    if !args.spawn {
        println!("Saved to {:?}", args.output);
    }

    Ok(())
}
