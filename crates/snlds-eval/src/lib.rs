//! Inference + Rerun logging for a trained `VariationalSnlds` checkpoint.
//!
//! The library exposes a single entry point [`run_eval`] used by the `snlds-eval`
//! binary; the binary just translates `clap` arguments into [`EvalConfig`].
//!
//! Outputs logged to Rerun (per `--sequences`):
//! - `snlds/markov/q_inferred` and `snlds/markov/q_inferred/weights` — softmax(`q_logits`).
//! - `snlds/state/strip_inferred` — argmax(γ) per timestep, as a colored band.
//! - `snlds/state/gamma` — posterior `γ` heatmap (Figure 6 of arXiv:2305.15925).
//! - `snlds/obs/x_hat[_d{d}]` — model reconstructions per sequence.
//!
//! The companion `snlds-viz` binary still owns the ground-truth side
//! (`q_true`, `strip_true`); a typical workflow logs both into the same `.rrd`
//! by running `snlds-viz` first and `snlds-eval` second with `--append`.

pub mod accuracy;

pub use accuracy::{align_with_hungarian, print_report, AccuracyReport};

use anyhow::Context;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, Tensor};
use glow_flow::prelude::TriangularInverse;
use ndarray::{Array2, Array3, Axis};
use snlds_data::{load_tensor_i32, Manifest};
use snlds_model::{
    CouplingType, EncoderKind, FlowSnlds, FlowSnldsConfig, PcaSvdBackend, SnldsConfig,
    VariationalSnlds,
};
use snlds_train::{FlowSnldsSnapshotMeta, SequenceBatcher, SequenceDataset, TrainSnapshot};
use std::path::{Path, PathBuf};

/// Which split of the dataset to evaluate against. All splits are accessed
/// through Burn's [`Dataset`] trait via [`SequenceDataset`].
///
/// `Test` is the per-epoch validation split consumed by `snlds-train`. `Eval`
/// is a separately-generated held-out split (schema v5+) that is never seen
/// during training — use it for unbiased final accuracy reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataSplit {
    Train,
    Test,
    Eval,
}

impl DataSplit {
    /// Open the corresponding [`SequenceDataset`]. For `Test` and `Eval`,
    /// returns `Err` when the dataset has no matching tensor.
    fn open(&self, data_dir: &Path) -> anyhow::Result<SequenceDataset> {
        match self {
            DataSplit::Train => SequenceDataset::open(data_dir)
                .with_context(|| format!("open obs_train dataset at {data_dir:?}")),
            DataSplit::Test => SequenceDataset::open_val(data_dir)
                .with_context(|| format!("open obs_test dataset at {data_dir:?}"))?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "dataset at {:?} has no obs_test tensor — regenerate with a test split or use --split train",
                        data_dir
                    )
                }),
            DataSplit::Eval => SequenceDataset::open_eval(data_dir)
                .with_context(|| format!("open obs_eval dataset at {data_dir:?}"))?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "dataset at {:?} has no obs_eval tensor — regenerate with `snlds-gen --eval-fraction > 0` or use --split {{train|test}}",
                        data_dir
                    )
                }),
        }
    }

    /// Name of the per-timestep ground-truth states tensor for this split
    /// inside `sequences.safetensors`.
    fn states_tensor_name(&self) -> &'static str {
        match self {
            DataSplit::Train => "states_train",
            DataSplit::Test => "states_test",
            DataSplit::Eval => "states_eval",
        }
    }
}

/// Resolved hyperparameters used during evaluation. All fields come from the
/// training-time snapshot ([`TrainSnapshot`]) by default and can be overridden
/// individually via [`EvalConfig`] before calling [`run_eval`].
#[derive(Clone, Debug)]
pub struct ResolvedHparams {
    pub hidden_dim: usize,
    pub temperature: f32,
    pub obs_noise_var: f32,
    pub beta: f32,
    /// Encoder/decoder family the model was trained with. Loaded from the
    /// snapshot so eval rebuilds the same `SnldsConfig` (and thus the same
    /// parameter layout the checkpoint expects).
    pub kind: EncoderKind,
    /// Present for FlowSNLDS runs (`train_config.json` includes `flow_snlds`).
    pub flow: Option<FlowSnldsSnapshotMeta>,
}

/// Configuration for one evaluation run.
///
/// Hyperparameters that must match the training run (`hidden_dim`, `temperature`,
/// `obs_noise_var`, `beta`) are loaded from the `train_config.json` snapshot
/// written by `snlds-train` next to its checkpoints. Pass `Some(…)` on any field
/// to override the snapshot value (e.g. annealing temperature for inference).
#[derive(Clone, Debug)]
pub struct EvalConfig {
    /// Directory containing `sequences.safetensors` + `metadata.json`.
    pub data_dir: PathBuf,
    /// Path to a `CompactRecorder` checkpoint produced by `snlds-train`.
    pub checkpoint: PathBuf,
    /// Recording stream output (file path) for new recordings; ignored when `spawn=true`.
    pub output: PathBuf,
    /// If `true`, spawn the live Rerun viewer instead of writing to `output`.
    pub spawn: bool,
    /// Which dataset split to evaluate against. Both splits flow through
    /// [`SequenceDataset`] (Burn's `Dataset` trait).
    pub split: DataSplit,
    /// Number of sequences to log from the chosen split.
    pub sequences: usize,
    /// Override the snapshot's `hidden_dim` (model layout — must match the checkpoint
    /// or `load_record` will fail).
    pub hidden_dim_override: Option<usize>,
    /// Override the snapshot's softmax temperature for `q_logits` / `pi_logits`.
    pub temperature_override: Option<f32>,
    /// Override the snapshot's observation noise variance used in the ELBO.
    pub obs_noise_var_override: Option<f32>,
    /// Override the snapshot's `beta` weight on the discrete-state ELBO term (VariationalSnlds only). Must be
    /// strictly positive so the forward pass populates posteriors.
    pub beta_override: Option<f32>,
    /// Override FlowSNLDS `w_msm` (must be > 0 for posteriors).
    pub w_msm_override: Option<f32>,
    /// Override FlowSNLDS `w_npca`.
    pub w_npca_override: Option<f32>,
    /// When `true`, load the corresponding ground-truth `states_*` tensor for
    /// the chosen split and print a Hungarian-aligned accuracy report
    /// comparing it against argmax(γ).
    pub report_accuracy: bool,
}

/// Resolve the per-run hyperparameters from the training snapshot, applying any
/// overrides set on `config`.
pub fn resolve_hparams(config: &EvalConfig) -> anyhow::Result<ResolvedHparams> {
    let snapshot = TrainSnapshot::load_for_checkpoint(&config.checkpoint).with_context(|| {
        format!(
            "load train_config.json snapshot next to checkpoint {:?}",
            config.checkpoint
        )
    })?;
    let hparams = ResolvedHparams {
        hidden_dim: config.hidden_dim_override.unwrap_or(snapshot.hidden_dim),
        temperature: config.temperature_override.unwrap_or(snapshot.temperature),
        obs_noise_var: config
            .obs_noise_var_override
            .unwrap_or(snapshot.obs_noise_var),
        beta: config.beta_override.unwrap_or(snapshot.beta),
        kind: snapshot.kind.clone(),
        flow: snapshot.flow_snlds.clone(),
    };
    if hparams.flow.is_some() {
        let w_msm = config
            .w_msm_override
            .unwrap_or_else(|| hparams.flow.as_ref().map(|m| m.w_msm).unwrap_or(1.0));
        anyhow::ensure!(
            w_msm > 0.0,
            "FlowSNLDS eval needs w_msm > 0 to populate posteriors (got {})",
            w_msm
        );
    } else {
        anyhow::ensure!(
            hparams.beta > 0.0,
            "beta must be > 0 to populate posteriors for VariationalSnlds (got {})",
            hparams.beta,
        );
    }
    Ok(hparams)
}

/// Surface a CNN/manifest mismatch as a user-facing error before the eventual
/// panic inside `SnldsConfig::init`. `manifest.dim_obs` must equal `3 * res * res`
/// when the snapshot says the run trained with `EncoderKind::Cnn { res }`.
fn ensure_kind_matches_manifest(kind: &EncoderKind, manifest: &Manifest) -> anyhow::Result<()> {
    if let EncoderKind::Cnn { res } = kind {
        let expected = 3 * res * res;
        anyhow::ensure!(
            manifest.dim_obs == expected,
            "data dim_obs {} does not match snapshot EncoderKind::Cnn {{ res: {res} }} (expected {expected})",
            manifest.dim_obs,
        );
    }
    Ok(())
}

/// Run inference + Rerun logging end-to-end.
pub fn run_eval<B: Backend<FloatElem = f32> + TriangularInverse + PcaSvdBackend>(
    config: &EvalConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    let hparams = resolve_hparams(config)?;
    let dataset = config.split.open(&config.data_dir)?;
    let manifest = dataset.manifest.clone();
    ensure_kind_matches_manifest(&hparams.kind, &manifest)?;

    let num_seqs = config.sequences.min(dataset.len());
    if num_seqs == 0 {
        anyhow::bail!(
            "--sequences resolved to 0 (split has {} sequences)",
            dataset.len()
        );
    }

    let obs_subset = build_obs_subset::<B>(&dataset, num_seqs, &manifest, device);

    let (q_inferred_tensor, gamma, x_hat) = if let Some(ref meta) = hparams.flow {
        let model = load_flow_checkpoint::<B>(config, &hparams, &manifest, device)?;
        let w_msm = config.w_msm_override.unwrap_or(meta.w_msm);
        let w_npca = config.w_npca_override.unwrap_or(meta.w_npca);
        let forward_output = model.forward(obs_subset, w_msm, w_npca, hparams.temperature, false);
        let gamma = forward_output.state_posteriors.ok_or_else(|| {
            anyhow::anyhow!("forward produced no state posteriors (w_msm must be > 0)")
        })?;
        let x_hat = model.decode_observations(
            forward_output.npca_output.z_pca.clone(),
            forward_output.npca_output.z_prefix.clone(),
            &forward_output.npca_output.latent_shapes,
            forward_output.npca_output.batch_stats.clone(),
            (num_seqs, manifest.seq_length),
        );
        (
            softmax(model.q_logits.val() / hparams.temperature, 1),
            gamma,
            x_hat,
        )
    } else {
        let model = load_variational_checkpoint::<B>(config, &hparams, &manifest, device)?;
        let forward_output = model.forward(
            obs_subset,
            hparams.beta,
            hparams.obs_noise_var,
            hparams.temperature,
        );
        let gamma = forward_output
            .state_posteriors
            .ok_or_else(|| anyhow::anyhow!("forward produced no posteriors (beta must be > 0)"))?;
        (
            softmax(model.q_logits.val() / hparams.temperature, 1),
            gamma,
            forward_output.obs_reconstructed,
        )
    };

    let q_inferred = tensor2_to_array(q_inferred_tensor)?;
    let gamma_array = tensor3_to_array(gamma)?;
    let x_hat_array = tensor3_to_array(x_hat)?;

    let recording = if config.spawn {
        rerun::RecordingStreamBuilder::new("snlds-eval")
            .spawn()
            .context("spawn Rerun viewer")?
    } else {
        rerun::RecordingStreamBuilder::new("snlds-eval")
            .save(&config.output)
            .with_context(|| format!("open output {:?}", config.output))?
    };

    snlds_viz::log_transition_matrix(&recording, "snlds/markov/q_inferred", q_inferred.view())?;

    let mut inferred_states_grid = Array2::<i32>::zeros((num_seqs, manifest.seq_length));

    for seq_idx in 0..num_seqs {
        let seq_idx_i64 = seq_idx as i64;
        recording.set_time_sequence("sequence", seq_idx_i64);

        let gamma_seq = gamma_array.index_axis(Axis(0), seq_idx);
        let x_hat_seq = x_hat_array.index_axis(Axis(0), seq_idx);

        snlds_viz::log_gamma_heatmap(&recording, "snlds/state/gamma", gamma_seq.view())?;

        let inferred_states: Vec<i32> = gamma_seq
            .axis_iter(Axis(0))
            .map(|row| {
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (state_idx, &value) in row.iter().enumerate() {
                    if value > best_val {
                        best_val = value;
                        best_idx = state_idx;
                    }
                }
                best_idx as i32
            })
            .collect();
        for (time_idx, &label) in inferred_states.iter().enumerate() {
            inferred_states_grid[[seq_idx, time_idx]] = label;
        }
        snlds_viz::log_state_strip(&recording, "snlds/state/strip_inferred", &inferred_states)?;

        snlds_viz::log_posteriors(&recording, seq_idx_i64, gamma_seq.view())?;
        snlds_viz::log_reconstructions(&recording, seq_idx_i64, x_hat_seq.view())?;
    }

    if !config.spawn {
        println!("Saved to {:?}", config.output);
    }

    if config.report_accuracy {
        let truth = load_truth_states(&config.data_dir, config.split, &manifest, num_seqs)?;
        let report = align_with_hungarian(
            truth.view(),
            inferred_states_grid.view(),
            manifest.num_states,
        )?;
        print_report(&report);
    }
    Ok(())
}

/// Pull the first `num_seqs` items out of `dataset` (Burn's `Dataset` trait)
/// and batch them into a `[num_seqs, T, D]` tensor via [`SequenceBatcher`].
fn build_obs_subset<B: Backend>(
    dataset: &SequenceDataset,
    num_seqs: usize,
    manifest: &Manifest,
    device: &B::Device,
) -> Tensor<B, 3> {
    let batcher = SequenceBatcher {
        seq_length: manifest.seq_length,
        obs_dim: manifest.dim_obs,
    };
    let items: Vec<_> = (0..num_seqs)
        .map(|seq_idx| {
            dataset
                .get(seq_idx)
                .expect("seq_idx < num_seqs <= dataset.len() by construction")
        })
        .collect();
    batcher.batch(items, device).obs
}

/// Load the first `num_seqs` rows of the split's ground-truth states tensor.
/// Mirrors [`SequenceDataset`]'s shard handling: if `<data_dir>` contains
/// `shard_NNN/` subdirectories, the states tensor is concatenated across
/// shards (in the same sort order as the obs side) and then sliced; otherwise
/// the top-level `sequences.safetensors` is read.
fn load_truth_states(
    data_dir: &Path,
    split: DataSplit,
    manifest: &Manifest,
    num_seqs: usize,
) -> anyhow::Result<Array2<i32>> {
    let tensor_name = split.states_tensor_name();
    let seq_length = manifest.seq_length;
    let shard_dirs = discover_shards_sorted(data_dir);
    let mut flat: Vec<i32> = if shard_dirs.is_empty() {
        let st_path = data_dir.join("sequences.safetensors");
        load_tensor_i32(&st_path, tensor_name)
            .with_context(|| format!("load {tensor_name} from {st_path:?}"))?
    } else {
        let mut combined = Vec::with_capacity(num_seqs.saturating_mul(seq_length));
        for shard_dir in &shard_dirs {
            let st_path = shard_dir.join("sequences.safetensors");
            let shard_flat = load_tensor_i32(&st_path, tensor_name)
                .with_context(|| format!("load {tensor_name} from {st_path:?}"))?;
            combined.extend(shard_flat);
            if combined.len() >= num_seqs * seq_length {
                break;
            }
        }
        combined
    };
    let needed = num_seqs * seq_length;
    anyhow::ensure!(
        flat.len() >= needed,
        "loaded {tensor_name} length {} < needed {needed} ({num_seqs} sequences × {seq_length} timesteps)",
        flat.len(),
    );
    flat.truncate(needed);
    Array2::from_shape_vec((num_seqs, seq_length), flat)
        .with_context(|| format!("reshape {tensor_name} into [{num_seqs}, {seq_length}]"))
}

/// Return shard subdirectories under `data_dir` in lexicographic order, or an
/// empty `Vec` if the directory is not sharded. Matches the discovery rule
/// used by `SequenceDataset::open` so the obs and states stay aligned.
fn discover_shards_sorted(data_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(data_dir) else {
        return Vec::new();
    };
    let mut shards: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name();
            if path.is_dir() && name.to_string_lossy().starts_with("shard_") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    shards
}

fn load_variational_checkpoint<B: Backend>(
    config: &EvalConfig,
    hparams: &ResolvedHparams,
    manifest: &Manifest,
    device: &B::Device,
) -> anyhow::Result<VariationalSnlds<B>> {
    let snlds_config = SnldsConfig::new(
        manifest.dim_obs,
        manifest.dim_latent,
        hparams.hidden_dim,
        manifest.num_states,
    )
    .with_kind(hparams.kind.clone());
    let model: VariationalSnlds<B> = snlds_config.init(device);
    let recorder = CompactRecorder::new();
    let record = recorder
        .load(config.checkpoint.clone(), device)
        .with_context(|| format!("load checkpoint {:?}", config.checkpoint))?;
    Ok(model.load_record(record))
}

fn load_flow_checkpoint<B: Backend<FloatElem = f32> + PcaSvdBackend>(
    config: &EvalConfig,
    hparams: &ResolvedHparams,
    manifest: &Manifest,
    device: &B::Device,
) -> anyhow::Result<FlowSnlds<B>> {
    let meta = hparams
        .flow
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("load_flow_checkpoint called without flow snapshot"))?;
    let coupling_type = match meta.glow_coupling.as_str() {
        "additive" => CouplingType::Additive,
        _ => CouplingType::Affine,
    };
    let flow_config = FlowSnldsConfig::new(
        manifest.dim_obs,
        manifest.dim_latent,
        hparams.hidden_dim,
        manifest.num_states,
        meta.res,
        meta.glow_levels,
        meta.glow_steps,
        meta.glow_hidden_features,
    )
    .with_coupling_type(coupling_type)
    .with_householder_rotation(meta.npca_rotation == "householder")
    .with_householder_reflectors(meta.npca_householder_reflectors.unwrap_or(32));
    let mut model: FlowSnlds<B> = flow_config.init(device);
    let recorder = CompactRecorder::new();
    let record = recorder
        .load(config.checkpoint.clone(), device)
        .with_context(|| format!("load checkpoint {:?}", config.checkpoint))?;
    model = model.load_record(record);
    model.npca.sync_training_mode_after_load();
    Ok(model)
}

fn tensor2_to_array<B: Backend>(tensor: Tensor<B, 2>) -> anyhow::Result<Array2<f32>> {
    let [rows, cols] = tensor.dims();
    let data = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("convert 2-D tensor to f32 vec failed: {err:?}"))?;
    Array2::from_shape_vec((rows, cols), data).context("reshape into Array2")
}

fn tensor3_to_array<B: Backend>(tensor: Tensor<B, 3>) -> anyhow::Result<Array3<f32>> {
    let [batch_size, seq_len, last_dim] = tensor.dims();
    let data = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("convert 3-D tensor to f32 vec failed: {err:?}"))?;
    Array3::from_shape_vec((batch_size, seq_len, last_dim), data).context("reshape into Array3")
}
