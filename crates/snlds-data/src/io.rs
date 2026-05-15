//! SafeTensors + JSON manifest export (see [docs/M1.md](../../../../docs/M1.md)).
//!
//! **`encode_safetensors`** holds per-tensor staging **`Vec<u8>`** buffers for one
//! **`safetensors::serialize`** call; **`TensorView`** borrows those slices (no **`Box::leak`**).

use crate::generate::{
    TrainTest, DEFAULT_INIT_MEAN_STD, DEFAULT_INIT_NOISE_STD, DEFAULT_TRANSITION_STEP_VAR,
};
use crate::transitions::EMISSION_HIDDEN_DIM;
use anyhow::Context;
use ndarray::ArrayViewD;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;

/// Bump history:
/// - **v5** (2026-04-30): adds the held-out **eval** split. New tensors
///   `latents_eval`, `obs_eval`, `states_eval` are written alongside the
///   train/test tensors. Shards that do not carry the eval batch persist
///   zero-row tensors (matching the existing test-split convention) so
///   `SequenceDataset::open_eval` can iterate every shard via Burn's
///   `Dataset` trait without a "tensor not found" error. Manifest gains
///   `num_samples_eval` (per-shard count, summed by the loader across
///   shards). v4 manifests on disk still load: `num_samples_eval` defaults
///   to `0` and `SequenceDataset::open_eval` returns `Ok(None)` when no
///   shard contains an `obs_eval` tensor.
/// - **v4** (2026-04-29): persist simulator hyperparameters that were previously
///   hardcoded in `generate.rs`: `init_noise_std`, `init_mean_std`, `transition_step_var`,
///   `emission_hidden_dim`. v3 manifests on disk still load — the new fields fall back
///   to the v3-era hardcoded defaults via `serde(default = "...")`.
/// - **v3** (2026-04-29): persist ground-truth Markov transition matrix `q_true` `[K, K]` and
///   initial distribution `pi_true` `[K]` so downstream viz / eval tools can compare against
///   a learned `Q`. Tensor names: `q_true`, `pi_true`.
/// - **v2** (M1): `states_*` stored as **`I32`** alongside the existing F32 tensors.
/// - **v1** (initial M1): `latents_*`, `obs_*` as `F32` and `states_*` accidentally as `F32`
///   (mirrored a Python `float64` layout); replaced by v2.
pub const MANIFEST_SCHEMA_VERSION: u32 = 5;

fn default_init_noise_std() -> f32 {
    DEFAULT_INIT_NOISE_STD
}
fn default_init_mean_std() -> f32 {
    DEFAULT_INIT_MEAN_STD
}
fn default_transition_step_var() -> f32 {
    DEFAULT_TRANSITION_STEP_VAR
}
fn default_emission_hidden_dim() -> usize {
    EMISSION_HIDDEN_DIM
}
fn default_num_samples_eval() -> usize {
    0
}

/// Run metadata written next to `sequences.safetensors`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct Manifest {
    pub schema_version: u32,
    pub seed: u64,
    pub num_states: usize,
    pub dim_obs: usize,
    pub dim_latent: usize,
    pub seq_length: usize,
    pub num_samples: usize,
    pub sparsity_prob: f32,
    pub data_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degree: Option<usize>,
    /// Std-dev of the Gaussian jitter on `z_0` used by the simulator.
    /// Default applied when loading a pre-v4 manifest (matches the v3-era hardcoded
    /// simulator constant [`crate::generate::DEFAULT_INIT_NOISE_STD`] = 0.1).
    #[serde(default = "default_init_noise_std")]
    pub init_noise_std: f32,
    /// Std-dev of the per-state init-mean prior used by the simulator.
    /// Default applied when loading a pre-v4 manifest (matches the v3-era hardcoded
    /// simulator constant [`crate::generate::DEFAULT_INIT_MEAN_STD`] = 0.7).
    #[serde(default = "default_init_mean_std")]
    pub init_mean_std: f32,
    /// Variance of the transition step noise added to `z_t` each step.
    /// (variance, not std-dev — fed to `Normal::new` as `sqrt(var)`.)
    /// Default applied when loading a pre-v4 manifest (matches the v3-era hardcoded
    /// simulator constant [`crate::generate::DEFAULT_TRANSITION_STEP_VAR`] = 0.05).
    #[serde(default = "default_transition_step_var")]
    pub transition_step_var: f32,
    /// Hidden dimension of the simulator's leaky-ReLU emission network.
    /// Default applied when loading a pre-v4 manifest (matches the v3-era hardcoded
    /// simulator constant [`crate::transitions::EMISSION_HIDDEN_DIM`] = 8).
    #[serde(default = "default_emission_hidden_dim")]
    pub emission_hidden_dim: usize,
    /// Number of held-out eval sequences in **this shard's**
    /// `sequences.safetensors` — per-shard, mirroring [`Self::num_samples`].
    /// In the sharded layout the entire eval batch lives in `shard_000/`, so
    /// only that shard's manifest carries a non-zero value; sibling shards
    /// report `0` and their `*_eval` tensors are zero-row. The dataset-wide
    /// eval count is the sum across shards (the loader does this when it
    /// opens the dataset). `0` everywhere means no eval split was generated.
    /// Default applied when loading a pre-v5 manifest is `0`.
    #[serde(default = "default_num_samples_eval")]
    pub num_samples_eval: usize,
}

/// Write `sequences.safetensors` and `metadata.json` into `out_dir`.
pub fn save_train_test(
    out_dir: impl AsRef<std::path::Path>,
    tt: &TrainTest,
    manifest: &Manifest,
) -> anyhow::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir).with_context(|| format!("create {:?}", out_dir))?;

    let data = encode_safetensors(tt)?;
    fs::write(out_dir.join("sequences.safetensors"), data)?;
    let manifest_bytes = serde_json::to_vec_pretty(manifest)?;
    fs::write(out_dir.join("metadata.json"), manifest_bytes)?;
    Ok(())
}

/// Load `metadata.json` written by [`save_train_test`].
pub fn load_manifest(path: impl AsRef<std::path::Path>) -> anyhow::Result<Manifest> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {:?}", path))?;
    let m = serde_json::from_slice::<Manifest>(&bytes)
        .with_context(|| format!("parse manifest {:?}", path))?;
    Ok(m)
}

fn pack_dyn_f32(view: ArrayViewD<'_, f32>) -> anyhow::Result<(Vec<usize>, Vec<u8>)> {
    let shape = view.shape().to_vec();
    let data: Vec<f32> = view.iter().copied().collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    Ok((shape, bytes))
}

fn pack_dyn_i32(view: ArrayViewD<'_, i32>) -> anyhow::Result<(Vec<usize>, Vec<u8>)> {
    let shape = view.shape().to_vec();
    let data: Vec<i32> = view.iter().copied().collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    Ok((shape, bytes))
}

fn tensor_view<'a>(
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
) -> anyhow::Result<TensorView<'a>> {
    TensorView::new(dtype, shape, data).map_err(|e| anyhow::anyhow!(e))
}

fn encode_safetensors(tt: &TrainTest) -> anyhow::Result<Vec<u8>> {
    let (sh_lt, b_lt) = pack_dyn_f32(tt.latents_train.view().into_dyn())?;
    let (sh_lte, b_lte) = pack_dyn_f32(tt.latents_test.view().into_dyn())?;
    let (sh_lev, b_lev) = pack_dyn_f32(tt.latents_eval.view().into_dyn())?;
    let (sh_ot, b_ot) = pack_dyn_f32(tt.obs_train.view().into_dyn())?;
    let (sh_ote, b_ote) = pack_dyn_f32(tt.obs_test.view().into_dyn())?;
    let (sh_oev, b_oev) = pack_dyn_f32(tt.obs_eval.view().into_dyn())?;
    let (sh_st, b_st) = pack_dyn_i32(tt.states_train.view().into_dyn())?;
    let (sh_ste, b_ste) = pack_dyn_i32(tt.states_test.view().into_dyn())?;
    let (sh_sev, b_sev) = pack_dyn_i32(tt.states_eval.view().into_dyn())?;
    let (sh_q, b_q) = pack_dyn_f32(tt.q_true.view().into_dyn())?;
    let (sh_pi, b_pi) = pack_dyn_f32(tt.pi_true.view().into_dyn())?;

    // The `*_eval` tensors are written unconditionally (matching the existing
    // `*_test` convention: shards that do not carry the eval batch persist
    // zero-row tensors so [`crate::data::SequenceDataset::open_shards`] can
    // iterate every shard without a "tensor not found" failure. Consumers
    // detect the dataset-wide presence of an eval split via
    // [`Manifest::num_samples_eval`] (per-shard count, summed by the loader),
    // and v4-and-older datasets continue to load via the graceful fallback
    // in `SequenceDataset::open_eval` (no `obs_eval` tensor in any shard ⇒
    // `Ok(None)`).
    let tensors: Vec<(&str, TensorView<'_>)> = vec![
        ("latents_train", tensor_view(Dtype::F32, sh_lt, &b_lt)?),
        ("latents_test", tensor_view(Dtype::F32, sh_lte, &b_lte)?),
        ("latents_eval", tensor_view(Dtype::F32, sh_lev, &b_lev)?),
        ("obs_train", tensor_view(Dtype::F32, sh_ot, &b_ot)?),
        ("obs_test", tensor_view(Dtype::F32, sh_ote, &b_ote)?),
        ("obs_eval", tensor_view(Dtype::F32, sh_oev, &b_oev)?),
        ("states_train", tensor_view(Dtype::I32, sh_st, &b_st)?),
        ("states_test", tensor_view(Dtype::I32, sh_ste, &b_ste)?),
        ("states_eval", tensor_view(Dtype::I32, sh_sev, &b_sev)?),
        ("q_true", tensor_view(Dtype::F32, sh_q, &b_q)?),
        ("pi_true", tensor_view(Dtype::F32, sh_pi, &b_pi)?),
    ];
    safetensors::serialize(tensors, &None).map_err(Into::into)
}

/// Round-trip load (`name`) for tests (`F32` tensors only).
pub fn load_tensor_f32(st_path: &std::path::Path, name: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = fs::read(st_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let tv = st.tensor(name)?;
    anyhow::ensure!(
        tv.dtype() == Dtype::F32,
        "tensor {:?}: expected F32, got {:?}",
        name,
        tv.dtype()
    );
    let floats: &[f32] = bytemuck::cast_slice(tv.data());
    Ok(floats.to_vec())
}

/// Round-trip load (`name`) for tests (`I32` tensors only).
pub fn load_tensor_i32(st_path: &std::path::Path, name: &str) -> anyhow::Result<Vec<i32>> {
    let bytes = fs::read(st_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let tv = st.tensor(name)?;
    anyhow::ensure!(
        tv.dtype() == Dtype::I32,
        "tensor {:?}: expected I32, got {:?}",
        name,
        tv.dtype()
    );
    let ints: &[i32] = bytemuck::cast_slice(tv.data());
    Ok(ints.to_vec())
}
