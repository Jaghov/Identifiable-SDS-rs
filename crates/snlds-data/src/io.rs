//! SafeTensors + JSON manifest export (see [docs/M1.md](../../../../docs/M1.md)).
//!
//! **`encode_safetensors`** holds per-tensor staging **`Vec<u8>`** buffers for one
//! **`safetensors::serialize`** call; **`TensorView`** borrows those slices (no **`Box::leak`**).

use crate::generate::TrainTest;
use anyhow::Context;
use ndarray::ArrayViewD;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;

pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

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
    let mj = serde_json::to_vec_pretty(manifest)?;
    fs::write(out_dir.join("metadata.json"), mj)?;
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
    let (sh_ot, b_ot) = pack_dyn_f32(tt.obs_train.view().into_dyn())?;
    let (sh_ote, b_ote) = pack_dyn_f32(tt.obs_test.view().into_dyn())?;
    let (sh_st, b_st) = pack_dyn_i32(tt.states_train.view().into_dyn())?;
    let (sh_ste, b_ste) = pack_dyn_i32(tt.states_test.view().into_dyn())?;

    let tensors: Vec<(&str, TensorView<'_>)> = vec![
        ("latents_train", tensor_view(Dtype::F32, sh_lt, &b_lt)?),
        ("latents_test", tensor_view(Dtype::F32, sh_lte, &b_lte)?),
        ("obs_train", tensor_view(Dtype::F32, sh_ot, &b_ot)?),
        ("obs_test", tensor_view(Dtype::F32, sh_ote, &b_ote)?),
        ("states_train", tensor_view(Dtype::I32, sh_st, &b_st)?),
        ("states_test", tensor_view(Dtype::I32, sh_ste, &b_ste)?),
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
