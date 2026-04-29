//! Load M1 SafeTensors splits into Burn tensors.

use anyhow::{anyhow, Context};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use ndarray::Array3;
use snlds_data::{load_manifest, load_tensor_f32, Manifest};
use std::path::Path;

/// Training observations loaded from disk.
pub struct ObsTensor<B: Backend> {
    /// Shape `[num_sequences, seq_length, obs_dim]`.
    pub obs: Tensor<B, 3>,
    pub manifest: Manifest,
}

/// Read `obs_train` as an [`ndarray::Array3`] together with the manifest.
///
/// Convenient for upstream pre-processing (e.g. PCA in the M5 warm-start path)
/// before turning the data into a Burn tensor.
pub fn load_train_obs_array(data_dir: &Path) -> anyhow::Result<(Array3<f32>, Manifest)> {
    let manifest = load_manifest(data_dir.join("metadata.json"))
        .with_context(|| format!("load manifest from {:?}", data_dir))?;
    let st_path = data_dir.join("sequences.safetensors");
    let obs_flat = load_tensor_f32(&st_path, "obs_train")
        .with_context(|| format!("load obs_train from {:?}", st_path))?;
    let shape = (manifest.num_samples, manifest.seq_length, manifest.dim_obs);
    let expected_len = shape.0 * shape.1 * shape.2;
    if obs_flat.len() != expected_len {
        return Err(anyhow!(
            "obs_train length {} does not match manifest {:?} = {}",
            obs_flat.len(),
            shape,
            expected_len,
        ));
    }
    let array = Array3::from_shape_vec(shape, obs_flat).context("reshape obs_train")?;
    Ok((array, manifest))
}

/// Read `obs_train` from `<data_dir>/sequences.safetensors` and the manifest
/// from `<data_dir>/metadata.json`, then build a Burn `[N, T, D]` tensor.
pub fn load_train_obs<B: Backend>(
    data_dir: &Path,
    device: &B::Device,
) -> anyhow::Result<ObsTensor<B>> {
    let manifest = load_manifest(data_dir.join("metadata.json"))
        .with_context(|| format!("load manifest from {:?}", data_dir))?;

    let st_path = data_dir.join("sequences.safetensors");
    let obs_flat = load_tensor_f32(&st_path, "obs_train")
        .with_context(|| format!("load obs_train from {:?}", st_path))?;

    let num_sequences = manifest.num_samples;
    let seq_length = manifest.seq_length;
    let obs_dim = manifest.dim_obs;
    let expected_len = num_sequences * seq_length * obs_dim;
    if obs_flat.len() != expected_len {
        return Err(anyhow!(
            "obs_train length {} does not match manifest [{}, {}, {}] = {}",
            obs_flat.len(),
            num_sequences,
            seq_length,
            obs_dim,
            expected_len,
        ));
    }

    let shape = [num_sequences, seq_length, obs_dim];
    let tensor_data = TensorData::new(obs_flat, shape);
    let obs = Tensor::<B, 3>::from_data(tensor_data, device);

    Ok(ObsTensor { obs, manifest })
}
