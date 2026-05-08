//! Load M1 SafeTensors splits into Burn tensors.
//!
//! Supports both single-directory datasets (`sequences.safetensors` + `metadata.json`)
//! and sharded datasets (`shard_000/`, `shard_001/`, …) produced by `snlds-gen --num-shards`.
//!
//! For large datasets, [`SequenceDataset`] memory-maps the SafeTensors files and serves
//! individual sequences on demand via Burn's [`Dataset`] trait. Pair with
//! [`SequenceBatcher`] and [`DataLoaderBuilder`] for streaming batch-level I/O.

use anyhow::{anyhow, Context};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use memmap2::Mmap;
use ndarray::{Array3, Axis};
use safetensors::SafeTensors;
use snlds_data::{load_manifest, load_tensor_f32, Manifest};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Training observations loaded from disk.
pub struct ObsTensor<B: Backend> {
    /// Shape `[num_sequences, seq_length, obs_dim]`.
    pub obs: Tensor<B, 3>,
    pub manifest: Manifest,
}

// ---------------------------------------------------------------------------
// Memory-mapped dataset (Burn `Dataset` trait)
// ---------------------------------------------------------------------------

/// A single training sequence: one `[T, D]` slice of f32 pixel data.
#[derive(Clone, Debug)]
pub struct SequenceItem {
    /// Flat f32 values, length = `seq_length * obs_dim`.
    pub data: Vec<f32>,
}

/// One memory-mapped SafeTensors shard, holding the byte offset into `obs_train`.
struct MappedShard {
    #[allow(dead_code)]
    mmap: Mmap,
    /// Start of `obs_train` tensor data within the mmap.
    obs_offset: usize,
    /// Bytes per sequence = `seq_length * obs_dim * 4`.
    seq_stride: usize,
}

/// Memory-mapped dataset across one or more SafeTensors shard files.
///
/// Implements Burn's [`Dataset`] trait: `get(i)` reads exactly one `[T, D]`
/// sequence from the OS page cache — no full-file load required.
///
/// Cheaply cloneable (inner data is `Arc`-shared) so it can be passed to
/// [`DataLoaderBuilder::build`] each epoch.
#[derive(Clone)]
pub struct SequenceDataset {
    inner: Arc<SequenceDatasetInner>,
    pub manifest: Manifest,
}

struct SequenceDatasetInner {
    shards: Vec<MappedShard>,
    cumulative: Vec<usize>,
}

impl SequenceDataset {
    /// Open the training split (`obs_train`) from a dataset directory.
    pub fn open(data_dir: &Path) -> anyhow::Result<Self> {
        Self::open_split(data_dir, "obs_train")
    }

    /// Open the test/validation split (`obs_test`) from a dataset directory.
    /// Returns `Ok(None)` if `obs_test` is not present in the SafeTensors file.
    pub fn open_val(data_dir: &Path) -> anyhow::Result<Option<Self>> {
        match Self::open_split(data_dir, "obs_test") {
            Ok(ds) => Ok(Some(ds)),
            Err(e) => {
                let msg = format!("{e}");
                if msg.contains("not found") {
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn open_split(data_dir: &Path, tensor_name: &str) -> anyhow::Result<Self> {
        if let Some(shard_dirs) = discover_shards(data_dir) {
            Self::open_shards(&shard_dirs, tensor_name)
        } else {
            Self::open_shards(&[data_dir.to_path_buf()], tensor_name)
        }
    }

    fn open_shards(dirs: &[PathBuf], tensor_name: &str) -> anyhow::Result<Self> {
        let mut shards = Vec::with_capacity(dirs.len());
        let mut cumulative = Vec::with_capacity(dirs.len());
        let mut combined_manifest: Option<Manifest> = None;
        let mut total_n = 0usize;

        for dir in dirs {
            let manifest = load_manifest(dir.join("metadata.json"))
                .with_context(|| format!("load manifest from {:?}", dir))?;

            let st_path = dir.join("sequences.safetensors");
            let file = std::fs::File::open(&st_path)
                .with_context(|| format!("open {:?}", st_path))?;
            // SAFETY: we don't modify the file while it's mapped.
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mmap {:?}", st_path))?;

            let (header_len, n) = {
                let st = SafeTensors::deserialize(&mmap)
                    .with_context(|| format!("parse safetensors header {:?}", st_path))?;
                let tv = st.tensor(tensor_name)
                    .with_context(|| format!("tensor {} not found in {:?}", tensor_name, st_path))?;
                let data_ptr = tv.data().as_ptr();
                let offset = unsafe { data_ptr.offset_from(mmap.as_ptr()) as usize };
                let n_seqs = tv.shape()[0];
                (offset, n_seqs)
            };

            let seq_stride = manifest.seq_length * manifest.dim_obs * std::mem::size_of::<f32>();

            if let Some(ref cm) = combined_manifest {
                anyhow::ensure!(
                    cm.seq_length == manifest.seq_length
                        && cm.dim_obs == manifest.dim_obs
                        && cm.num_states == manifest.num_states,
                    "shard {:?} has incompatible manifest",
                    dir,
                );
            } else {
                combined_manifest = Some(manifest);
            }

            total_n += n;
            cumulative.push(total_n);
            shards.push(MappedShard {
                mmap,
                obs_offset: header_len,
                seq_stride,
            });
        }

        let mut manifest =
            combined_manifest.ok_or_else(|| anyhow!("no non-empty shards found"))?;
        manifest.num_samples = total_n;

        eprintln!(
            "Opened {} shard(s): {} total sequences [{}] (mmap, batch-level streaming)",
            shards.len(),
            total_n,
            tensor_name,
        );

        Ok(Self {
            inner: Arc::new(SequenceDatasetInner {
                shards,
                cumulative,
            }),
            manifest,
        })
    }
}

impl Dataset<SequenceItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceItem> {
        let inner = &self.inner;
        if index >= self.len() {
            return None;
        }
        let shard_idx = inner.cumulative.partition_point(|&c| c <= index);
        let shard = &inner.shards[shard_idx];
        let local_idx = if shard_idx == 0 {
            index
        } else {
            index - inner.cumulative[shard_idx - 1]
        };
        let start = shard.obs_offset + local_idx * shard.seq_stride;
        let end = start + shard.seq_stride;
        let bytes = &shard.mmap[start..end];
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        Some(SequenceItem {
            data: floats.to_vec(),
        })
    }

    fn len(&self) -> usize {
        *self.inner.cumulative.last().unwrap_or(&0)
    }
}

// ---------------------------------------------------------------------------
// Batcher: SequenceItem → batched Tensor<B, 3>
// ---------------------------------------------------------------------------

/// Batch of observation sequences for the flow model.
#[derive(Clone, Debug)]
pub struct SequenceBatch<B: Backend> {
    /// `[batch_size, seq_length, obs_dim]`.
    pub obs: Tensor<B, 3>,
}

/// Collates [`SequenceItem`]s into a single `[N, T, D]` tensor on device.
#[derive(Clone)]
pub struct SequenceBatcher {
    pub seq_length: usize,
    pub obs_dim: usize,
}

impl<B: Backend> Batcher<B, SequenceItem, SequenceBatch<B>> for SequenceBatcher {
    fn batch(&self, items: Vec<SequenceItem>, device: &B::Device) -> SequenceBatch<B> {
        let n = items.len();
        let total_len = n * self.seq_length * self.obs_dim;
        let mut flat = Vec::with_capacity(total_len);
        for item in &items {
            flat.extend_from_slice(&item.data);
        }
        let shape = [n, self.seq_length, self.obs_dim];
        let tensor_data = TensorData::new(flat, shape);
        let obs = Tensor::<B, 3>::from_data(tensor_data, device);
        SequenceBatch { obs }
    }
}

/// Discover shard subdirectories (`shard_000/`, `shard_001/`, …) under `data_dir`.
/// Returns `None` if this is a plain (non-sharded) dataset directory.
fn discover_shards(data_dir: &Path) -> Option<Vec<std::path::PathBuf>> {
    let mut shards = Vec::new();
    let entries = std::fs::read_dir(data_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("shard_") && entry.path().is_dir() {
            shards.push(entry.path());
        }
    }
    if shards.is_empty() {
        return None;
    }
    shards.sort();
    Some(shards)
}

/// Load a single shard's obs_train as a flat `Vec<f32>` and its manifest.
fn load_single_obs(dir: &Path) -> anyhow::Result<(Vec<f32>, Manifest)> {
    let manifest = load_manifest(dir.join("metadata.json"))
        .with_context(|| format!("load manifest from {:?}", dir))?;
    let st_path = dir.join("sequences.safetensors");
    let obs_flat = load_tensor_f32(&st_path, "obs_train")
        .with_context(|| format!("load obs_train from {:?}", st_path))?;
    let expected = manifest.num_samples * manifest.seq_length * manifest.dim_obs;
    if obs_flat.len() != expected {
        return Err(anyhow!(
            "obs_train length {} != manifest expectation {} in {:?}",
            obs_flat.len(),
            expected,
            dir,
        ));
    }
    Ok((obs_flat, manifest))
}

/// Read `obs_train` as an [`ndarray::Array3`] together with the manifest.
///
/// Convenient for upstream pre-processing (e.g. PCA in the M5 warm-start path)
/// before turning the data into a Burn tensor.
///
/// Auto-detects sharded datasets: if `data_dir` contains `shard_*/` subdirectories
/// they are loaded and concatenated along axis 0 (sequences).
pub fn load_train_obs_array(data_dir: &Path) -> anyhow::Result<(Array3<f32>, Manifest)> {
    if let Some(shards) = discover_shards(data_dir) {
        load_train_obs_array_sharded(&shards)
    } else {
        load_train_obs_array_single(data_dir)
    }
}

fn load_train_obs_array_single(data_dir: &Path) -> anyhow::Result<(Array3<f32>, Manifest)> {
    let (obs_flat, manifest) = load_single_obs(data_dir)?;
    let shape = (manifest.num_samples, manifest.seq_length, manifest.dim_obs);
    let array = Array3::from_shape_vec(shape, obs_flat).context("reshape obs_train")?;
    Ok((array, manifest))
}

fn load_train_obs_array_sharded(
    shard_dirs: &[std::path::PathBuf],
) -> anyhow::Result<(Array3<f32>, Manifest)> {
    let mut arrays = Vec::with_capacity(shard_dirs.len());
    let mut combined_manifest: Option<Manifest> = None;
    let mut total_n = 0usize;

    for dir in shard_dirs {
        let (obs_flat, manifest) = load_single_obs(dir)?;
        if manifest.num_samples == 0 {
            continue;
        }
        let shape = (manifest.num_samples, manifest.seq_length, manifest.dim_obs);
        let array = Array3::from_shape_vec(shape, obs_flat)
            .with_context(|| format!("reshape obs_train from {:?}", dir))?;
        total_n += manifest.num_samples;

        if let Some(ref cm) = combined_manifest {
            anyhow::ensure!(
                cm.seq_length == manifest.seq_length
                    && cm.dim_obs == manifest.dim_obs
                    && cm.num_states == manifest.num_states,
                "shard {:?} has incompatible manifest (seq_length/dim_obs/num_states mismatch)",
                dir,
            );
        } else {
            combined_manifest = Some(manifest);
        }
        arrays.push(array);
    }

    let mut manifest = combined_manifest.ok_or_else(|| anyhow!("no non-empty shards found"))?;
    manifest.num_samples = total_n;

    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let combined =
        ndarray::concatenate(Axis(0), &views).context("concatenate shards along axis 0")?;

    eprintln!(
        "Loaded {} shards: {} total sequences",
        shard_dirs.len(),
        total_n
    );
    Ok((combined, manifest))
}

/// Read `obs_train` from `<data_dir>/sequences.safetensors` and the manifest
/// from `<data_dir>/metadata.json`, then build a Burn `[N, T, D]` tensor.
///
/// Auto-detects sharded datasets: if `data_dir` contains `shard_*/` subdirectories
/// they are loaded and concatenated along axis 0 (sequences).
pub fn load_train_obs<B: Backend>(
    data_dir: &Path,
    device: &B::Device,
) -> anyhow::Result<ObsTensor<B>> {
    if let Some(shards) = discover_shards(data_dir) {
        load_train_obs_sharded(&shards, device)
    } else {
        load_train_obs_single(data_dir, device)
    }
}

fn load_train_obs_single<B: Backend>(
    data_dir: &Path,
    device: &B::Device,
) -> anyhow::Result<ObsTensor<B>> {
    let (obs_flat, manifest) = load_single_obs(data_dir)?;
    let shape = [manifest.num_samples, manifest.seq_length, manifest.dim_obs];
    let tensor_data = TensorData::new(obs_flat, shape);
    let obs = Tensor::<B, 3>::from_data(tensor_data, device);
    Ok(ObsTensor { obs, manifest })
}

fn load_train_obs_sharded<B: Backend>(
    shard_dirs: &[std::path::PathBuf],
    device: &B::Device,
) -> anyhow::Result<ObsTensor<B>> {
    let mut tensors = Vec::with_capacity(shard_dirs.len());
    let mut combined_manifest: Option<Manifest> = None;
    let mut total_n = 0usize;

    for dir in shard_dirs {
        let (obs_flat, manifest) = load_single_obs(dir)?;
        if manifest.num_samples == 0 {
            continue;
        }
        let shape = [manifest.num_samples, manifest.seq_length, manifest.dim_obs];
        let tensor_data = TensorData::new(obs_flat, shape);
        let t = Tensor::<B, 3>::from_data(tensor_data, device);
        total_n += manifest.num_samples;

        if let Some(ref cm) = combined_manifest {
            anyhow::ensure!(
                cm.seq_length == manifest.seq_length
                    && cm.dim_obs == manifest.dim_obs
                    && cm.num_states == manifest.num_states,
                "shard {:?} has incompatible manifest",
                dir,
            );
        } else {
            combined_manifest = Some(manifest);
        }
        tensors.push(t);
    }

    let mut manifest = combined_manifest.ok_or_else(|| anyhow!("no non-empty shards found"))?;
    manifest.num_samples = total_n;
    let obs = Tensor::cat(tensors, 0);

    eprintln!(
        "Loaded {} shards: {} total sequences",
        shard_dirs.len(),
        total_n
    );
    Ok(ObsTensor { obs, manifest })
}
