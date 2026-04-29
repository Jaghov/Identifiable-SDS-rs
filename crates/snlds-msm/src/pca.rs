//! PCA reduction of observation tensors via [`linfa_reduction`].
//!
//! Mirrors the Python warm-start pipeline in `train_snlds.py`:
//! `PCA(n_components=dim_latent).fit_transform(obs.reshape(-1, dim_obs))`,
//! reshaped back to `[N, T, dim_latent]`.

use anyhow::{anyhow, Context};
use linfa::traits::{Fit, Transformer};
use linfa::Dataset;
use linfa_reduction::Pca;
use ndarray::{Array2, Array3};

/// Fit linear PCA on the flattened observations and return the reduced sequences.
///
/// - `obs`: shape `[num_sequences, seq_length, obs_dim]`
/// - `n_components`: target embedding size (typically `dim_latent`)
///
/// Returns an array of shape `[num_sequences, seq_length, n_components]`.
pub fn pca_fit_transform(obs: &Array3<f32>, n_components: usize) -> anyhow::Result<Array3<f32>> {
    let (num_sequences, seq_length, obs_dim) = obs.dim();
    if n_components == 0 {
        return Err(anyhow!("n_components must be > 0"));
    }
    if n_components > obs_dim {
        return Err(anyhow!(
            "n_components ({n_components}) must be <= obs_dim ({obs_dim})"
        ));
    }

    let flat_rows = num_sequences * seq_length;
    let obs_f64: Array2<f64> = obs
        .to_shape((flat_rows, obs_dim))
        .context("reshape obs to [N*T, obs_dim]")?
        .mapv(|value| value as f64);

    let dataset = Dataset::from(obs_f64);
    let model = Pca::params(n_components)
        .fit(&dataset)
        .map_err(|err| anyhow!("PCA fit failed: {err}"))?;
    let projected = model.transform(dataset);
    let projected_records = projected.records().to_owned();
    let projected_f32 = projected_records.mapv(|value| value as f32);

    if projected_f32.shape() != [flat_rows, n_components] {
        return Err(anyhow!(
            "PCA returned shape {:?}, expected [{flat_rows}, {n_components}]",
            projected_f32.shape()
        ));
    }

    let reshaped = projected_f32
        .into_shape_with_order((num_sequences, seq_length, n_components))
        .context("reshape PCA output back to [N, T, n_components]")?;

    Ok(reshaped)
}
