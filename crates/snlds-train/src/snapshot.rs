//! Persisted training-time hyperparameters that downstream tools (e.g. `snlds-eval`)
//! need in order to reproduce the model layout and ELBO numbers.
//!
//! The snapshot is written to `<output_dir>/train_config.json` once at the start
//! of a training run and remains stable for all checkpoints in that directory.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use snlds_model::EncoderKind;
use std::path::Path;

/// Default observation noise variance used in the reconstruction term of the ELBO.
///
/// Matches the canonical Python value (`var = 5e-4` in `identifiable-SDS`); promoted
/// from a private constant so callers (`snlds-eval`, tests) can refer to it by name
/// instead of re-typing the literal.
pub const DEFAULT_OBS_NOISE_VAR: f32 = 5e-4;

/// Filename written to `output_dir` so checkpoints carry their training context.
pub const TRAIN_SNAPSHOT_FILENAME: &str = "train_config.json";

/// Bump when the snapshot schema gains/loses fields. `2` adds `kind` for the
/// M6 encoder/decoder selector. v1 snapshots are intentionally not loadable —
/// no production checkpoints exist yet.
pub const TRAIN_SNAPSHOT_SCHEMA_VERSION: u32 = 2;

/// Subset of [`crate::TrainConfig`] that downstream tools need to recover.
///
/// Excludes IO paths, RNG seed, optimizer-only flags (`grad_clip`, `learning_rate`,
/// `checkpoint_every`), and resume settings since none of those affect inference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainSnapshot {
    pub schema_version: u32,
    pub hidden_dim: usize,
    pub beta: f32,
    pub temperature: f32,
    pub obs_noise_var: f32,
    /// Encoder/decoder family the run used. Required (no implicit default).
    pub kind: EncoderKind,
}

impl TrainSnapshot {
    /// Persist `self` as `<output_dir>/train_config.json` (pretty-printed JSON).
    pub fn save(&self, output_dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("create snapshot dir {output_dir:?}"))?;
        let path = output_dir.join(TRAIN_SNAPSHOT_FILENAME);
        let bytes = serde_json::to_vec_pretty(self).context("serialize TrainSnapshot")?;
        std::fs::write(&path, bytes).with_context(|| format!("write snapshot {path:?}"))
    }

    /// Read a snapshot from `<dir>/train_config.json`.
    pub fn load_from_dir(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join(TRAIN_SNAPSHOT_FILENAME);
        let bytes = std::fs::read(&path).with_context(|| format!("read snapshot {path:?}"))?;
        let snapshot: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse snapshot {path:?}"))?;
        Ok(snapshot)
    }

    /// Convenience: read the snapshot that lives next to `checkpoint_path`.
    pub fn load_for_checkpoint(checkpoint_path: &Path) -> anyhow::Result<Self> {
        let dir = checkpoint_path.parent().ok_or_else(|| {
            anyhow::anyhow!("checkpoint path {checkpoint_path:?} has no parent directory")
        })?;
        Self::load_from_dir(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_with_cnn_kind() {
        let original = TrainSnapshot {
            schema_version: TRAIN_SNAPSHOT_SCHEMA_VERSION,
            hidden_dim: 32,
            beta: 1.0,
            temperature: 1.0,
            obs_noise_var: 5e-4,
            kind: EncoderKind::Cnn { res: 16 },
        };
        let bytes = serde_json::to_vec(&original).expect("serialize");
        let parsed: TrainSnapshot = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(original, parsed);
    }
}
