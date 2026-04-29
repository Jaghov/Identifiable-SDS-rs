//! M4: training CLI for `VariationalSnlds`.
//!
//! Loads the SafeTensors splits produced by `snlds-data` (M1), runs minibatch Adam
//! training of a [`snlds_model::VariationalSnlds`] on the autodiff CPU backend,
//! and saves checkpoints with [`burn::record::CompactRecorder`].

pub mod data;
pub mod train;

pub use data::{load_train_obs, ObsTensor};
pub use train::{train, TrainConfig};
