//! M4: training CLI for `VariationalSnlds`.
//!
//! Loads the SafeTensors splits produced by `snlds-data` (M1), runs minibatch Adam
//! training of a [`snlds_model::VariationalSnlds`] on the autodiff CPU backend,
//! and saves checkpoints with [`burn::record::CompactRecorder`].

pub mod data;
pub mod train;
pub mod warm_start;

pub use data::{load_train_obs, load_train_obs_array, ObsTensor};
pub use train::{build_model_config, train, train_with_model, TrainConfig};
pub use warm_start::{run_warm_start, MsmWarmStartConfig};
