//! M4: training CLI for `VariationalSnlds`.
//!
//! Loads the SafeTensors splits produced by `snlds-data` (M1), runs minibatch Adam
//! training of a [`snlds_model::VariationalSnlds`] on the autodiff CPU backend,
//! and saves checkpoints with [`burn::record::CompactRecorder`].

pub mod checkpoint_recon;
pub mod config_file;
pub mod data;
pub mod snapshot;
pub mod train;
pub mod training_log;
pub mod warm_start;

pub mod flow_train;

pub use config_file::{
    load_train_config_file, resolve_encoder_kind, resolve_train, EncoderCli, GlowCouplingCli,
    NpcaRotationCli, NpcaRotationFile, ResolvedMode, ResolvedTrain, TrainArgs, TrainCli,
    TrainConfigFile,
};
pub use data::{
    load_train_obs, load_train_obs_array, ObsTensor, SequenceBatch, SequenceBatcher,
    SequenceDataset, SequenceItem,
};
pub use snapshot::{
    FlowSnldsSnapshotMeta, TrainSnapshot, DEFAULT_OBS_NOISE_VAR, TRAIN_SNAPSHOT_FILENAME,
    TRAIN_SNAPSHOT_SCHEMA_VERSION,
};
pub use train::{build_model_config, train, train_with_model, TrainConfig};
pub use warm_start::{run_warm_start, MsmWarmStartConfig};

pub use flow_train::{
    build_flow_config, train_flow_from_dataset, train_flow_with_dataset, FlowEpochStats,
    FlowTrainConfig,
};
