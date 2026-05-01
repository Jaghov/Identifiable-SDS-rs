//! M1: synthetic data generation and SafeTensors IO (parity with `identifiable-SDS`).

pub mod generate;
pub mod io;
pub mod polynomial;
pub mod render;
pub mod transitions;

pub use generate::{generate_train_test, GenConfig, ObservationKind, SimulatorKind, TrainTest};
pub use io::{
    load_manifest, load_tensor_f32, load_tensor_i32, save_train_test, Manifest,
    MANIFEST_SCHEMA_VERSION,
};
pub use transitions::TransitionPattern;
