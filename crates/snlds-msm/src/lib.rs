//! M5: optional NeuralMSM warm-start for [`snlds_model::VariationalSnlds`].

pub mod msm;
pub mod pca;
pub mod transfer;

pub use msm::{NeuralMsm, NeuralMsmConfig};
pub use pca::pca_fit_transform;
pub use transfer::transfer_into_snlds;
