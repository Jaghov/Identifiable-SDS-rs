//! Neural PCA (Li & Hooi 2022): block after Glow — [`PcaBatchNorm`] (beta=0) plus either
//! SVD [`PcaRotate`] or a learned [`HouseholderStack`]. See repo `docs/NEURAL_PCA.md` for rationale.

mod flatten;
mod geometry;
mod householder;
mod neural_pca;
mod pca_batchnorm;
mod pca_rotate;
mod prior;

pub use flatten::{flatten_zs, unflatten_zs};
pub use geometry::{
    default_glow_config_for_npca, flat_nhwc_rows_to_nchw, glow_flattened_latent_dim,
    glow_last_split_dim,
};
pub use glow_flow::prelude::CouplingType;
pub use householder::HouseholderStack;
pub use neural_pca::{NeuralPca, NeuralPcaConfig, NeuralPcaOutput};
pub use pca_batchnorm::PcaBatchNorm;
pub use pca_rotate::{PcaRotate, PcaSvdBackend, RotateOutput};
pub use prior::log_p_z_isotropic;
