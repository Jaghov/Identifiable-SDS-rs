pub mod cnn;
pub mod mlp;
pub mod model;
mod tests;

pub use cnn::{validate_cnn_res, CnnDecoder, CnnDecoderConfig, CnnEncoder, CnnEncoderConfig};
pub use mlp::{Mlp, MlpConfig};
pub use model::{EncoderKind, ForwardOutput, SnldsConfig, VariationalSnlds};
