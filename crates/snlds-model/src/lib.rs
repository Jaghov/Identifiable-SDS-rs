pub mod cnn;
pub mod mlp;
pub mod model;
mod tests;

pub use cnn::{CnnDecoder, CnnDecoderConfig, CnnEncoder, CnnEncoderConfig};
pub use mlp::{Mlp, MlpConfig};
pub use model::{EncoderKind, ForwardOutput, SnldsConfig, VariationalSnlds};
