use burn::{
    config::Config,
    module::Module,
    nn::{
        activation::{Activation, ActivationConfig},
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

/// Two-hidden-layer MLP matching the Python `modules.MLP`.
///
/// Architecture: `Linear → activation → Linear → activation → Linear`
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub first_linear: Linear<B>,
    pub second_linear: Linear<B>,
    pub output_linear: Linear<B>,
    pub activation: Activation<B>,
}

/// Configuration for [`Mlp`].
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dim: usize,
    pub activation: ActivationConfig,
}

impl MlpConfig {
    /// Softplus MLP — used for transition networks.
    pub fn softplus(input_dim: usize, output_dim: usize, hidden_dim: usize) -> Self {
        use burn::nn::SoftplusConfig;
        Self {
            input_dim,
            output_dim,
            hidden_dim,
            activation: ActivationConfig::Softplus(SoftplusConfig { beta: 1.0 }),
        }
    }

    /// Leaky-ReLU MLP (negative_slope = 0.2) — used for encoder/decoder.
    pub fn leaky_relu(input_dim: usize, output_dim: usize, hidden_dim: usize) -> Self {
        use burn::nn::LeakyReluConfig;
        Self {
            input_dim,
            output_dim,
            hidden_dim,
            activation: ActivationConfig::LeakyRelu(LeakyReluConfig {
                negative_slope: 0.2,
            }),
        }
    }

    /// Initialise the module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            first_linear: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            second_linear: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            output_linear: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
            activation: self.activation.init(device),
        }
    }
}

impl<B: Backend> Mlp<B> {
    /// Forward pass: `[*, input_dim] → [*, output_dim]`.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let first_hidden = self.activation.forward(self.first_linear.forward(input));
        let second_hidden = self
            .activation
            .forward(self.second_linear.forward(first_hidden));
        self.output_linear.forward(second_hidden)
    }
}
