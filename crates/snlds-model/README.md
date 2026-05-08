# `snlds-model` (M3)

**`VariationalSnlds`** — Burn module implementing the variational SNLDS: recurrent / factored encoder, per-state dynamics, switching transitions, reconstruction ELBO pieces. Exposes **MLP** and **CNN** encoder/decoder paths for vector vs flat-RGB image observations.

Also includes **Neural PCA** (`npca` module): Glow + PCA BatchNorm + SVD rotation + non-isotropic latent prior.

## Contents

- **`model`** — [`VariationalSnlds`](src/model.rs), [`SnldsConfig`](src/model.rs), [`EncoderKind`](src/model.rs) (`Mlp` vs `Cnn { res }`), [`ForwardOutput`](src/model.rs).
- **`mlp`**, **`cnn`** — building blocks and config structs (`Mlp`, `CnnEncoder`, `CnnDecoder`, …).
- **`npca`** — [`NeuralPca`](src/npca/neural_pca.rs), [`NeuralPcaConfig`](src/npca/neural_pca.rs), [`PatchMode`](src/npca/neural_pca.rs), [`SigmaSchedule`](src/npca/neural_pca.rs), and PCA internals.

Library-only crate (no binary).

## Dependencies

- **`snlds-core`** — HMM / inference kernels.
- **`glow_flow`** — Glow stack from [`Glow-rs`](https://github.com/Jaghov/Glow-rs.git) (git dependency).
- **`burn`** `0.20` with `cpu` and `autodiff`.
- **`linfa-linalg`** + **`ndarray`** — CPU SVD path for Neural PCA rotation.

## Usage

Constructed via `SnldsConfig::new(dim_obs, dim_latent, hidden_dim, num_states)` (see tests and `snlds-train`). Training uses the **`ndarray`** autodiff backend from `snlds-train`, not necessarily the same CPU feature set as local unit tests.

```toml
snlds-model = { path = "../snlds-model" }
```

## See also

- Training entrypoint: [`../snlds-train/README.md`](../snlds-train/README.md)
- Repository overview: [`../../README.md`](../../README.md)
