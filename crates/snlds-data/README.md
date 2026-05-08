# `snlds-data` (M1)

Synthetic **switching nonlinear dynamical system** sequence generation aligned with `identifiable-SDS`, plus **SafeTensors** export (`sequences.safetensors`, `metadata.json`). Parity focus: cosine / polynomial dynamics, leaky-ReLU emissions, pluggable **Markov transition** topology.

## Contents

- **`generate`** — [`GenConfig`](src/generate.rs), [`generate_train_test`](src/generate.rs), [`TrainTest`](src/generate.rs); vector vs **image** observations via [`ObservationKind`](src/generate.rs) (`Image { res }` requires `dim_latent == 2` and `dim_obs == res * res * 3`).
- **`transitions`** — [`TransitionPattern`](src/transitions.rs), [`get_trans_mat`](src/transitions.rs) for cyclic or caller-provided row-stochastic `Q`.
- **`io`** — manifest + tensor load/save, schema version.
- **`polynomial`**, **`render`** — dynamics helpers and RGB frame rendering for the image path.

Re-exports: `TransitionPattern`, `Manifest`, `MANIFEST_SCHEMA_VERSION`, etc. (see [`src/lib.rs`](src/lib.rs)).

## Dependencies

External: `ndarray`, `safetensors`, `serde`, `rand`, `anyhow`, … (see [`Cargo.toml`](Cargo.toml)).

Workspace crates: **none**.

## Binary

**`snlds-gen`** — CLI wrapper around `generate_train_test` + `save_train_test` (requires `cli` feature, on by default).

```sh
cargo run -p snlds-data --bin snlds-gen -- --help
```

For **image sequences**, configure `GenConfig` in Rust with `observation: ObservationKind::Image { res }` (the CLI currently pins defaults; see root README).

## See also

- Schema and tensors: [`../../docs/M1.md`](../../docs/M1.md)
- Repository overview: [`../../README.md`](../../README.md)
