# SNLDS — Burn port

Rust port of [identifiable-SDS](identifiable-SDS/README.md) using [Burn](https://burn.dev).

## Requirements

- Rust stable (1.80+)

## Build & test

```sh
cargo build --workspace
cargo test --workspace
```

## Lint

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Data generation (M1)

The **`snlds-data`** crate synthesizes cosine / polynomial SDS-style sequences and writes **`sequences.safetensors`** plus **`metadata.json`** (see [docs/M1.md](docs/M1.md)). Latents and observations use **F32** tensors; discrete state sequences use **`I32`** (**schema v2** in M1). SafeTensors encoding uses scoped in-memory staging only (**no process-global allocations** left behind per export).

```sh
cargo run -p snlds-data --bin snlds-gen -- --seed 42 --dim-obs 2 --dim-latent 2 --num-states 3 \
  --seq-length 64 --num-samples 32 --data-type cosine --out ./out/run1
# or --data-type poly --degree 3
```

Rust API: [`snlds_data::generate_train_test`](crates/snlds-data/src/generate.rs), [`save_train_test`](crates/snlds-data/src/io.rs), and [`load_manifest`](crates/snlds-data/src/io.rs) for **`metadata.json`**.

## Backend

`snlds-core` / M0 use **Burn 0.20** with the **`cpu`** feature (CubeCL-based CPU backend in this toolchain). GPU paths may follow later — see [docs/PRD-burn-port.md](docs/PRD-burn-port.md).
