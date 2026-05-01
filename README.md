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

## Backend

`snlds-core` / M0 use **Burn 0.20** with the **`cpu`** feature (CubeCL-based CPU backend in this toolchain). `snlds-train` / `snlds-msm` / `snlds-eval` use the **`ndarray`** Burn backend (with `autodiff` for training). GPU paths may follow later — see [docs/PRD-burn-port.md](docs/PRD-burn-port.md).

---

## Workspace crates

The workspace is split by milestone. Library crates expose Rust APIs; the four crates with binaries (`snlds-data`, `snlds-viz`, `snlds-train`, `snlds-eval`) ship a CLI for end-to-end use.

A typical end-to-end pipeline:

```sh
cargo run -p snlds-data  --bin snlds-gen   -- --out ./out/run1 --seed 42 --num-states 3 --seq-length 64 --num-samples 32
cargo run -p snlds-viz   --bin snlds-viz   -- --input ./out/run1 --output ./out/run1/gt.rrd
cargo run -p snlds-train --bin snlds-train -- --data-dir ./out/run1 --output-dir ./out/run1/ckpt --epochs 50
cargo run -p snlds-eval  --bin snlds-eval  -- --data-dir ./out/run1 --checkpoint ./out/run1/ckpt/snlds_final.bin --output ./out/run1/inferred.rrd
```

### `snlds-core`

Burn primitives + HMM kernels used by the model crate. Library only.

```rust
use burn::backend::{Cpu, Autodiff};
use snlds_core::hmm; // forward / backward / posterior kernels

type B = Autodiff<Cpu<f32>>;
```

### `snlds-data` — synthetic generation + SafeTensors IO

Generates cosine / polynomial SDS-style sequences and writes `sequences.safetensors` + `metadata.json. Latents and observations are **F32**; discrete states are **I32**.

CLI (`snlds-gen`):

```sh
cargo run -p snlds-data --bin snlds-gen -- \
  --seed 42 --dim-obs 2 --dim-latent 2 --num-states 3 \
  --seq-length 64 --num-samples 32 --data-type cosine \
  --out ./out/run1
# or --data-type poly --degree 3
```

Library API:

```rust
use snlds_data::{
    generate_train_test, save_train_test, load_manifest,
    GenConfig, Manifest, SimulatorKind, TransitionPattern,
};

let cfg = GenConfig {
    seed: 42,
    num_states: 3,
    seq_length: 64,
    num_samples: 32,
    kind: SimulatorKind::Cosine,
    transition: TransitionPattern::Cyclic { self_prob: 0.9 },
    ..GenConfig::default()
};
let tt = generate_train_test(&cfg)?;
save_train_test("./out/run1".as_ref(), &tt, &Manifest { /* ... */ })?;
let manifest = load_manifest("./out/run1/metadata.json")?;
```

For a custom topology pass `TransitionPattern::Provided(matrix)`; `get_trans_mat` validates row-stochasticity once per generation and threads the same `Q` into both `roll_sequences` and `q_true`.

### `snlds-viz` — ground-truth visualisation

Logs ground-truth sequences, the true `Q`, and Figure-6-style state strips into Rerun.

CLI (`snlds-viz`):

```sh
cargo run -p snlds-viz --bin snlds-viz -- \
  --input ./out/run1 --sequences 5 --split train \
  --output ./out/run1/gt.rrd
# add --spawn for the live viewer, or --render for image-channel previews
```

Library API:

```rust
use rerun::RecordingStreamBuilder;
use snlds_viz::{log_transition_matrix, log_state_strip, log_reconstructions};

let rec = RecordingStreamBuilder::new("snlds").save("./out/run1/gt.rrd")?;
log_transition_matrix(&rec, "snlds/markov/q_true", &tt.q_true.view());
log_state_strip(&rec, "snlds/state/strip_true", tt.states_train.row(0));
```

### `snlds-model` — `VariationalSnlds` + encoders/decoders

Library only. Provides `VariationalSnlds`, the MLP/CNN encoders, and `SnldsConfig`.

```rust
use burn::backend::{Cpu, Autodiff};
use snlds_model::{SnldsConfig, VariationalSnlds, EncoderKind};

type B = Autodiff<Cpu<f32>>;
// SnldsConfig::new(dim_obs, dim_latent, hidden_dim, num_states); EncoderKind::Mlp by default.
let cfg = SnldsConfig::new(2, 2, 64, 3).with_kind(EncoderKind::Mlp);
let model: VariationalSnlds<B> = cfg.init(&Default::default());
let out = model.forward(obs_tensor, /* beta */ 1.0, /* obs_noise_var */ 5e-4, /* temperature */ 1.0);
```

### `snlds-train` — training CLI + library

Loads data splits, trains on the autodiff `ndarray` backend, writes `CompactRecorder` checkpoints + `train_config.json`.

CLI (`snlds-train`):

```sh
cargo run -p snlds-train --bin snlds-train -- \
  --data-dir ./out/run1 --output-dir ./out/run1/ckpt \
  --epochs 100 --batch-size 32 --lr 3e-4 \
  --beta 1.0 --temperature 1.0 --grad-clip 1.0 \
  --hidden-dim 64 --obs-noise-var 5e-4 \
  --checkpoint-every 10
```

Library API:

```rust
use snlds_train::{train, TrainConfig, load_train_obs};

let cfg = TrainConfig { /* fields mirror CLI flags */ ..Default::default() };
let obs = load_train_obs::<MyBackend>(&cfg.data_dir, &device)?;
train::<MyBackend>(&cfg, obs, &device)?;
```

Optional NeuralMSM warm start is exposed via `--msm-init` (CLI) or `snlds_train::run_warm_start` (library); see `MsmWarmStartConfig`.

### `snlds-msm` — NeuralMSM warm-start

Library: linfa-reduction PCA → simplified `NeuralMsm` → parameter transfer into a `VariationalSnlds`.

```rust
use snlds_msm::{pca_fit_transform, NeuralMsm, NeuralMsmConfig, transfer_into_snlds};

let reduced = pca_fit_transform(&obs_train, /* n_components */ cfg.dim_latent)?;
let msm_cfg = NeuralMsmConfig::new(cfg.dim_latent, cfg.num_states);
let msm: NeuralMsm<B> = msm_cfg.init(&device);
// ... fit msm on `reduced` ...
let snlds = transfer_into_snlds(msm, snlds_model)?;
```

`snlds-train --msm-init` wires this into the training loop end-to-end.

### `snlds-eval` — inference + Rerun logging

Loads a `snlds-train` checkpoint, runs forward inference on `obs_train`, and logs the inferred `Q`, posteriors `γ`, state strips, and reconstructions to Rerun. Reads `train_config.json` next to the checkpoint automatically; CLI flags override.

CLI (`snlds-eval`):

```sh
cargo run -p snlds-eval --bin snlds-eval -- \
  --data-dir ./out/run1 --checkpoint ./out/run1/ckpt/snlds_final.bin \
  --output ./out/run1/inferred.rrd --sequences 5
# add --spawn for the live viewer
```

Library API:

```rust
use snlds_eval::{run_eval, EvalConfig};

let cfg = EvalConfig {
    data_dir: "./out/run1".into(),
    checkpoint: "./out/run1/ckpt/snlds_final.bin".into(),
    output: "./out/run1/inferred.rrd".into(),
    spawn: false,
    sequences: 5,
    // optional per-field overrides for hidden_dim / temperature / obs_noise_var / beta
};
run_eval::<MyBackend>(&cfg, &device)?;
```

A typical workflow logs ground-truth (`snlds-viz`) and inferred (`snlds-eval`) into the same `.rrd` for side-by-side inspection.
