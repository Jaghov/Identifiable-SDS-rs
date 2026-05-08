# SNLDS — Burn port

Rust port of [identifiable-SDS](identifiable-SDS/README.md) using [Burn](https://burn.dev).

## Requirements

- Rust stable (1.80+)
- Optional: [Rerun viewer](https://www.rerun.io/docs/getting-started/installing-viewer) (or `rerun --help`) to open `.rrd` logs from `snlds-viz` / `snlds-eval`

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

## See results (end-to-end)

**Training** prints per-epoch diagnostics to the terminal. **Artifacts:** `metadata.json` + `sequences.safetensors` in the data dir; `train_config.json` and `*.mpk` checkpoints in the train output dir. **Rerun:** log ground truth with `snlds-viz`, inferred posteriors / reconstructions with `snlds-eval`, then open the `.rrd` files in the [Rerun viewer](https://www.rerun.io/docs/getting-started/installing-viewer) (or pass `--spawn` where supported).

**Checkpoint names** (Burn `CompactRecorder`):

| Train mode | Final weights (after `--epochs N`) |
|------------|-------------------------------------|
| `VariationalSnlds` (default) | `checkpoint_XXXX.mpk` with `XXXX = N - 1` (zero-padded), e.g. `checkpoint_0019` for `N = 20`. |
| `--neural-pca` | `npca_checkpoint_XXXX.mpk` (Neural PCA only; **not** for `snlds-eval`). |
| `--flow-snlds` | `flow_checkpoint_XXXX.mpk` |

`snlds-eval` expects a **VariationalSnlds** or **FlowSNLDS** checkpoint; use the matching path above.

### 1) Vector observations (default `snlds-gen`)

```sh
cargo run -p snlds-data --bin snlds-gen -- \
  --out ./out/vec --seed 42 --num-states 3 --seq-length 64 --num-samples 128

cargo run -p snlds-viz --bin snlds-viz -- \
  --input ./out/vec --output ./out/vec/gt.rrd --sequences 5

cargo run -p snlds-train --bin snlds-train -- \
  --data-dir ./out/vec --output-dir ./out/vec/ckpt \
  --epochs 20 --batch-size 32

cargo run -p snlds-eval --bin snlds-eval -- \
  --data-dir ./out/vec --checkpoint ./out/vec/ckpt/checkpoint_0019.mpk \
  --output ./out/vec/inferred.rrd --sequences 5
```

Open `gt.rrd` and `inferred.rrd` in Rerun (or merge streams as you prefer).

### 2) Bouncing ball – RGB frames (`--observation image`)

 Simulator uses **2-D latent** ball dynamics and **`--num-states`** for the discrete Markov chain; **`dim_obs` / `dim_latent`** are fixed by `--res` (`dim_obs = 3·res²`, `dim_latent = 2`).

```sh
cargo run -p snlds-data --bin snlds-gen -- \
  --observation image --res 32 --num-states 3 \
  --seq-length 64 --num-samples 256 --seed 1 --out ./out/ball

cargo run -p snlds-viz --bin snlds-viz -- \
  --input ./out/ball --output ./out/ball/gt.rrd --sequences 5

cargo run -p snlds-train --bin snlds-train -- \
  --data-dir ./out/ball --output-dir ./out/ball/ckpt \
  --encoder cnn --res 32 --epochs 20 --batch-size 16

cargo run -p snlds-eval --bin snlds-eval -- \
  --data-dir ./out/ball --checkpoint ./out/ball/ckpt/checkpoint_0019.mpk \
  --output ./out/ball/inferred.rrd --sequences 5
```

### 3) Neural PCA or FlowSNLDS (`snlds-train` CLI modes)

Pick **`--neural-pca`** or **`--flow-snlds`** (or set `mode` in `--config` TOML). Optional Glow coupling: `--npca-glow-coupling affine` (default) or `additive`. See [docs/FLOW_SNLDS.md](docs/FLOW_SNLDS.md) for the Flow joint objective.

**Neural PCA only** (density on images; checkpoints are `npca_checkpoint_*.mpk`):

```sh
cargo run -p snlds-train --bin snlds-train -- \
  --data-dir ./out/ball --output-dir ./out/ball/npca-ckpt \
  --neural-pca --res 32 --epochs 10 --batch-size 16 \
  --npca-glow-levels 2 --npca-glow-steps 2 --npca-glow-hidden 16 \
  --npca-glow-coupling affine
```

**FlowSNLDS** (Neural PCA + switching head; checkpoints `flow_checkpoint_*.mpk`; eval-supported):

```sh
cargo run -p snlds-train --bin snlds-train -- \
  --data-dir ./out/ball --output-dir ./out/ball/flow-ckpt \
  --flow-snlds --encoder cnn --res 32 --epochs 10 --batch-size 8 \
  --npca-glow-coupling affine --w-msm 1.0 --w-npca 1.0

cargo run -p snlds-eval --bin snlds-eval -- \
  --data-dir ./out/ball --checkpoint ./out/ball/flow-ckpt/flow_checkpoint_0009.mpk \
  --output ./out/ball/flow_inferred.rrd --sequences 5
```

(Use `flow_checkpoint_{N-1:04}.mpk` for your chosen `--epochs N`.)

You can move most of these settings into a TOML file (`--config path.toml`) and override individual fields from the CLI; see [`crates/snlds-train/train.example.toml`](crates/snlds-train/train.example.toml).

---

## Workspace crates

The workspace is split by milestone. Library crates expose Rust APIs; the four crates with binaries (`snlds-data`, `snlds-viz`, `snlds-train`, `snlds-eval`) ship a CLI for end-to-end use.

Start from **[See results (end-to-end)](#see-results-end-to-end)** for runnable commands; the sections below document each crate in more detail.

### `snlds-core`

Burn primitives + HMM kernels used by the model crate. Library only.

```rust
use burn::backend::{Cpu, Autodiff};
use snlds_core::hmm; // forward / backward / posterior kernels

type B = Autodiff<Cpu<f32>>;
```

### `snlds-data` — synthetic generation + SafeTensors IO

Generates cosine / polynomial SDS-style sequences and writes `sequences.safetensors` + `metadata.json`. Latents and observations are **F32**; discrete states are **I32**.

CLI (`snlds-gen`):

```sh
cargo run -p snlds-data --bin snlds-gen -- \
  --seed 42 --dim-obs 2 --dim-latent 2 --num-states 3 \
  --seq-length 64 --num-samples 32 --data-type cosine \
  --out ./out/run1
# Bouncing-ball RGB: e.g. `--observation image --res 32` (see **See results** above).
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

### `snlds-model` — `VariationalSnlds` + encoders/decoders + Neural PCA

Library only. Provides `VariationalSnlds`, the MLP/CNN encoders, `SnldsConfig`, plus `npca` (`NeuralPca`, `NeuralPcaConfig`, `PatchMode`, `SigmaSchedule`) for Glow-followed Neural PCA experiments.

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

Optional TOML defaults (**CLI overrides** the file per field): **`--config train.toml`**. See [`crates/snlds-train/train.example.toml`](crates/snlds-train/train.example.toml).

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
  --data-dir ./out/run1 --checkpoint ./out/run1/ckpt/checkpoint_0019.mpk \
  --output ./out/run1/inferred.rrd --sequences 5
# add --spawn for the live viewer; checkpoint name must match `--epochs` (see [See results](#see-results-end-to-end))
```

Library API:

```rust
use snlds_eval::{run_eval, EvalConfig};

let cfg = EvalConfig {
    data_dir: "./out/run1".into(),
    checkpoint: "./out/run1/ckpt/checkpoint_0019.mpk".into(),
    output: "./out/run1/inferred.rrd".into(),
    spawn: false,
    sequences: 5,
    // optional per-field overrides for hidden_dim / temperature / obs_noise_var / beta
};
run_eval::<MyBackend>(&cfg, &device)?;
```

A typical workflow logs ground-truth (`snlds-viz`) and inferred (`snlds-eval`) into the same `.rrd` for side-by-side inspection.
