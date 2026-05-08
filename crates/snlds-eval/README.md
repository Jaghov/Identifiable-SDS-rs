# `snlds-eval` (M-Viz+)

**Inference + Rerun logging** for a trained **`VariationalSnlds`** checkpoint: loads observations from M1 data dir, runs forward inference, logs **inferred** transition structure, **posterior** \(\gamma\) heatmaps, state strips, and reconstructions to Rerun.

Pairs with **`snlds-viz`** (ground truth): run both binaries (each writes its own `.rrd` by default) and **open the recordings together** in Rerun to compare inferred vs true paths (entity namespaces differ, e.g. `q_inferred` vs `q_true`).

## Contents

- **`run_eval`** — main library entry ([`src/lib.rs`](src/lib.rs)); takes [`EvalConfig`](src/lib.rs) and resolves hyperparameters from **`train_config.json`** next to the checkpoint unless overridden.
- Uses **`snlds-train`**’s `load_train_obs` + **`TrainSnapshot`** for consistent layout and snapshot schema.

## Dependencies

- **`snlds-data`**, **`snlds-model`**, **`snlds-train`**, **`snlds-viz`**
- **`burn`** (`ndarray` feature only — inference path)
- **`rerun`**, **`ndarray`**, **`anyhow`**, **`clap`**

## Binary

**`snlds-eval`**

```sh
cargo run -p snlds-eval --bin snlds-eval -- --help
```

Requires `--data-dir` (M1 outputs) and `--checkpoint` (from `snlds-train`).

## See also

- Ground-truth logging: [`../snlds-viz/README.md`](../snlds-viz/README.md)
- Training: [`../snlds-train/README.md`](../snlds-train/README.md)
- Repository overview: [`../../README.md`](../../README.md)
