# `snlds-train` (M4)

Training harness for **`VariationalSnlds`**: loads M1 **SafeTensors** splits from disk, runs minibatch Adam on Burn’s **`ndarray`** autodiff backend, writes **`CompactRecorder`** checkpoints and a **`train_config.json`** snapshot next to the checkpoint.

Optional **NeuralMSM warm-start** (`--msm-init`) delegates to `snlds-msm` before main ELBO training.

## Contents

- **`train`** — [`TrainConfig`](src/train.rs), [`train`](src/train.rs) / [`train_with_model`](src/train.rs), [`build_model_config`](src/train.rs) (wires manifest → [`SnldsConfig`](../snlds-model/src/model.rs)).
- **`data`** — [`load_train_obs`](src/data.rs) / `ObsTensor` for batched observations.
- **`snapshot`** — [`TrainSnapshot`](src/snapshot.rs) for serde’d hyperparameters + encoder kind (paired with eval).
- **`warm_start`** — [`run_warm_start`](src/warm_start.rs), [`MsmWarmStartConfig`](src/warm_start.rs).
- **`config_file`** — optional TOML defaults ([`train.example.toml`](train.example.toml)); [`TrainCli`](src/config_file.rs) merges file + CLI ([`resolve_train`](src/config_file.rs)).

## Dependencies

- **`snlds-model`**, **`snlds-data`**, **`snlds-msm`**
- **`burn`** with `ndarray` + `autodiff`
- **`ndarray`**, **`anyhow`**, **`clap`**, **`serde`**, **`toml`**

## Binary

**`snlds-train`**

```sh
cargo run -p snlds-train --bin snlds-train -- --help
```

### TOML defaults (`--config`)

Pass **`--config path.toml`** for a flat TOML table of defaults (see [`train.example.toml`](train.example.toml)). **`data_dir`** and **`output_dir`** must appear in the file **or** on the command line. Any flag you pass explicitly overrides the file for that field. For flow / Neural PCA runs, set **`mode = "flow_snlds"`** or **`"neural_pca"`** in the file, or use **`--flow-snlds` / `--neural-pca`** on the CLI to force the mode regardless of the file.

```sh
cargo run -p snlds-train --bin snlds-train -- --config train.toml --epochs 5
```

### Neural PCA

Trains [`NeuralPca`](../snlds-model/src/npca/neural_pca.rs) on **image** M1 data (`dim_obs == 3 * res * res`). Writes `npca_train_config.json` and `npca_checkpoint_*.mpk` (not SNLDS checkpoints).

```sh
cargo run -p snlds-train --bin snlds-train -- \
  --neural-pca --res 16 --data-dir ./out/img --output-dir ./out/img_npca --epochs 10
```

## See also

- MSM warm-start implementation: [`../snlds-msm/README.md`](../snlds-msm/README.md)
- Inference / Rerun: [`../snlds-eval/README.md`](../snlds-eval/README.md)
- Repository overview: [`../../README.md`](../../README.md)
