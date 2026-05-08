# `snlds-msm` (M5)

Optional **NeuralMSM** warm-start for **`VariationalSnlds`**: PCA-style observation reduction (**linfa-reduction**), train a small **NeuralMSM** model, then **`transfer_into_snlds`** to copy compatible parameters into SNLDS before full variational training.

Consumed by **`snlds-train`** via `--msm-init`; no standalone CLI.

## Contents

- **`msm`** — [`NeuralMsm`](src/msm.rs), [`NeuralMsmConfig`](src/msm.rs).
- **`pca`** — [`pca_fit_transform`](src/pca.rs) for reducing observation dimension before MSM fit.
- **`transfer`** — [`transfer_into_snlds`](src/transfer.rs).

## Dependencies

- **`snlds-core`**, **`snlds-model`**
- **`burn`** with `ndarray` + `autodiff`
- **`ndarray`**, **`linfa`**, **`linfa-reduction`**, **`anyhow`**, **`rand`**

Workspace crates: **`snlds-data`** is dev-only (tests).

## Usage

Library-only:

```toml
snlds-msm = { path = "../snlds-msm" }
```

Typical entrypoint is `snlds_train::run_warm_start` with [`MsmWarmStartConfig`](../snlds-train/src/warm_start.rs).

## See also

- Training CLI: [`../snlds-train/README.md`](../snlds-train/README.md)
- Repository overview: [`../../README.md`](../../README.md)
