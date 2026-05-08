# `snlds-viz` (M-Viz)

**Rerun** logging for **ground-truth** SNLDS sequences: transition matrix, discrete-state strips, posterior-style panels, reconstructions, and training scalars. Includes small **colormap** / **render** helpers for consistent visuals.

Use this after M1 generation to inspect `q_true`, latents, and observations before or during training.

## Contents

- **`log`** — [`log_transition_matrix`](src/log.rs), [`log_state_strip`](src/log.rs), [`log_gamma_heatmap`](src/log.rs), [`log_posteriors`](src/log.rs), [`log_reconstructions`](src/log.rs), [`log_train_scalars`](src/log.rs).
- **`colormap`**, **`render`** — palettes and drawing helpers.

## Dependencies

- **`snlds-data`** (manifest + tensor layout awareness)
- **`rerun`** SDK
- **`ndarray`**, **`anyhow`**, **`clap`**

## Binary

**`snlds-viz`** — reads a directory with `sequences.safetensors` + `metadata.json`, writes a `.rrd` (or spawns the viewer).

```sh
cargo run -p snlds-viz --bin snlds-viz -- --help
```

Use **`--render`** when visualising image observations (`dim_latent == 2`).

## See also

- Inferred posteriors / Markov logging (checkpoint side): [`../snlds-eval/README.md`](../snlds-eval/README.md)
- Repository overview: [`../../README.md`](../../README.md)
