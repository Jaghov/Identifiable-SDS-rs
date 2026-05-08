# `snlds-core`

Shared **Burn** primitives for the SNLDS stack: primarily **HMM-style forward / backward / posterior** kernels used by the variational model (`snlds-model`). Library-only crate (no binary).

## Contents

- **`hmm`** — local filtering / smoothing-style computations on Burn tensors (see [`src/hmm.rs`](src/hmm.rs)).

## Dependencies

- **`burn`** `0.20` with `cpu` and `autodiff` features.

Workspace crates: **none** (this is the lowest-level crate).

## Usage

Use as a dependency from `snlds-model` / higher crates. No CLI.

```toml
snlds-core = { path = "../snlds-core" }
```

## See also

- Repository overview: [`../../README.md`](../../README.md)
- Product requirements: [`../../docs/PRD-burn-port.md`](../../docs/PRD-burn-port.md)
