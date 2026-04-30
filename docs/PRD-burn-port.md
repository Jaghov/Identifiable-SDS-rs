# PRD: Port identifiable-SDS (SNLDS) to Burn

**Document version:** 1.18  
**Last updated:** 2026-04-29  
**Status:** Draft (living document)

This file is the **single source of truth** for the Burn port. **Update it when scope, dependencies, or milestones change** so everyone (including automated agents) stays aligned.

---

## How we collaborate on this PRD

- **Living document:** Requirements evolve; keep this file accurate. Prefer small, dated edits over stale sections.
- **Changelog:** Append a row to [§ Changelog](#changelog) for non-trivial changes (scope, milestones, pinned versions).
- **Conflict avoidance:** If two agents work in parallel, one should own “model/inference” vs “data/train” sections for a given session, or merge edits in one PR with an explicit note in the changelog.
- **Decisions:** Record resolved open questions in [§ Open questions](#12-open-questions) (mark **Resolved:** with date and outcome).
- **Burn / Rust / Rerun versions:** When pinned, document them in [§ Dependencies](#8-dependencies-rust-ecosystem) (including Rerun SDK vs viewer compatibility) and bump the document version + changelog.

---

## Changelog

| Date       | Version | Summary |
|------------|---------|---------|
| 2026-04-29 | 1.18    | **M1 schema v4 / simulator hparams:** surface `init_noise_std`, `init_mean_std`, `transition_step_var`, `emission_hidden_dim`, `initial_distribution` on `GenConfig`; persist the four scalars in `Manifest`; bump `MANIFEST_SCHEMA_VERSION` to **4** with serde defaults so v3 files still load. `EMISSION_HIDDEN_DIM` becomes a default, not a fixed constant. See [docs/M1.md](M1.md) and [docs/CLEANUP-hardcoded-values.md](CLEANUP-hardcoded-values.md) PR 3. |
| 2026-04-29 | 1.17    | **Train/eval config snapshot:** `snlds-train` adds `obs_noise_var` to `TrainConfig` + CLI and writes `<output_dir>/train_config.json` (`hidden_dim`, `beta`, `temperature`, `obs_noise_var`); `snlds-eval` reads it automatically with optional overrides. M1 doc block now lists v1 history. |
| 2026-04-29 | 1.16    | **M-Viz+ extension (feat/m-viz-graphs):** Markov-chain graph view + Figure-6-style segmentation panels in `snlds-viz` (`log_transition_matrix`, `log_state_strip`, `log_gamma_heatmap`, palettes in `colormap.rs`); **new `snlds-eval` crate/binary** consumes a checkpoint and logs `q_inferred` + posteriors; **M1 schema v3** persists `q_true`/`pi_true`. Trackers: [M1.md](M1.md), [M-Viz+.md](M-Viz+.md). |
| 2026-04-29 | 1.15    | **M5** merged + **M6** explicitly deferred: tracker [M5.md](M5.md); `snlds-msm` crate (linfa-reduction PCA + simplified NeuralMSM) + `snlds-train --msm-init`; §8.2/§8.5 updated; §9 status. |
| 2026-04-29 | 1.14    | **M-Viz+** + **M4** merged: trackers [M-Viz+.md](M-Viz+.md), [M4.md](M4.md) updated; §9 status; **`snlds-train`** / **`CompactRecorder`** checkpoint note in §12. |
| 2026-04-29 | 1.13    | Milestone trackers [M-Viz](M-Viz.md), [M3](M3.md), [M-Viz+](M-Viz+.md), [M4](M4.md), [M5](M5.md), [M6](M6.md); §9 links + table footnotes. |
| 2026-04-29 | 1.12    | [docs/M2.md](M2.md) M2 milestone tracker (HMM kernels / local evidence); §9 link + table footnote. |
| 2026-04-29 | 1.11    | **`snlds-data`**: **`rand` 0.9** / **`rand_chacha` 0.9** / **`rand_distr` 0.5** (drop unused **`thiserror`**); §8.5 RNG pins. |
| 2026-04-29 | 1.10    | M1 poly integration shape/finite/range test; observation emission **`func_leaky_relu_batch`** only; **`sample_adj_mat`** cleanup; [docs/M1.md](M1.md). |
| 2026-04-29 | 1.9     | M1 **`Manifest` `Deserialize`** + **`load_manifest`**; polynomial scalar **`poly_mean_for_state`**; **`EMISSION_HIDDEN_DIM`**; [docs/M1.md](M1.md) § Updates. |
| 2026-04-29 | 1.8     | M1 **`encode_safetensors`**: scoped staging buffers (**no `Box::leak`**); [docs/M1.md](M1.md) § Implementation / SafeTensors IO. |
| 2026-04-29 | 1.7     | M1 SafeTensors schema **v2**: **`states_*`** persisted as **`I32`** ([docs/M1.md](M1.md)); manifest **`schema_version`** 2. |
| 2026-04-29 | 1.6     | M1 codebase: **`snlds-data`** + **`snlds-gen`**; PRD §8.5 extra dependency pins. |
| 2026-04-29 | 1.5     | [docs/M1.md](M1.md) synthetic data milestone tracker + SafeTensors key schema reference. |
| 2026-04-29 | 1.4     | [docs/M0.md](M0.md) milestone tracker for implementation checklist + testing gates. |
| 2026-04-28 | 1.3     | SafeTensors-first persistence (§8.1, §4.1, §5.1, M1); ndarray-npy/NPZ optional; §12 resolved item. |
| 2026-04-28 | 1.2     | Milestone order: M-Viz (ground-truth only) immediately after M1; M-Viz+ after M3 for posteriors / training scalars. |
| 2026-04-28 | 1.1     | Rerun visualization: generated sequences, posterior \(\gamma_{t,k}\), paper-style figures; new milestone M-Viz; deps and functional requirements. |
| 2026-04-28 | 1.0     | Initial PRD: core SNLDS, data gen, training, optional NeuralMSM; deps and milestones. |

---

## 1. Summary

Port the **Switching Nonlinear Dynamical System (SNLDS)** training stack from the Python project under `identifiable-SDS/`, including **synthetic data generation**, **training**, **optional NeuralMSM warm-start**, and **Rerun-based visualization** (sequences, posteriors, training curves), so experiments can run natively in Rust with GPU/CPU backends supported by Burn.

**Source of truth:** ICML 2024 code paths documented in `identifiable-SDS/README.md`, especially `VariationalSNLDS`, `generate_data_and_train_snlds.py`, and (optionally) `NeuralMSM` + `train_snlds.py` initialization behavior.

---

## 2. Goals

1. **Functional parity (core):** Reproduce the main SDS pipeline: generate synthetic sequences, train `VariationalSNLDS` with `inference='alpha'`, `annealing=False`, recurrent (or staged) encoder, ELBO with switching term, temperature schedule as in Appendix F.3–style hyperparameters in code.
2. **Optional parity:** Support **NeuralMSM pre-training** and parameter transfer into SNLDS, gated by configuration (not required for default training).
3. **Operability:** CLI or binary entrypoints for `generate-data`, `train`, and optional `msm-warmstart`; configurable seeds, dimensions, and device/backend.
4. **Maintainability:** Clear module boundaries (model, inference kernels, data gen, training loop, IO).
5. **Visualization ([Rerun](https://rerun.io)):** Log **generated and reconstructed sequences**, **posterior discrete-state marginals** \(\gamma_{t}(k)\) (paper-style, e.g. [arXiv:2305.15925](https://arxiv.org/abs/2305.15925)), ground-truth vs inferred trajectories when labels exist, and training diagnostics (ELBO, MSE, temperature). Prefer `.rrd` export for sharing; optional live streaming during training.

---

## 3. Non-goals (initial release)

- Full reproduction of **every** Python experiment (e.g. all figure sweeps) unless explicitly scheduled.
- **PolyMSM** and classical **MSM** baselines beyond **NeuralMSM** optional path.
- **Jupyter / Matplotlib** as the **primary** exploratory stack — not required; **Rerun** is the default visualization path (see §5.4).
- Bit-exact RNG parity with NumPy (see success criteria).

---

## 4. Scope

### 4.1 In scope

| Area | Description |
|------|-------------|
| **VariationalSNLDS** | Transition MLPs per discrete state; decoder MLP; \(\pi\), \(Q\), init means/covs, emission covs; fixed observation noise var; beta and temperature handling. |
| **Encoder** | **Minimum:** `recurrent` (bi-LSTM → causal LSTM → Gaussian params) matching Python. **Fallback milestone:** `factored` MLP encoder for earlier integration tests. |
| **Inference** | `_compute_local_evidence`, `_alpha`, `_beta`; `inference='alpha'` ELBO path using `log_Z`; optional non-alpha path deferred unless needed. |
| **Loss / forward** | Reconstruction (factorized Normal log-prob), entropy / KL terms for \(q(z)\), MSM term aggregation consistent with Python. |
| **Data generation** | Markov \(\pi\) and \(Q\) from `get_trans_mat`; latent rollouts with cosine / polynomial / softplus-style dynamics; batched leaky-ReLU **`func_leaky_relu_batch`** for observations; train/test splits; **SafeTensors on disk** by default (optional `.npy`/NPZ for parity scripts). |
| **Training** | Minibatches, Adam, LR schedule (StepLR analogue), gradient clipping, early-stop hooks matching key thresholds where documented, multi-restart loops, checkpoints. |
| **Optional NeuralMSM** | Fit on PCA-reduced observations (latent dimension); copy into SNLDS: transitions, \(\log Q\), \(\log \pi\), `init_mean`, `init_cov`; then run main training. |
| **Rerun visualization** | Log per-timestep scalars (e.g. \(\gamma_{t,k}\) vs \(t\) for each discrete state \(k\)), latent/observation trajectories (2D as paths or higher-D as components), decoded rollouts, and optional heatmaps. Align naming with a stable **log schema** (e.g. `snlds/state_posterior/k`, `snlds/latent/true`). |

### 4.2 Out of scope (v1)

- `train_neuroscience.py` workflows.
- Official Python binding or automatic conversion of PyTorch checkpoints (nice-to-have later).

---

## 5. Functional requirements

### 5.1 Data generation

- **Inputs:** Seeds, `num_states`, `dim_obs`, `dim_latent`, `T`, `num_samples`, `sparsity_prob`, `data_type` (`cosine` \| `poly` \| softplus analogue), polynomial `degree` when applicable.
- **Outputs:** Observation arrays `[N, T, D]` (and latent/state ground truth for evaluation). Persist as **`.safetensors`** with named tensors unless a Python parity path requires `.npy`/NPZ.
- **Determinism:** Document RNG strategy (Rust `rand` / Burn); optional seed parameter for repeatable runs.

### 5.2 Model / training

- Load generated (or compatible) tensors; batch shuffle.
- Instantiate SNLDS; optional MSM init branch.
- Training loop with logged metrics: ELBO, MSE (reconstruction), MSM term, temperature when annealed.
- Save/load model state (checkpoint format — see §7).

### 5.3 Optional NeuralMSM

- PCA or equivalent dimensionality reduction to `dim_latent` before MSM fit when mirroring `train_snlds.py`.
- Hyperparameters surfaced (epochs, lr, hidden dim, cosine activation, restart count) with sensible defaults aligned to Python scripts.
- After warm-start: same unlock schedule for \(\beta\), \(Q/\pi\) gradients as in Python (document any simplification).

### 5.4 Visualization (Rerun)

- **Sequences:** Log generated latents \(z_t\), observations \(x_t\), and model reconstructions \(\hat{x}_t\) (and \(\hat{z}_t\) when available) for selected batch indices and sequence length \(T\).
- **Posterior / paper-style figures:** When \(\gamma_{t,k} = p(s_t=k \mid \cdot)\) is computed (e.g. from `_compute_posteriors` or an eval-only forward), log **overlaid or stacked** series vs \(t\) for each \(k\), comparable to discrete-state inference plots in the paper.
- **Training:** Scalar timelines for ELBO, reconstruction term, MSM term, MSE, learning rate, and temperature annealing.
- **Ground truth:** For simulator-generated data, log true discrete states \(s_t\) alongside \(\gamma_t\) for qualitative comparison.
- **Implementation notes:** Under `inference='alpha'` training, full posteriors may require a dedicated **eval pass** with posteriors enabled (mirroring Python’s `forward` vs `predict_sequence`). For **2D latents**, use path/line entities; for **1D** observation dimensions, use time-series scalars. Export **`.rrd`** for CI artifacts or reviewer sharing as needed.

---

## 6. Non-functional requirements

| Concern | Target |
|---------|--------|
| **Numerical stability** | Cholesky (or stable) Gaussian log-density for full covariances; `logsumexp` for forward–backward; match Python eps regularization patterns (`1e-6`, etc.). |
| **Performance** | GPU training path via Burn backend (e.g. CUDA/WGPU/Candle — project-chosen); batching preserves throughput expectations for synthetic dataset sizes (~5k × 200 steps). |
| **Testing** | Unit tests on HMM kernels vs exported PyTorch tensors; smoke tests end-to-end on CPU with tiny \(N,T\). |
| **Docs** | Build/run instructions, feature flags, backend selection, parity limitations (RNG, optional image path). |
| **Visualization** | Rerun sessions remain **reproducible** (entity paths documented); avoid logging full training tensors every step at high frequency by default (sampling / stride). |

---

## 7. Technical design (high level)

- **crate layout:** `snlds-core` (HMM kernels), `snlds-model` (Burn `Module`s), `snlds-data` (pure Rust simulation + IO), `snlds-train` (CLI + loops), `snlds-msm` (NeuralMSM warm-start), `snlds-viz` (ground-truth Rerun logging + binary), `snlds-eval` (checkpoint → inferred Rerun logging binary).
- **Burn:** Core tensor ops, autodiff, optimizers (`burn::optim`), module derive macros; backend trait `Backend` parameterized throughout.
- **Rerun:** Use the official Rust SDK crate (**`rerun`**, `features` as needed) to record streams; view with the **Rerun viewer** (desktop or browser). Optional: `spawn` viewer from training binary behind `--viz` / `RERUN` env.
- **Checkpointing:** Burn `Recorder` / built-in persistence, or **`safetensors`** + manifest if interoperability matters.

---

## 8. Dependencies (Rust ecosystem)

### 8.1 Required (minimum viable)

| Crate / area | Role |
|--------------|------|
| **`burn`** (with features for chosen backends: `wgpu`, `cuda`, `candle`, `ndarray`) | Tensors, autodiff, `Module`, optimizers, training primitives. |
| **`rand`** + **`rand_distr`** | Synthetic data RNG, Gaussian samples. |
| **`ndarray`** | In-memory indexing and arrays during synthetic data simulation; optional preprocessing before tensors enter Burn. |
| **`safetensors`** (+ **`burn-store`** where checkpoints need Burnpack / SafeTensors loading) | **Preferred on-disk format** for generated datasets — multiple named tensors per `.safetensors` (`obs`, `latents`, `states`, …), aligned with **`burn-store`** / Hugging Face tooling. **`ndarray-npy`** / **NPZ** optional **only** for NumPy/Python interoperability or parity tests against existing `.npy` scripts — not the default artifact format. |
| **`clap`** | CLI parsing (parity with argparse scripts). |
| **`thiserror`** / **`anyhow`** | Error handling in binaries. |
| **`serde`** + **`serde_json`** | Config files and experiment metadata. |
| **`rerun`** | Time-series and tensor logging; paper-style posterior and trajectory visualization; `.rrd` export ([docs](https://www.rerun.io/docs/reference/types)). |

### 8.2 Likely needed for parity

| Crate / area | Role |
|--------------|------|
| **`linfa`** + **`linfa-reduction`** | PCA for MSM warm-start (M5 — `Pca::params(n).fit(...).transform(...)`). |
| **Cholesky / LAPACK** via **`linfa`** transitive (`ndarray-linalg`) or tensor-side only | Full-cov Gaussian `log_prob` if not delegated to Burn helpers (M5 currently uses **diagonal** factors only — see [docs/M5.md](M5.md)). |

### 8.3 Optional / milestone-specific

| Crate / area | Role |
|--------------|------|
| **`statrs`** or combinatorics helpers | Binomial coefficients / polynomial term counts (`comb`) for polynomial data type. |
| **`image`** / **`png`** | Deferred image rendering for bouncing-ball sequences. |

### 8.4 Not ported as libraries

- **OpenCV**, **scikit-learn**, **scipy** — functionality reimplemented in Rust (PCA, polynomial features, special functions).
- **tqdm** — `indicatif` if progress bars desired.

### 8.5 Pinned versions

| Crate | Version | Notes |
|-------|---------|-------|
| **`burn`** | `0.20.1` | features: `cpu`, `autodiff` (snlds-core); `snlds-data` deps tracked with M1 |
| **`burn-cpu`** | `0.20.1` | CubeCL CPU runtime backend |
| **`cubecl`** | `0.9.0` | transitive; JIT needs `RUST_MIN_STACK=33554432` in debug builds (`.cargo/config.toml`) |
| **`rerun`** | TBD | pin when wired in M-Viz |
| **`safetensors`** | `0.4.5` | `snlds-data` dataset export (`sequences.safetensors`) |
| **`ndarray`** | `0.16.1` | in-memory rollout arrays |
| **`rand_chacha` / `rand` / `rand_distr`** | `0.9.0 / 0.9.4 / 0.5.1` | **`snlds-data`** seeded `ChaCha8`, Gaussians (**no longer pins `rand` 0.8** alongside other workspace crates) |
| **`serde` / `serde_json`** | `1.x` | `metadata.json` |
| **`itertools`** | `0.13` | polynomial exponent order (`combinations_with_replacement`) |
| **`linfa`** / **`linfa-reduction`** | `0.8` | PCA for the M5 MSM warm-start (`snlds-msm`) |
| **`tempfile`** | `3` (`dev`) | integration tests |
| Rust toolchain | stable 1.95 | Ubuntu stable for CI |

---

## 9. Milestones

**Recommended implementation order:** M0 → **M1** → **M-Viz (GT)** → parallel **M2** kernels → **M3** → **M-Viz+** → **M4** train → **M5** optional MSM.

Detailed **M0** checklist + testing gates: **[docs/M0.md](M0.md)**.

Detailed **M1** checklist, schema, testing gates: **[docs/M1.md](M1.md)**.

Detailed **M2** checklist + testing gates: **[docs/M2.md](M2.md)**.

Detailed **M-Viz** checklist + testing gates: **[docs/M-Viz.md](M-Viz.md)**.

Detailed **M3** checklist + testing gates: **[docs/M3.md](M3.md)**.

Detailed **M-Viz+** checklist + testing gates: **[docs/M-Viz+.md](M-Viz+.md)**.

Detailed **M4** checklist + testing gates: **[docs/M4.md](M4.md)**.

Detailed **M5** checklist + testing gates: **[docs/M5.md](M5.md)**.

Detailed **M6** checklist + testing gates: **[docs/M6.md](M6.md)**.

| Phase | Deliverable |
|-------|-------------|
| **M0** | Repo skeleton, Burn backend smoke test, CI (fmt, clippy, test). *[Tracker: [M0.md](M0.md)]* |
| **M1** | Data gen (vector obs, cosine + **poly**), **SafeTensors** export by default, deterministic seeds. *[Tracker: [M1.md](M1.md)]* |
| **M-Viz** | **After M1:** Rerun **ground-truth only** — log synthetic \(z_t\), \(x_t\), true \(s_t\); entity schema + `.rrd`; no model required. *[Tracker: [M-Viz.md](M-Viz.md)]* |
| **M2** | Tensor tests: HMM forward–backward + local evidence vs reference tensors (exported or hand-checked). *(Can overlap M-Viz.)* *[Tracker: [M2.md](M2.md)]* |
| **M3** | `MLP`, kernel ops, full `VariationalSNLDS` with `factored` encoder first. *[Tracker: [M3.md](M3.md)]* |
| **M-Viz+** | **After M3:** extend Rerun with \(\gamma_{t,k}\), \(\hat{x}_t\), training scalars (ELBO, MSE, temperature). **✓ Merged** (library APIs in `snlds-viz`; training `--viz` optional). *[Tracker: [M-Viz+.md](M-Viz+.md)]* |
| **M4** | Factored training CLI (**`snlds-train`**), Adam, gradient clipping, **`CompactRecorder`** checkpoints. **✓ Merged** (StepLR / `--viz` deferred — see tracker). *[Tracker: [M4.md](M4.md)]* |
| **M5** | Optional NeuralMSM + warm-start (`snlds-msm` + `snlds-train --msm-init`). **✓ Merged** (simplifications documented). *[Tracker: [M5.md](M5.md)]* |
| **M6** | CNN encoder/decoder path (`EncoderKind { Mlp, Cnn { res } }` on `SnldsConfig`; `ObservationKind::Image { res }` on `GenConfig`). **✓ Merged** — promoted from stretch as the data-flow template for the planned `FlowSNLDS` encoder. *[Tracker: [M6.md](M6.md)]* |

---

## 10. Success criteria

1. **Core:** On synthetic data generated by either Python or Rust, trained SNLDS reaches qualitatively similar behavior (e.g. MSE, ELBO trajectory, recovered \(Q\) structure) under same hyperparameters — **statistical** match, not bitwise float equality.
2. **Kernels:** HMM \(\alpha/\beta\) and `log_Z` outputs match reference tensors within tight tolerance when inputs are identical.
3. **Optional:** With `--msm-init` (or equivalent), training runs without error and checkpoints document transferred parameters.
4. **Visualization:** A recorded Rerun session can replay **at least one** sequence with **\(\gamma_{t,k}\)** vs \(t\) and overlaid **true** \(s_t\) when ground truth exists, without viewer errors.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Bi-LSTM + stack differences vs PyTorch | Implement `factored` first; add integration test; consider GRU if LSTM parity is costly. |
| Full-cov Gaussian numerics | Unit tests vs PyTorch; Cholesky + symmetrize cov. |
| Long compile times / Burn API churn | Pin Burn version in §8.5 and changelog; periodic upgrade task. |
| Rerun log volume / perf | Log every-`k` scalars on a stride; optional `--viz-epoch` only. |

---

## 12. Open questions

- [x] Target **Burn version** and **default backend** — **Resolved (M0):** `burn 0.20.1`, CubeCL CPU (`burn-cpu`) for CI; WGPU/CUDA for GPU milestones.
- [x] **`npy` vs SafeTensors for datasets** — **Resolved (1.3):** SafeTensors-first; optional `ndarray-npy`/NPZ for NumPy parity only; not a gate for M3.
- **Checkpoint format (partial)** — **M4:** training checkpoints use Burn **`CompactRecorder`** **`.mpk`** (MessagePack record). Cross-tool **safetensors** export for collaborators remains optional / future work.
- [ ] Rerun: **offline-only** (save `.rrd` post-run) vs **streaming** during training by default.
- [x] **Persistent tensor naming for generated data** — **Resolved (1.5):** [docs/M1.md](M1.md) § SafeTensors key schema v1 (`latents_*`, `obs_*`, `states_*` + sidecar metadata). *(Rerun **entity** paths remain an M-Viz question.)*

_When resolved, move outcomes here or to §8.5 and note in changelog._

---

## 13. Appendix: Reference files (Python)

- `identifiable-SDS/models/VariationalSNLDS.py`
- `identifiable-SDS/models/modules.py` (MLP, `CNNFastEncoder`, `CNNFastDecoder`)
- `identifiable-SDS/generate_data_and_train_snlds.py`
- `identifiable-SDS/utils/transitions.py`
- `identifiable-SDS/models/NeuralMSM.py` (optional)
- `identifiable-SDS/train_snlds.py` (optional init pattern)

---

## Document history

| Version | Date       | Notes |
|---------|------------|-------|
| 1.19    | 2026-04-30 | **M6** merged: CNN encoder/decoder via `EncoderKind { Mlp, Cnn { res } }` + `ObservationKind::Image { res }`; §3 / §4.2 image carve-out lifted; §9 row flipped to merged; §13 reference list updated. |
| 1.18    | 2026-04-29 | **M1 schema v4**: simulator hparams on `GenConfig` + `Manifest`; configurable `initial_distribution`; `EMISSION_HIDDEN_DIM` demoted to default; v3 manifests still load. |
| 1.15    | 2026-04-29 | **M5** merged (`snlds-msm`, `--msm-init`); **M6** explicitly deferred; §8.2 / §8.5 deps add **`linfa`** / **`linfa-reduction`**. |
| 1.14    | 2026-04-29 | **M-Viz+** + **M4** merged; §9 table + §12 checkpoint partial resolve; changelog 1.14. |
| 1.13    | 2026-04-29 | Milestone trackers M-Viz, M3, M-Viz+, M4, M5, M6; §9 links + table footnotes. |
| 1.12    | 2026-04-29 | [docs/M2.md](M2.md) M2 milestone tracker (HMM kernels); §9 M2 tracker link. |
| 1.11    | 2026-04-29 | **`snlds-data`** RNG stack 0.9; remove **`thiserror`**; §8.5 pins. |
| 1.10    | 2026-04-29 | M1 poly integration test; **`func_leaky_relu_batch`** naming in §4.1; [docs/M1.md](M1.md). |
| 1.9     | 2026-04-29 | M1 **`load_manifest`**, **`poly_mean_for_state`**, **`EMISSION_HIDDEN_DIM`**; [docs/M1.md](M1.md) § Updates. |
| 1.8     | 2026-04-29 | M1 **`encode_safetensors`**: scoped **`Vec<u8>`** (**no leak**); [docs/M1.md](M1.md) Implementation / IO. |
| 1.7     | 2026-04-29 | M1 schema v2: `states_*` **I32** in SafeTensors; [docs/M1.md](M1.md). |
| 1.6     | 2026-04-29 | `snlds-data`/`snlds-gen` landed; §8.5 pins table extended. |
| 1.5     | 2026-04-29 | docs/M1.md M1 milestone + §8.5 note; §12 partial resolve on-disk tensor naming. |
| 1.4     | 2026-04-29 | docs/M0.md milestone tracker. |
| 1.3     | 2026-04-28 | SafeTensors-first datasets + aligned scope/M1/open question. |
| 1.2     | 2026-04-28 | Milestone ordering: M-Viz (GT) after M1; M-Viz+ after M3. |
| 1.1     | 2026-04-28 | Rerun visualization scope, §5.4, **`rerun`** dependency, milestone M-Viz, success criterion + open questions. |
| 1.0     | 2026-04-28 | Initial authoring. |

When bumping **Document version** at the top of this file, add a changelog row (§top) and a row above.
