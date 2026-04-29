# M-Viz+ milestone — Rerun posteriors, reconstructions, training scalars

**PRD:** [§9 Milestones](PRD-burn-port.md#9-milestones) (row **M-Viz+**), [§5.4 Visualization (Rerun)](PRD-burn-port.md#54-visualization-rerun).

**Status:** merged — library logging APIs + smoke tests in **`snlds-viz`** (training CLI wiring optional; see **Updates**).

---

## Goal

Extend **[M-Viz](M-Viz.md)** Rerun logging so that **trained** (**[M3](M3.md)**) or **eval** runs can visualize **posterior discrete-state marginals** \(\gamma_{t,k}=p(s_t{=}k\mid\cdot)\), **reconstructed** observations \(\hat{x}_t\), and **training diagnostics** (ELBO, MSE / reconstruction, MSM term, learning rate, temperature) on shared **entity paths** and timeline conventions. Export **`.rrd`** for CI artifacts and sharing; optional **live** viewer (**`spawn`**) behind flags.

Depends on **M3** (or a minimal eval path that exposes \(\gamma\) and \(\hat{x}\)); ground-truth logging from **M-Viz** remains valid without a model.

---

## Entity schema (extends M-Viz)

Reuse **M-Viz** paths for \(z_t\), \(x_t\), \(s_t\) where ground truth exists. **Implemented** paths:

| Entity path (implementation) | Rerun type | Content |
|-------------------------------|------------|---------|
| `snlds/state/gamma_{k}` | `Scalars::single` per timestep | Posterior \(\gamma_{t,k}\) vs \(t\) (one entity per \(k\); timelines `sequence` + `time`) |
| `snlds/obs/x_hat` | `LineStrips2D` when **`obs_dim == 2`** | Reconstruction \(\hat{x}_t\) as a single path |
| `snlds/obs/x_hat_d{d}` | `Scalars::single` per timestep | Per-dimension series when **`obs_dim != 2`** |
| `snlds/train/elbo` | `Scalars::single` | ELBO (caller-defined sign; typically maximised ELBO) on timeline **`train_step`** |
| `snlds/train/mse` | `Scalars::single` | Scalar passed by caller (e.g. reconstruction MSE) |
| `snlds/train/temperature` | `Scalars::single` | Temperature |
| `snlds/markov/q_true` (nodes + edges) | `GraphNodes` + `GraphEdges` (Directed) | Ground-truth Markov chain — nodes `s0..s{K-1}` colored from the categorical state palette, edges `i → j` filtered by `\|q_{ij}\| ≥ TRANSITION_EDGE_EPSILON` |
| `snlds/markov/q_true/weights` | `Image` (RGB U8, viridis) | `[K, K]` heatmap of `q_true` for exact transition probabilities (graph view drops edge labels) |
| `snlds/markov/q_inferred` (+ `/weights`) | same as above | Same layout for the **softmax(`q_logits`)** of a trained model — logged by `snlds-eval` |
| `snlds/state/strip_true` | `Image` (RGB U8) | `STATE_STRIP_HEIGHT × T` colored band of true `s_t` (Figure-6-style segmentation), per `sequence` |
| `snlds/state/strip_inferred` | `Image` (RGB U8) | Same band for **argmax(γ_t)** of a trained model — logged by `snlds-eval` |
| `snlds/state/gamma` | `Image` (RGB U8, viridis) | `K × T` posterior heatmap (rows = states), per `sequence` |

Align with PRD [§5.4](PRD-burn-port.md#54-visualization-rerun): **paper-style** discrete-state plots (\(\gamma_{t,k}\) vs \(t\)); **overlay** with true \(s_t\) when **[M1](M1.md)** labels exist. **Stride** or **subsample** logging to control **log volume** ([PRD §6](PRD-burn-port.md#6-non-functional-requirements)).

---

## Completion checklist

- [x] **Library API** in **`snlds-viz`** — [`log_posteriors`](../crates/snlds-viz/src/log.rs), [`log_reconstructions`](../crates/snlds-viz/src/log.rs), [`log_train_scalars`](../crates/snlds-viz/src/log.rs); re-exported from **`snlds_viz`** root ( **`rerun`** pin unchanged from **M-Viz** / **§8.5** ).
- [x] **Entity paths** documented above and **stable** (append-only extension of **[M-Viz](M-Viz.md)** schema).
- [ ] **Training / eval binary** calls the new log functions (optional; **[M4](M4.md)** ships stdout metrics only — wire **`--viz`** or env in a follow-up).
- [ ] **Posterior** semantics: callers pass \(\gamma_{t,k}\) from model eval; dedicated **`predict_sequence`-style** path still optional if **`forward`** omits full posteriors (PRD §5.4).
- [ ] **Ground-truth overlay** automated test with **`states_*`** from **M1** (manual Rerun compare remains valid).
- [x] **Smoke tests** — no panic on tiny tensors (`crates/snlds-viz/tests/smoke.rs`).
- [ ] [docs/PRD-burn-port.md](PRD-burn-port.md) **§8.5** only if **`rerun`** pin changes (unchanged).

---

## Testing requirements

Inherit **M0** gates: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

| Area | Requirement |
|------|-------------|
| **Smoke** | Record **≥1** scalar series + **≥1** posterior or reconstruction channel without viewer errors |
| **Finite** | Logged tensor / scalar values **finite** for toy step |
| **Stride** | Document default logging stride; avoid fulltensor-per-step blowups |

**Out of scope:** **NeuralMSM**-specific viz unless folded from **[M5](M5.md)**; **image/CNN** modalities — **[M6](M6.md)**.

---

## Out of scope (milestone boundaries)

- **Full training product** defaults — **[M4](M4.md)**.
- **Ground-truth-only** viewer without model — **[M-Viz](M-Viz.md)**.
- **Core model** implementation — **[M3](M3.md)**.

---

## Downstream

**[M4](M4.md)** may still add **M-Viz+** behind **`--viz`** / **`RERUN`** on **`snlds-train`**. PRD [§10 Success criteria](PRD-burn-port.md#10-success-criteria) item **4** needs a small eval or scripted logging session that records \(\gamma_{t,k}\) and true \(s_t\) together — library support is in place; orchestration can follow.

---

## Updates

| Item | Description |
|------|-------------|
| 2026-04-29 | **Merged:** `snlds-viz` exposes `log_posteriors`, `log_reconstructions`, `log_train_scalars` + smoke tests. Entity paths use `gamma_{k}` (indexed), not a literal `gamma_k` suffix. |
| 2026-04-29 | **Markov + segmentation views (feat/m-viz-graphs):** new `snlds-viz` API `log_transition_matrix` (graph + sibling weight heatmap), `log_state_strip`, `log_gamma_heatmap`; centralised palettes/anchors in `crates/snlds-viz/src/colormap.rs`. `snlds-viz` binary surfaces `q_true` graph + `strip_true` per sequence. New **`snlds-eval`** crate consumes a checkpoint and logs `q_inferred`, `strip_inferred`, `snlds/state/gamma` heatmap, and reconstructions. |

---

## Document history

| Date       | Note |
|------------|------|
| 2026-04-29 | **Merged** — implementation checkpoint + residual follow-ups (training CLI `--viz`, overlay test). |
| 2026-04-29 | Initial tracker (aligned with M-Viz / M3). |
| 2026-04-29 | Added Markov-chain graph + Figure-6-style segmentation panels (`log_transition_matrix`, `log_state_strip`, `log_gamma_heatmap`) and `snlds-eval` binary for inferred-Q comparison. |
