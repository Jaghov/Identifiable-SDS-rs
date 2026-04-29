# M-Viz+ milestone — Rerun posteriors, reconstructions, training scalars

**PRD:** [§9 Milestones](PRD-burn-port.md#9-milestones) (row **M-Viz+**), [§5.4 Visualization (Rerun)](PRD-burn-port.md#54-visualization-rerun).

**Status:** planned — checkboxes open until implementation merges.

---

## Goal

Extend **[M-Viz](M-Viz.md)** Rerun logging so that **trained** (**[M3](M3.md)**) or **eval** runs can visualize **posterior discrete-state marginals** \(\gamma_{t,k}=p(s_t{=}k\mid\cdot)\), **reconstructed** observations \(\hat{x}_t\), and **training diagnostics** (ELBO, MSE / reconstruction, MSM term, learning rate, temperature) on shared **entity paths** and timeline conventions. Export **`.rrd`** for CI artifacts and sharing; optional **live** viewer (**`spawn`**) behind flags.

Depends on **M3** (or a minimal eval path that exposes \(\gamma\) and \(\hat{x}\)); ground-truth logging from **M-Viz** remains valid without a model.

---

## Entity schema (extends M-Viz)

Reuse **M-Viz** paths for \(z_t\), \(x_t\), \(s_t\) where ground truth exists. Add (names illustrative — finalize in implementation):

| Entity path (example) | Rerun type | Content |
|----------------------|------------|---------|
| `snlds/state/gamma_k` | `Scalar` (per \(k\)) or stacked series | Posterior \(\gamma_{t,k}\) vs \(t\) |
| `snlds/obs/x_hat` | `Points2D` / `LineStrips2D` or tensor | Reconstruction \(\hat{x}_t\) |
| `snlds/train/elbo` | `Scalar` | ELBO (or minimized loss) timeline |
| `snlds/train/mse` | `Scalar` | Reconstruction MSE when logged |
| `snlds/train/temperature` | `Scalar` | Temperature schedule |

Align with PRD [§5.4](PRD-burn-port.md#54-visualization-rerun): **paper-style** discrete-state plots (\(\gamma_{t,k}\) vs \(t\)); **overlay** with true \(s_t\) when **[M1](M1.md)** labels exist. **Stride** or **subsample** logging to control **log volume** ([PRD §6](PRD-burn-port.md#6-non-functional-requirements)).

---

## Completion checklist

- [ ] **API / hooks** in training or eval binary to emit \(\gamma_{t,k}\), \(\hat{x}_t\), and chosen scalars to Rerun ( **`rerun`** pin per **M-Viz** / **§8.5** ).
- [ ] **Entity paths** documented and **stable** (append-only extension of **[M-Viz](M-Viz.md)** schema).
- [ ] **Posterior** logging matches eval semantics (dedicated **eval forward** if training `forward` omits full posteriors, mirroring Python `forward` vs `predict_sequence` notes in PRD §5.4).
- [ ] **Ground-truth overlay** path tested when `states_*` / sequences available from **M1** data.
- [ ] **`.rrd` export** integration smoke: non-empty recording after short train or eval on tiny config.
- [ ] [docs/PRD-burn-port.md](PRD-burn-port.md) **§8.5** / **§5.4** cross-links updated if schema or pins change.

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

**[M4](M4.md)** may bundle **M-Viz+** behind **`--viz`** / **`RERUN`** flags on the training CLI. PRD [§10 Success criteria](PRD-burn-port.md#10-success-criteria) item **4** (replay with \(\gamma_{t,k}\) vs \(t\) and true \(s_t\)) is satisfied once this milestone ships with **[M3](M3.md)** eval.

---

## Updates

| Item | Description |
|------|-------------|
| | |

---

## Document history

| Date       | Note |
|------------|------|
| 2026-04-29 | Initial tracker (aligned with M-Viz / M3). |
