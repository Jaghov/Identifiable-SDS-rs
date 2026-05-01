# Cleanup plan — hardcoded values across producer/consumer boundaries

**Status:** active
**Triggered by:** post-merge audit of `feat/m-viz-graphs` + `fix/eval-config-snapshot`, where a `5e-4` (train) vs `0.1` (eval) `obs_noise_var` mismatch slipped through review as a "non-blocking nit". Followup audit (this doc) found additional similar smells across the workspace.

---

## Goal

Eliminate "same number, two places" patterns where a numeric constant lives on both the producer side (data generator, training loop, model layout) and the consumer side (eval, downstream visualisation), with no compile-time link guaranteeing they match. Route every such constant through a single source of truth — either a config struct, a manifest field, or a model parameter.

---

## Findings (severity-ranked)

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | **High** | `crates/snlds-model/src/model.rs:29-31` vs `crates/snlds-train/src/train.rs:148` | `SnldsConfig::obs_noise_var` is a dead field — `forward()` takes the value as a runtime parameter and `snlds-train` reads `TrainConfig::obs_noise_var`, never the model field. Two sources of truth. |
| 2 | Medium | `crates/snlds-data/src/generate.rs:184` | Initial state sampled `Uniform(0..K)` while `pi_true` is computed alongside; no link enforces correspondence. Today both are uniform by construction; future divergence is silent. |
| 3 | Medium | `crates/snlds-data/src/transitions.rs:9` | `EMISSION_HIDDEN_DIM = 8` not in manifest. Changing the simulator's emission MLP width breaks no consumer visibly. |
| 4 | Medium | `crates/snlds-data/src/generate.rs:170,172,195` | Simulator noise/spread constants (`init_noise_std=0.1`, `init_mean_std=0.7`, `transition_step_var=0.05`) hardcoded, not on `GenConfig`, not in manifest. |
| 5 | Low | `crates/snlds-data/src/transitions.rs:15-20` | `0.9 / 0.1` cyclic Q hardcoded; faithfully persisted via `q_true` so consumer-side is fine. Smell is simulator-side rigidity only. |
| 6 | Low | several non-test files (see PR 2 below) | `.unwrap()` on infallible-by-construction calls — should be `.expect("…")` for postmortem clarity, or `?` / `anyhow::ensure!` when it represents a contract. |

---

## Execution

Three sequential PRs (PR 1–3), each landed via the standard worktree → bad-mood reviewer → commit + merge pipeline. Reviewer must approve **with all concerns (including nitpicks) addressed** before merge. **PR 4** (Markov topology pluggability) landed separately — see § PR 4 below.

### PR 1 — `fix/snlds-config-obs-noise-dedupe` (issue #1)

**Approach:** delete `SnldsConfig::obs_noise_var` and the `#[config(default = "5e-4")]`. `forward()` already takes `obs_noise_var: f32` as an argument; that's the single entry point. Rationale: `SnldsConfig` is layout (dims, K), not optimisation hyperparameters.

**Files**
- `crates/snlds-model/src/model.rs` — remove field + default from `SnldsConfig`.
- `crates/snlds-model/src/tests.rs` — unaffected (already passes literal at `forward` call sites).
- `crates/snlds-train/`, `crates/snlds-eval/`, `crates/snlds-msm/` — verify no readers of the removed field.

**Acceptance**
- `rg "SnldsConfig.*obs_noise_var|obs_noise_var.*SnldsConfig"` returns zero hits.
- `cargo test --workspace` green.
- Reviewer-blocking: any remaining "second copy" of `obs_noise_var` is a reject.

**Effort:** ~30 min. Net deletion.

---

### PR 2 — `chore/unwrap-audit` (issue #6)

**Approach:** Convert each non-test `.unwrap()` to either:
- `.expect("…")` when truly infallible by construction (static-literal `Normal::new`, contiguous-row slicing, etc.) — improves panic-site clarity.
- `?` / `anyhow::ensure!(...)` when it represents an upstream contract.

**Targets**

| File:line | Current | Recommended |
|-----------|---------|-------------|
| `crates/snlds-data/src/generate.rs:99,170,172,197,231` | `Normal::new(...).unwrap()` | `.expect("std must be > 0")` (args are static literals) |
| `crates/snlds-data/src/generate.rs:203` | `trans_probs.as_slice().unwrap()` | `.expect("transition row should be contiguous")` |
| `crates/snlds-train/src/warm_start.rs:86` | `best.expect("…")` | Convert to `anyhow::ensure!(restarts > 0, …)` upstream of the loop, propagate `Err` instead of panicking |
| `crates/snlds-core/src/hmm.rs:44` | `log_alphas.last().unwrap()` | `.expect("log_alphas non-empty after init push")` |

`tensor.into_data().to_vec().unwrap()` calls in `snlds-core` and `snlds-model` non-test code should also be audited; replace with `.expect("Burn TensorData::to_vec on owned data")` at minimum.

**Acceptance**
- `rg '\.unwrap\(\)' crates --type rust -g '!**/tests/**' -g '!**/test*.rs'` returns either zero hits or a documented allowlist.
- No behavior change in tests.

**Effort:** ~1 hr.

---

### PR 3 — `feat/m1-schema-v4-simulator-hparams` (issues #2, #3, #4)

**Approach:** bundle three data-crate cleanups into one schema bump (v3 → v4). Surface simulator hyperparameters on `GenConfig` with sensible defaults matching today's hardcoded values, persist them in `Manifest`, and make `pi_true` driven by an actual field.

**Scope**

#### 3a. `GenConfig` extension

```rust
pub struct GenConfig {
    // ... existing ...
    /// Std-dev of the Gaussian jitter added to z_0 around the per-state init mean.
    pub init_noise_std: f32,            // default 0.1
    /// Std-dev of the per-state init-mean prior.
    pub init_mean_std: f32,             // default 0.7
    /// Variance of the transition step noise added to z_t each step.
    pub transition_step_var: f32,       // default 0.05
    /// Hidden dimension of the leaky-ReLU emission network in the simulator.
    pub emission_hidden_dim: usize,     // default EMISSION_HIDDEN_DIM (= 8)
    /// Initial-state distribution. None = uniform; Some(arr) must be length num_states and sum to 1.
    pub initial_distribution: Option<Vec<f32>>,
}
```

Make `EMISSION_HIDDEN_DIM` the **default constructor value**, not the only allowed value. Consumers (snlds-model, tests) that need to know the simulator's emission H read it from `Manifest`.

#### 3b. Wire through `generate_split`

- Replace `Normal::new(0.0, 0.1).unwrap()` etc. with `Normal::new(0.0, cfg.init_noise_std)`.
- Replace `Array2::<f32>::zeros((..., EMISSION_HIDDEN_DIM))` with `cfg.emission_hidden_dim`.
- Initial-state sampling: if `cfg.initial_distribution.is_some()`, validate (length K, sums to 1, finite) and sample from it; else uniform. Compute `pi_true` from the actual sampling distribution.

#### 3c. Manifest schema v4

- Bump `MANIFEST_SCHEMA_VERSION = 4`.
- Add fields to `Manifest`: `init_noise_std`, `init_mean_std`, `transition_step_var`, `emission_hidden_dim`. (Don't include `initial_distribution` — `pi_true` already covers it as a tensor.)
- Update the rustdoc bump-history block.
- **Backwards-compat:** serde `default` attributes on the new fields so v3 manifests still load (with the prior hardcoded numbers as defaults). Document this in `M1.md`.

#### 3d. Tests

- Round-trip integration test: assert all four new manifest fields persist through serialize/deserialize.
- New test: non-uniform `initial_distribution` produces empirical first-state frequencies that match within tolerance for large `num_samples`.
- Existing tests get explicit defaults (no behavior change for the `Default::default()` path).

#### 3e. Doc updates

- `docs/M1.md`: schema v4 history row, expanded `GenConfig` table, deprecation note on `EMISSION_HIDDEN_DIM` constant (kept as default-only).
- `docs/PRD-burn-port.md`: changelog row, version bump.

#### 3f. Followups (out of scope for PR 3)

- **`snlds-gen` CLI flags** for the new simulator hyperparameters
  (`--init-noise-std`, `--init-mean-std`, `--transition-step-var`,
  `--emission-hidden-dim`, `--initial-distribution`). PR 3 leaves the
  `snlds-gen` CLI flags for these as a follow-up; they're consumable via the
  library API or by editing `crates/snlds-data/src/bin/snlds-gen.rs`.
  A `TODO(M1+)` breadcrumb in `main()` points back here.

**Acceptance**
- `cargo test -p snlds-data` green including new round-trip + non-uniform-π tests.
- Constructed `Manifest` with `schema_version: 3` (and missing the new fields) still deserialises.
- Reviewer-blocking: any remaining hardcoded simulator constant in `generate.rs` outside the `Default for GenConfig` impl is a reject.

**Effort:** ~3–4 hrs.

---

### PR 4 — `feat/markov-topology-pluggable` (issue #5) — **merged**

**Approach:** Replace the implicit fixed cyclic builder with `TransitionPattern { Cyclic { self_prob }, Provided(Array2<f32>) }` on `GenConfig`. `get_trans_mat(pattern, num_states)` validates shape and row-stochasticity; `generate_split` calls it once and passes the same `Array2<f32>` into `roll_sequences` and `TrainTest.q_true`. Cyclic `K=1` yields `[[1.0]]` (identity row-stochastic matrix).

**Files:** `crates/snlds-data/src/transitions.rs`, `generate.rs`, `lib.rs`; integration tests; [docs/PRD-burn-port.md](PRD-burn-port.md) changelog.

---

## Reviewer brief addendum

Add to the standard reviewer prompt for every PR going forward:

> **Hardcoded numeric constants are a blocker, not a nit, when they appear on the consumer side of a producer/consumer pair (e.g. eval crate vs train crate, model vs data) — even if the consumer-side scope is the focus of the branch.**
>
> **Reviewer concerns must all be addressed before merge, including nitpicks. "Non-blocking" is reserved for genuinely cosmetic suggestions and even those should be applied unless explicitly waived.**

That single addition would have caught the `obs_noise_var = 0.1` bug on the original M-Viz+ pass.

---

## Execution order & overall budget

| Order | PR | Time | Dependencies |
|-------|----|------|--------------|
| 1 | PR 1 (`obs_noise_var` dedupe) | 30 min | None |
| 2 | PR 2 (unwrap audit) | 1 hr | None |
| 3 | PR 3 (schema v4) | 3–4 hrs | Touches `snlds-data`; coordinate with any in-flight data work |
| 4 | PR 4 (`feat/markov-topology-pluggable`) | landed | After PR 3 |

**Total active work (PR 1–3):** ~5 hrs across three PRs; PR 4 tracked separately.

---

## Document history

| Date | Note |
|------|------|
| 2026-04-29 | PR 4 (Markov topology / `TransitionPattern`) merged; cyclic Q no longer fixed-only in code paths — configured via `GenConfig.transition`. |
| 2026-04-29 | Initial plan; triggered by post-merge audit of `feat/m-viz-graphs`. |
