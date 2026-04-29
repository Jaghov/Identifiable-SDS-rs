# M-Viz milestone — Ground-truth Rerun visualization

**PRD:** [§9 Milestones](PRD-burn-port.md#9-milestones) (row **M-Viz**), [§5.4 Visualization](PRD-burn-port.md#54-visualization-rerun).

**Status:** ✅ complete — merged to main 2026-04-29.

---

## Goal

New crate **`snlds-viz`** that loads a completed **M1** dataset (`sequences.safetensors` + `metadata.json`) and logs ground-truth sequences to a **Rerun** session (`.rrd` export or live viewer). No trained model required.

Scope extension vs PRD: **image rendering** (`_draw` port — latents → synthetic video frames) is included here rather than deferred to M6, because the frames require only the already-generated latents (`dim_latent == 2`) and no encoder.

---

## Deliverables

### 1. Crate `snlds-viz`

New workspace member `crates/snlds-viz/` with:

| Module | Responsibility |
|--------|---------------|
| `render.rs` | `draw_sequence(latents, res) -> Array4<f32>` — port of Python `_draw` |
| `log.rs` | Rerun logging helpers (entity paths, type mapping) |
| `bin/snlds-viz.rs` | CLI: load dataset → log to Rerun → save `.rrd` |

Dependencies (in `snlds-viz` only — `snlds-data` and `snlds-core` stay Rerun-free):

| Crate | Version | Role |
|-------|---------|------|
| `snlds-data` | path | load `TrainTest` + `Manifest` |
| `rerun` | `0.31` | Rerun Rust SDK |
| `ndarray` | `0.16` | frame buffer in `render.rs` |
| `anyhow` | `1` | error propagation |
| `clap` | `4` | CLI args |

---

### 2. `render.rs` — `_draw` port

Translates a 2D latent trajectory (`[T, 2]`) into a float32 video (`[T, res, res, 3]`), pixel values in `[0, 1]`.

**World bounds** (fixed, matching Python):

| Axis | Min | Max |
|------|-----|-----|
| x (col) | −3 | 4 |
| y (row) | −4 | 3 |

**Algorithm per frame:**
1. Black canvas `[res, res, 3]`
2. Map `(x, y)` → pixel `(col, row)` via `space_res = (max_x - min_x) / res`
3. Fill circle at `(col, row)`, radius `floor(1 / space_res)`, color `[173/255, 146/255, 0]`
4. 2×2 box blur
5. After all frames: add background `[81/255, 88/255, 93/255]`, clamp to `[0, 1]`

Implemented with plain ndarray arithmetic — no new image-processing dependency.

Only invoked when `manifest.dim_latent == 2`; skipped silently otherwise.

---

### 3. Rerun entity schema

Timeline: two axes per recording — `sequence` (batch index `n`) and `time` (step `t`).

| Entity path | Rerun type | Content |
|-------------|-----------|---------|
| `snlds/latent/z` | `Points2D` / `LineStrips2D` | True latent `z_t` (2D path, one strip per sequence) |
| `snlds/obs/x` | `Points2D` / `LineStrips2D` | Observation `x_t` (2D path when `dim_obs == 2`) |
| `snlds/state/s` | `Scalar` | True discrete state `s_t` per timestep |
| `snlds/render/frame` | `Image` | Rendered frame from `_draw` (only when `dim_latent == 2`) |

Entity paths are **stable** across M-Viz and M-Viz+ (posteriors added later under `snlds/state/gamma_k`).

---

### 4. CLI (`snlds-viz`)

```
snlds-viz --input <dir>            # dir containing sequences.safetensors + metadata.json
          [--sequences <n>]        # how many sequences to log (default: 5)
          [--split train|test]     # which split (default: train)
          [--output <path.rrd>]    # save recording (default: snlds_gt.rrd)
          [--render]               # enable _draw image frames (requires dim_latent == 2)
          [--spawn]                # spawn live Rerun viewer
```

---

## Completion checklist

- [x] Workspace member **`snlds-viz`** added to root `Cargo.toml`
- [x] **`render.rs`**: `draw_sequence` implemented; tests: `draw_sequence_ball_pixel_lit`, `draw_sequence_values_in_range`, `draw_sequence_blur_softens_edges`
- [x] **`log.rs`**: entity paths and Rerun types defined — `log_latent_z`, `log_obs_x`, `log_state_s`, `log_render_frames`
- [x] **`bin/snlds-viz.rs`**: loads dataset, logs `z_t`, `x_t`, `s_t`, saves `.rrd`
- [x] **`--render`** flag logs video frames via `draw_sequence` when `dim_latent == 2`; `anyhow::bail!` if dim != 2
- [x] **Unit test**: `draw_sequence_ball_pixel_lit` — 1-step known input, ball center pixel > 0.5, all values in [0,1]
- [x] **Integration smoke test**: `cli_smoke_writes_rrd` — cosine dataset to tempdir, assert `.rrd` size > 0
- [x] [docs/PRD-burn-port.md](PRD-burn-port.md) **§8.5** updated with `rerun = "0.31"` pin
- [x] [docs/PRD-burn-port.md](PRD-burn-port.md) §9 links this tracker ([M-Viz.md](M-Viz.md) footnote in milestone table)

---

## Testing requirements

Inherit **M0** gates: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

| Area | Requirement |
|------|-------------|
| **`draw_sequence`** | 1-frame reference: known `(x, y)` → expected ball pixel lit, background pixel correct, values in `[0, 1]` |
| **`draw_sequence` blur** | After blur, ball edge pixels differ from unblurred (non-regression — not brittle value check) |
| **CLI smoke** | `.rrd` file written, size > 0; no panics on `cosine_tiny_cfg` tiny dataset |
| **`dim_latent != 2`** | `--render` flag logs a warning and skips frames rather than panicking |

---

## Out of scope

- Posterior `γ_{t,k}` visualization — **M-Viz+** (after M3)
- Reconstructed observations `x̂_t` — **M-Viz+**
- Training scalar timelines (ELBO, MSE, temperature) — **M-Viz+**
- CNN encoder / image-input training — **M6**
- Neuroscience datasets

---

## Downstream

**M-Viz+** (after M3) extends entity paths with `snlds/state/gamma_k` posteriors, `snlds/obs/x_hat` reconstructions, and training scalar timelines, all under the same entity schema defined here.

---

## Updates

| Item | Description |
|------|-------------|
| Box blur | Always divides by 4.0 (zero-padding), matching `scipy.ndimage.uniform_filter` — kernel reads left/top neighbours so ball bleeds right/down |
| `log_latent_z` / `log_obs_x` | Take `ArrayView2<f32>` directly (not `&[f32] + t_len`) |
| Rerun 0.31 API | `LineStrips2D::new([points])`, `Scalars::single(val)`, `Image::from_elements(&u8_pixels, [w,h], ColorModel::RGB)`, `rec.set_time_sequence("timeline", val)` |

---

## Document history

| Date | Note |
|------|------|
| 2026-04-29 | Initial tracker. Includes `_draw` rendering pulled forward from M6 (latents-only, no encoder needed). |
| 2026-04-29 | PRD §9 milestone table links [M-Viz.md](M-Viz.md) (doc 1.13). |
| 2026-04-29 | M-Viz complete and merged to main; checklist and Updates filled in. |
