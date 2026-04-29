# Agent work split (Cursor-agent vs Claude)

Who does what so two assistants do not trample the same files or PR scope. **Update this file** if ownership shifts.

---

## Cursor-agent (this workspace / implementation-first)

**Owns**

- Cargo workspace: `crates/**`, root `Cargo.toml`, `Cargo.lock`, `.cargo/`, `.github/workflows/`
- Executable quality: `cargo fmt`, `cargo clippy -D warnings`, `cargo test --workspace`
- Git mechanics you run locally or in CI: branches, commits, `gh pr` (no blanket `git add .`)

**Suggested first PRs after `docs/` is tracked**

1. `chore: stop ignoring docs/` — `.gitignore` only (if not already merged).
2. Code PR(s): e.g. pending **`snlds-data`** changes, **`snlds-core`**, new crates (`snlds-model`, `snlds-train`, …) as milestones land.

**Avoid unless coordinated**

- Large purely-editorial rewrites of `PRD-burn-port.md` narrative (hand off to Claude for prose; Cursor-agent can still do version/changelog rows when merging).

---

## Claude (prose / milestone trackers / PRD structure)

**Owns**

- Milestone trackers: `docs/M*.md`, `docs/M-Viz.md`, `docs/M-Viz+.md`
- PRD substance: `docs/PRD-burn-port.md` sections on scope, milestones, success criteria, risks — **wording**, consistency across docs, cross-links
- README “how to run” copy when it reflects docs milestones (coordinate before duplicating pins)

**Suggested doc PR stack** (each branch off `main` after prior merge, or one stacked branch — your choice; Cursor-agent can execute the git steps Claude specifies)

| Slice | Files (stage only these) |
|-------|----------------------------|
| C1 | `docs/M0.md` |
| C2 | `docs/M1.md` |
| C3 | `docs/M2.md` |
| C4 | `docs/M-Viz.md` |
| C5 | `docs/M3.md` |
| C6 | `docs/M-Viz+.md` |
| C7 | `docs/M4.md` |
| C8 | `docs/M5.md` |
| C9 | `docs/M6.md` |
| C10 | `docs/PRD-burn-port.md` + `docs/AGENT_WORK_SPLIT.md` (if PRD should link all trackers in one go) |

*Alternative:* one PR with all `docs/*.md` + PRD if you prefer fewer reviews; still stage named paths only.

**Avoid unless coordinated**

- Changing Rust sources or `Cargo.toml` without an explicit handoff.

---

## Handoff rules

1. **One owner per PR** — don’t mix Cursor-agent Rust edits with Claude doc rewrites in the same commit without agreement.
2. **Pins and versions** — Burn / `rerun` / toolchain: decide in **PRD §8.5** (Claude edits prose; Cursor-agent bumps `Cargo.toml` + lockfile).
3. **Milestone order** — Implementation order stays [PRD §9](PRD-burn-port.md#9-milestones); agents only parallelize **non-overlapping files**.

---

## Document history

| Date       | Note |
|------------|------|
| 2026-04-29 | Initial split; `docs/` tracked in git. |
