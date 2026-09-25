# Reference-arm plan (pre-registered 2026-09-24, before the engine code)

Committed before `engine.py`, `discover.py` and `sweep.py` exist (PRD
scenario 19). Every hypothesis names a metric, a comparison and a numeric
threshold, and is scored pass or fail in `findings/` after each rung's test
read (scenario 20). Changes are dated addenda at the end, never edits in
place. Deviation ids are RUN.md's D-SB-n.

## 1. Arms

| arm | probe (forward-sharing unit) | statistic thresholded | what it is |
|---|---|---|---|
| `trace/core` | `core`: the `SampleLevelCausalDiscovery` tensor build, strategy `full`, ancestral history with `g = min(3, c)` | `calc_lag_info_gain`: per-particle Bernoulli KL (cause observed ‖ cause noised), then the mean over particles, non-finite coerced (shipped path) | what the author recommends |
| `trace/cli` | `cli-full`: `compute_cmi_matrix` internals, strategy `full`, real prefix | mean over particles, then the KL | the documented command-line path; differs from `trace/core` in exactly the history source and the averaging order |
| `trace/cli-atomic` | `cli-atomic`: strategy `atomic` (only the candidate cause replaced) | the atomic KL against the fully-real baseline | the command-line path at its shipped default; differs from `trace/cli` in the construction only |

Paths (Bug Policy): `shipped` (v0.1.9 algebra: noise over `[0, V)`, float32
clamp), `fixed-kl` (fix PR #1 only; the same forward and draws as `shipped`,
common random numbers), `fixed` (every merged fix: PR #1 + PR #3 noise over
real ids; a separate forward). Cuts: `frozen` (the validation-swept τ) for
every arm; `shipped` (the tool's own pooled-percentile rule with lag decay,
`CLI_SHIPPED_RULE`, no truth) for the two cli arms, inheriting `(c, N)` from
the frozen sibling cell. Cell key `<arm>/<path>/<cut>/<grain>`.

## 2. Model and sequences

One backbone per corpus (`plans/caps.md`), trained on `end-session`, probed
on both grains; `end` ordering; sequence = `BOS + ids + EOS` under
`--max-len 64`; BOS counts toward `c`. The type-level score of a token pair is
the max over its within-sequence occurrences (mean and count recorded);
within-operation pairs and special tokens are dropped at projection. Every
probed sequence's full strict-upper-triangle matrix is stored (D-SB-11).

## 3. Grids (validation sweep, blind)

- `c`: request `{1, 2, 3}`, session `{2, 4, 8}` (D-SB-8); `g = min(3, c)`.
- `N`: `{2, 8, 32}`; `xl`: `{2, 8, 16}` (D-SB-9, `plans/caps.md`).
- τ (score-side, `scoresweep`): half-decades `1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
  3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1` (12 values; `3e-5` is the paper's
  Table 2 threshold).
- Floor: the benchmark's default `0.05` for selection; every floor of the
  sweep reported.
- `--probe-amp none` (float32) is the reference precision (the CLI loads the
  model without a dtype); `bf16` only by dated addendum, paired.
- Every cell: the validation split's sample per `plans/caps.md`, `head`
  order.

## 4. Freeze rule

Per cell `<arm>/<path>/frozen/<grain>`: the full-grid argmax of validation
directed F1 at the default floor over `(c, N, τ)`; ties → smaller `N`, then
smaller `c`, then larger τ. The shipped cut of a cli arm inherits the frozen
cell's `(c, N)` and applies the tool's rule at test time (no freeze value).
The freeze is one dated file per corpus, committed before the test read, and
the test read asserts the commit is an ancestor, predates the run, and every
frozen value equals the command line and the model hash (PRD scenarios 13,
29).

## 5. Coverage rule

`annotate` reports the fraction of truth-directed edges whose token pair
co-occurs in the probed sample (the reachable-recall ceiling). If it is below
`0.90` on either grain of a corpus, `--num-sequences` is raised in a new
dated validation replica before any freeze of that corpus.

## 6. Hypotheses

- **H-estimator (core vs cli at the frozen cut).** At their own frozen
  `(τ, c, N)`, `trace/core/shipped/frozen` and `trace/cli/shipped/frozen`
  differ in directed F1 at the default floor by **≤ 0.05** on every corpus
  and grain (five-seed mean of the paired per-seed difference). *Basis:* the
  sibling's gate read −0.005 between its two readings. *Fail:* any (rung,
  variant, grain) where the mean paired difference exceeds 0.05 in
  magnitude. The `c = 1` request cell, where `g = 1` makes the two tensor
  builds coincide, isolates the averaging order and is reported as its own
  row.
- **H-construction (cli vs cli-atomic).** At the frozen cut, per-lag recall
  at lag ≥ 2 of `trace/cli-atomic` is **≥ 2×** that of `trace/cli` on every
  corpus where `trace/cli`'s lag-≥ 2 recall is below 0.2 (the lab's
  lag-graded limit). *Fail:* any such corpus where the ratio is below 2.
- **H-hazard (the float32 clamp, Goal 3).** On the `shipped` path, the
  corrupted-cell count (collapsed + saturated) is **> 0** on at least one
  corpus of every rung from `m` upward, and wherever a saturated cell exists
  it sits at the **top of the shipped ranking** (a `3.4e38` type-level max).
  *Fail:* a rung ≥ `m` with zero corrupted cells on all ten corpora, or a
  saturated cell that does not top its ranking. Reported, never gated.
- **H-cut (the shipped rule, Goal 3).** `trace/cli/shipped/shipped` yields an
  **empty edge set** on at least one corpus whose corrupted-cell count is
  > 0 (an induced empty prediction); on every other corpus the
  shipped-versus-frozen difference in directed F1 is reported. *Fail:* no
  induced empty prediction anywhere while corrupted cells exist.
- **H-fix (paired shipped vs fixed, Bug Policy).** `fixed-kl` directed F1 at
  the default floor is **≥ shipped − 0.01** on every cell (five-seed mean of
  the paired difference), and `fixed` versus `fixed-kl` is reported.
  *Fail:* any cell where the fix costs more than 0.01.
- **H-lag.** At the frozen cut, lag-1 recall is **≥ 0.8×** the coverage
  ceiling and lag-≥ 3 recall is **< 0.2×** it for `trace/core` and
  `trace/cli` on every corpus. *Fail:* either bound violated.
- **H-sat.** The frozen `N` is **≤ 8** on at least **80 %** of reference
  cells, and the validation F1 at `N = 32` exceeds that at `N = 8` by
  **< 0.01** on those cells. *Fail:* either clause.
- **H-regime.** The oracle score ε̂ is **< 0.1** on every corpus. Reported
  only; an out-of-regime model is labelled, never dropped.

## 7. What is not a hypothesis

Absolute F1 levels, SID / AID values, the latent-variant bidirected axis
(structurally zero for every arm) and the per-sequence axis (its own plan)
are reported without a pass / fail clause.

## Addenda

(none)

## Addenda

### 2026-09-25 — after the xs calibration run (owner-reviewed; `plans/caps.md` addendum of the same date)

- **§4 Freeze rule — model.** The frozen model is the argmin-validation checkpoint written by
  `pretrain --model-choice argmin-val` (caps addendum); the `1.01×` trigger, the argmin re-sweep and
  `freeze --alt-val-tables / --checkpoint-choice` are withdrawn. The freeze record carries
  `diagnostics`: the pretrain record's `model_choice`, its oracle at the chosen and at the final
  checkpoint, and per frozen cell the reachable-recall ceiling and an `at_grid_edge` flag.
- **§4 Freeze rule — grid edges.** A frozen τ that sits at either end of its grid is recorded
  (`at_grid_edge`), never refused: a bottom-edge argmax is a finding about the arm (its best cut
  is "every positive score"), not a grid defect. At xs the session argmax of `trace/core` and
  `trace/cli-atomic` sat at the top (`3e-1`) and the request argmax of `baseline/granger` and
  `trace/cli-atomic` at the bottom, so the grids widen: **τ = half-decades `1e-7 … 3`** (16
  values: `1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1,
  3`). A scoresweep is CPU-only over the stored matrices, so a widened grid needs no new probe.
- **§5 Coverage rule → reported.** The reachable-recall ceiling is a class-2 instrument check
  (caps addendum): reported beside every headline, in the freeze per cell and in `annotate`,
  with no threshold. At xs the whole validation split gave 0.477 (request, `c = 1`) and 0.592
  (session, `c = 2`), so the rule's action — a larger sample — could not reach 0.90.
- **H-regime.** Unchanged in substance (reported only); the value reported is ε̂ at the frozen
  checkpoint. xs `latent/seed=0`: ε̂ = 0.62 at the last checkpoint, ≈ 0.22 at step 1000 —
  out of regime either way; a result at xs is labelled so.
- **H-sat, note.** At xs the request-grain F1 is flat in N to ± 0.005 (saturated at N = 2); the
  session grain prefers N = 32 by ≈ 0.02 on one seed. Scored at the rung's test read as written.

