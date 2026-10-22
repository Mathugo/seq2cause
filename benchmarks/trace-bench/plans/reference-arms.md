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

### 2026-10-05 — `xl`: the τ grid reaches zero (owner decision, before any `xl` freeze or test read)

- **What the `xl` seed-0 validation tables showed** (registered grid `1e-7 … 3`; request grain,
  `c = 1`; 48,578 pairs scored by each probe from the 10,000-sequence sample):

  | arm | positive scores | positive, below the grid floor | exactly zero or negative | F1 at the floor |
  |---|---|---|---|---|
  | `trace/core` | 39,460 | 44 – 59 | 9,118 | 0.0578 |
  | `trace/cli` | 39,220 – 39,460 | 1,102 – 1,475 | 9,118 – 9,358 | 0.0575 |
  | `trace/cli-atomic` | 48,206 – 48,578 | 1,419 – 1,967 | 0 – 372 | 0.0599 |

  Every reference cell at the request grain froze at the bottom of the grid, and the pairs left
  below it still carried a marginal precision of 0.036 – 0.059 against the break-even of
  F1 / 2 ≈ 0.03 — the cells were still rising where the grid stopped. At the session grain every
  reference cell has an interior argmax (`3e-5` or `1e-4`); nothing changes there.
- **§3 Grids — τ.** For `xl` the grid is **`0, 1e-9, 1e-8, 3e-8`** followed by the sixteen
  half-decades `1e-7 … 3` (20 values). Selection is strict (`score > τ`), so `τ = 0` is the cut
  the 2026-09-25 addendum described in words — "every positive score" — and a bottom-edge
  argmax now means exactly that. The freeze rule and its tie-break (larger τ) are unchanged.
- **No negative τ.** A candidate that also kept the zero and negative scores would be the cut
  "every pair the probe scored", which is co-occurrence in the sample, not a threshold on the
  arm's evidence; on the seed-0 request tables it scores F1 0.0600, above `trace/core` and
  `trace/cli` and equal to `trace/cli-atomic`. It is not a frozen candidate. It is recorded as a
  **reference line**: the scorer's verdict on the full ranking (`scoresweep`:
  `every_scored_pair`; `freeze`: `every_scored_pair_f1` per cell), reported beside every `xl`
  validation headline. The zero scores of `trace/core` and `trace/cli` are pairs the probe saw
  but its construction never tested; `trace/cli-atomic` tests every pair and has none.
- **What the extension is expected to do, and not to do.** It removes a floor that differed by
  arm (the baselines' addendum of the same date carries the larger case). It does not make the
  request grain discriminating: with the 10k sample reaching at most 3.7 % of the truth edges,
  the F1-optimal cut of every arm is everything it scored positively, so the request-grain
  ordering at `xl` reads as the count of positively scored pairs. The same reading applies to
  `l`, where the best swept request cell of seed 0 (0.0932) already equalled the reference line
  (0.0930); it is carried in the `l` and `xl` findings.
- **Scope.** `xl` only. The `l` rung is closed on the registered grid — its freezes, test reads
  and tables stand as landed, with their grid-edge cells recorded. The selection is changed on
  validation data alone and before any `xl` test read exists.
- **Mechanics.** No new probe: the five `xl` validation sweeps are re-scored on the CPU from
  their synced score arrays (`scoresweep` with the extended grids, a second scoring run per
  seed). The first scoring runs, on the registered grid, are kept as records; the rows the two
  grids share must be equal in both.

### 2026-10-05, later — the re-score ran one commit early; the reference line is taken from its counts

The five `xl` re-score runs were started while the change that records the scorer's
`every_scored_pair` line was still under review, so their boxes ran the previous commit: the
extended grids of the addendum above (those are arguments of the run), every swept row and the
coverage block as they would be on the merged code, and no `every_scored_pair` key. Nothing a
freeze selects on is affected, and the runs are not repeated. The reference line for those tables
is taken from the counts the scorer returned for the same full-ranking call — true positives =
ceiling × truth, predictions = the scored pairs inside the universe, F1 = 2·tp / (predictions +
truth) — and the freeze names the source per cell (`every_scored_pair_source`: `scorer` or
`coverage`). The identity is asserted on the fixture, where both are present, and the scorer's
conventions it rests on were checked on all 15,960 swept rows of the ten first-pass `xl`
validation tables (true plus false positives = predictions inside the universe; precision and
F1 from those counts; true positives plus false negatives = truth) with no exception.

### 2026-10-22 — sensitivity reads: N = 64 / 128 and a session context of 16 (validation only; before any such run)

Source: the co-author's review of 2026-10-19 (point 4: the paper's N = 128 against this grid's
{2, 8, 32}, and session contexts above 8) and the lab's reply of 2026-10-21, which commits to
"N = 64 / 128 and a session context of 16 as validation-only sensitivity rows on xs and s".

- **Scope.** xs and s, latent, **seed 0**, the `end` view, the validation split. Validation
  only: no freeze, no test read, nothing selected on these cells. The landed freezes, test reads,
  tables and verdicts stand as landed.
- **Backbone.** The landed seed-0 backbone of each rung, the model of
  `freezes/2026-09-25-xs-latent-s0.json` and `freezes/2026-09-27-s-latent-s0.json`; its hash is
  asserted equal to the freeze's at landing. No new pretrain.
- **Cells: two factors, each against the registered grid.**
  - (a) *Particles.* The particle probes (`core`, `cli-full`, `cli-atomic`) × both noises × the
    registered `c` of each grain (request {1, 2, 3}, session {2, 4, 8}) × **`N` ∈ {64, 128}**,
    `g = min(3, c)`. Saliency and Shapley have no `N` axis and are not re-read.
  - (b) *Session context.* The five probes (saliency and Shapley at `N = 0`, noise `all`) ×
    **`c = 16`** × the registered `N` {2, 8, 32}, `g = 3`, session grain only (request
    sequences are short by construction: median 5 spans, p99 11).
  - Everything else as registered for the rung: the sample of `plans/caps.md` (xs the whole
    split; s 10,000 / 5,000 with the Shapley sample 2,000 / 500; `head`), `--max-len 64`,
    `--max-lag 63`, `--probe-amp none`, the shipped rule, `--memory-cap-gb 20`, and the rung's
    scoresweep grids (τ `1e-7 … 3`, Granger `1e-5 … 3`, quantiles p50 … p99).
- **Memory.** The D-SB-12 estimate on the landed s backbone (V = 323) at the largest new cell
  (session, `N = 128`, `c = 2`) is 12.9 GiB, under the 20 GiB cap; the estimate was 1.7× the
  measured device peak at `N = 32`. xs (V = 79) is lower at every cell.
- **Staging.** xs as `smallest-rung`; s names the landed xs seed-0 `sweep` of the same grain
  by its ledger entry, as the s end-view sweeps did. Nothing stages on these reads, so they get
  no ledger entries.
- **Report.** One document, `tables/sensitivity-latent.{json,md}`, labelled "validation only,
  seed 0, one corpus per rung, not re-selected". Per rung, grain and frozen cell
  `<arm>/<path>/frozen/<grain>` of the landed seed-0 freeze, two rows:
  - the landed frozen value, `(c, N, τ)` and validation directed F1 at the default floor;
  - for each new `(c, N)` of that arm and path, the validation F1 at the argmax over τ of that
    `(c, N)` on the rung's grids, and its difference from the landed frozen value.
  The per-`(c, N)` rows of the landed validation table at `N` ∈ {2, 8, 32} (both grains) and
  `c` ∈ {2, 4, 8} (session) are printed beside them from the landed table, so each factor reads
  as a curve. The table is written by a tracked script committed before the landing. Records
  land under `results-sensitivity/<rung>/latent/seed=0/` (stage records and the val tables;
  score arrays stay off the repository).
- **Expectation, stated in advance (reported, not scored as a hypothesis).** H-sat's band
  carried up the grid: at every cell the best validation F1 at `N` ∈ {64, 128} exceeds the best
  at `N` ≤ 32 by less than 0.01, and at the session grain the best at `c = 16` exceeds the best
  at `c` ≤ 8 by less than 0.01. A larger gain is reported as it is, with the caution that it
  is one seed of one corpus per rung.
- **The sample at `c = 16` (measured 2026-10-22, before any such run).** A sequence is probed
  only when its window holds a position pair (`len(ids) − c ≥ 2`, BOS and EOS included), so
  `c = 16` reads only the longer sessions. On the seed-0 validation samples the probed counts at
  `c` = 8 / 16 are 387 / 125 of 437 (xs) and 4,466 / 2,522 of 5,000 (s); over seeds 0–4 the
  `c = 16` share of the `c = 8` count is 0.32–0.47 (xs) and 0.56–0.81 (s). The `c = 16` rows are
  therefore printed with their probed count and reachable-recall ceiling beside the F1, and a
  fall at `c = 16` against `c = 8` is read as the smaller sample first, not as the context
  hurting; the expectation above is checked on F1 as registered.
