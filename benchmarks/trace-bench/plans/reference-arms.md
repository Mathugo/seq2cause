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

### 2026-10-31 — the context family: score × context on the full-staircase core probe, and the atomic CMI arm (seq2cause v0.3.1; before any context-family code or read)

Source: the method author's mail of 2026-10-30 — keep TRACE as a baseline, name the configuration run, and
re-run a score × context 2 x 2 at N = 8 on the frozen backbones under the same sweep-and-freeze protocol —
and his v0.3.1 release of 2026-10-31, which puts the CMI score on the command-line path too; owner decisions
of 2026-10-30 (the 2 x 2 on the core probe; N = 8; rungs; hop 0 on the root-cause axes) and 2026-10-31 (one
atomic CMI arm, keep layout only; `none` at both grains after the P1 check below; both paths and both cuts on
the atomic arm; both root-cause axes for it). The paper's results are frozen (2026-10-29): the family runs in
the harness only and the paper decides later; nothing landed moves. Package: seq2cause v0.3.1 `280ed948`
(merged 2026-10-30 / 10-31, D-SB-18 and D-SB-19; `constants.CONTEXT_FAMILY_VERSION`); every landed cell
stays a v0.1.9 read.

**§1 Arms.** Five arms in their own tuple `CONTEXT_ARMS` (family `context`; `REFERENCE_ARMS` and every
landed table are untouched); class `reference`; cut `frozen`; the rung's τ grid.

| arm | probe (forward-sharing unit) | statistic thresholded | context (request / session) | c grid (request / session) | g | paths | cuts |
|---|---|---|---|---|---|---|---|
| `trace/core-ig-overwrite` | `core` (the landed pass) | `calc_lag_info_gain` (as `trace/core`) | overwrite / overwrite | {1, 2, 3} / {2, 4, 8} | min(3, c) | `fixed-kl` | `frozen` |
| `trace/core-cmi-overwrite` | `core` | `calc_lag_cmi` (v0.3.0) | overwrite / overwrite | {1, 2, 3} / {2, 4, 8} | min(3, c) | `fixed-kl` | `frozen` |
| `trace/core-ig-keep` | `core-keep` (new: the core build with `context_mode = "none"`) | `calc_lag_info_gain` | none / none | {1} / {1} | 1 | `fixed-kl` | `frozen` |
| `trace/core-cmi-keep` | `core-keep` | `calc_lag_cmi` | none / none | {1} / {1} | 1 | `fixed-kl` | `frozen` |
| `trace/cli-atomic-cmi-keep` | `cli-atomic-cmi` (new: v0.3.1 `compute_cmi_matrix(score="cmi", strategy="atomic")` internals, real prefix) | the atomic CMI | `context_len 1, prepend_len 0` / the same | {1} / {1} | 1 (carried; the cli probes ignore g) | `fixed-kl`, `fixed` | `frozen`, `shipped` |

The two overwrite arms are read from the landed `core` pass (every future `core` pass writes their two
columns beside `trace/core`'s three and Granger's two; `trace-core-ig-overwrite__fixed-kl` is by
construction the same column as `trace-core__fixed-kl` — identity check (ii)). The arm set is one atomic
arm, not an atomic 2 x 2: on the cli-atomic path the keep layout is the landed `c = 1` cell, so "information
gain, keep" at request already exists in the landed `trace/cli-atomic` sweeps and is read from them (§7).

**§2 Scores.** *Information gain* — the landed statistic: the Bernoulli KL of the effect's probability with
the observed cause against its mean over the cause-noised particles; unbounded. *CMI* — the author's
`causal_strength = "CMI"` (v0.3.0 `calc_lag_cmi`, core class) and `score = "cmi"` (v0.3.1
`compute_cmi_matrix`): `CMI = π · KL(q1 ‖ m) + (1 − π) · KL(q0 ‖ m)`, `m = π q1 + (1 − π) q0`, with π the
model probability of the observed cause, q1 the effect's probability with the cause present and q0 with the
cause absent; bounded by log 2; always in log space (its `fixed-kl` reading is v0.3.1's own default:
log-space KL, noise over `[0, V)`; there is no `shipped`-algebra CMI, the author never released one). On the
core probe the cause-absent branch is the staircase's uniform draw and the CMI is read from the same
log-probabilities as the information gain of the same pass (per particle, then the mean, as
`calc_lag_info_gain`). On the atomic probe the cause-absent branch is **a draw from the model**,
`P(X_j | x_<j, X_j ≠ x_j)` (`model_sample_excluding_observed`, N particles, the log-mean over particles,
then the KL — the cli path's order), with one unintervened forward giving π and q1 as the same vector: a new
forward with its own draws, not a re-read of the landed atomic pass. The atomic arm's `fixed` path is the
same construction with the specials excluded from the model draw (`min_id = 4`, a second forward), not a
uniform draw over real ids; its `shipped` cut is the tool's own pooled-percentile rule with lag decay
(`CLI_SHIPPED_RULE`) on the CMI scores, inheriting `(c, N)` from the frozen sibling, as for the landed cli arms.

**§3 Context layouts and the P1 outcome.** *overwrite* (the landed layout): the first g observed tokens
kept, positions g … c − 1 overwritten by tokens sampled from the model, positions c … L − 1 tested. *none*:
the first g observed tokens kept, no sampled token, positions g … L − 1 tested; the family fixes `g = c = 1`
(BOS only), so every observed token is tested and the window offset equals c as on every landed record.
*prepend* (the author's suggestion for sessions: c sampled tokens placed before the full sequence) **is not
run**, by the premise check P1 of 2026-10-30 / 31 (seed-0 backbones of the five rungs, the 200 longest
validation sessions, N = 8, g = 1, v0.3.0's prepend construction; artefacts off the repository). The
registered rule was: c enters the keep session grid iff the mean NLL of the observed tokens at model
positions ≥ 66 (the backbone's position budget, `max_len + 2`) rises by ≤ 0.10 nats and the whole-sequence
mean by ≤ 5 %. The positions past the budget are harmless everywhere (−0.047 … +0.080 nats), but the
whole-sequence NLL rises by xs +18 / +38 / +68 %, s +9 / +15 / +20 %, m +6 / +16 / +31 %, l +3 / +3 / +6 %,
xl +2 / +3 / +6 % at c = 2 / 4 / 8 — only l and xl at c ≤ 4 pass. By observed index the damage sits on the
first tokens (t = 1: +4.5 … +5.5 nats on every rung) and decays, at m with c = 8 still +2.3 nats at t = 6–10.
The cause is structural: the benchmark's sessions are head-truncated (`encode_sequence` keeps `ids[:max_len]`),
so every session the backbone saw starts at its true beginning, and a sampled past before it is a
construction the backbone never saw; the author's prepend presumes windows cut from a longer stream. The
in-distribution way to keep every observed token is `none`, which the request grain used already (the landed
`trace/core` request freezes chose c = 1 on every s and m seed). Owner decision 2026-10-31: `none` at both
grains. At the session grain c = 1 is a new cell for every arm (the landed session grid is {2, 4, 8}).

**§4 Grids.** N = 8 (the author's recommendation: N = 8 against N = 128 moves his AP by < 0.01; the landed
H-sat band agrees); c as the §1 table; τ the rung's registered grid (§3 above and the addenda of 2026-10-05);
the shipped rule `CLI_SHIPPED_RULE`; `--max-len 64`, `--max-lag 63`, `--probe-amp none`,
`--memory-cap-gb 20`; the validation sample of `plans/caps.md` per rung, `head` order; the request-grain
note of 2026-10-05 (from l up every request cell freezes at the bottom of its τ grid; the "every observed
pair" reference line printed beside each headline) travels with every table of the family.

**§5 Protocol.** The frozen backbones of the landed freezes are reused, hash-asserted (no pretrain). Per rung
and seed, the validation sweeps: `core` overwrite (request c 1 2 3; session c 2 4 8), `core-keep` none
(c 1 both grains), `cli-atomic-cmi` (c 1 both grains, noises `all` and `real`), all N = 8, staged by
`--prior-rung-record` as the landed sweeps; `scoresweep --arms` names the five arms (the `trace/core`,
Granger and shipped-path columns of the new `core` pass are not re-scored into the family's tables; they
serve identity check (i) in a scratch scoresweep); `freeze` derives the family `context` from the arms
(mixing refused, as `floor`), applies the registered rule — the argmax of validation directed F1 at the
floor 0.05 over (c, N = 8, τ); ties → smaller N, smaller c, larger τ; τ < 0 set aside — over the family's
grid only (`n_rows_outside_family_grid` recorded), and writes `freezes/<date>-<rung>-latent-s<k>-context.json`
(`freeze@3` plus `context_mode` and `window_offset` per cell); the test reads, `annotate`, `seqscore` and
the root-cause reads bind to it as every landed cell does (`freezecheck` compares `context_mode`). Rungs:
xs, s, m on every axis (type level, per sequence, root cause); l and xl on the type-level and per-sequence
axes only. Records land under `results-context/<rung>/latent/seed=<k>/` (cells, `freeze/`, `val-sweep/`,
`test-read/`, `test-score/`; score arrays off the repository), ledger entries added as the landed ones,
`report --results-dir results-context` with a reasons file marking the required reference and floor cells
"not in this family" → `tables/<rung>-latent-context/`, and a tracked `contexttable.py` →
`tables/context-latent/` (per rung and grain the 2 x 2 of the core family, the atomic arm beside the landed
`trace/cli-atomic/fixed-kl/frozen` and `/shipped/frozen`, paired differences against the landed
`trace/core/fixed-kl/frozen`, τ drift). Findings in `findings/context-latent.md`.

**§6 Caveats.** (a) `calc_lag_cmi` fills no corrupted-cell statistics: the core CMI rows carry the
information-gain `fixed-kl` counts of the same pass (`counts_source: ig-same-pass`); the atomic CMI arm has no
counts at all (`counts_source: none`; v0.3.1's `score = "cmi"` branch reads neither `kl_mode` nor `stats`) —
H-hazard is not scored for the family. (b) The CMI is bounded by log 2 ≈ 0.69, so τ ≥ 1 of the grid is
empty for it (harmless) and the bottom of the grid may bind more often; `at_grid_edge` records it, a grid
extension downward is a dated addendum plus a CPU re-score (the 2026-10-05 precedent). (c) The keep cells
coincide with the overwrite cells at c = 1 on the information gain (identity check (iii)); at request the
family's keep rows are therefore the landed c = 1 cells re-read at N = 8, and the context effect of the
family is a session-grain effect. (d) The atomic arm's two paths are two forwards per sequence; its
per-trace root-cause replica is its own layout (its draws differ from the landed `trace/cli-atomic`
forward; no dedup).

**§7 Identity checks (landing checks; a failure blocks the landing).** (i) The new `core` overwrite pass's
`trace/core` `fixed-kl` and `shipped` rows at the landed (c, N = 8, τ) reproduce the landed validation-table
rows of the same seed (F1, precision, recall, `n_edges`) — a scratch scoresweep over those columns.
(ii) `trace-core-ig-overwrite__fixed-kl == trace-core__fixed-kl` bit for bit in every score file of the pass.
(iii) `trace/core-ig-keep` at (c = 1, N = 8) equals `trace/core-ig-overwrite` at (c = 1, N = 8) bit for bit
at request (`none` with g = c = 1 and `overwrite` with g = c = 1 build the same tensors). (iv) The atomic
arm's `fixed-kl` column equals v0.3.1 `compute_cmi_matrix(score="cmi", strategy="atomic", context_len=1,
prepend_len=0, noise_min_id=0)` bit for bit on the fixture under one seed, and its `shipped` cut equals the
v0.3.1 CLI's (`--score cmi --strategy atomic --context-len 1`) — the parity tests of D-SB-2 extended.

**§8 Hypotheses (scored in `findings/context-latent.md` after each rung's test read; paired over the five
seeds with `report.paired_table`'s test; "p" is its p-value).**
- **H-ctx-cmi-f1.** |CMI − information gain| in directed F1 at the floor 0.05 is ≤ 0.01 at matched context
  and grain (`trace/core-cmi-overwrite` vs `trace/core-ig-overwrite`; `trace/core-cmi-keep` vs
  `trace/core-ig-keep`), five-seed mean of the paired difference. *Fail:* |Δ| > 0.01 with p < 0.05.
- **H-ctx-cmi-tau.** The τ drift `log10(τ*_test / τ_frozen)`, with τ*_test the test-side argmax recomputed
  by `contexttable.py` through the benchmark's scorer (at l and xl, where the test score arrays are not
  re-swept, the F1 drift `F1_test(τ_frozen) − F1_val(τ_frozen)` only): the CMI arm's mean |drift| ≤ the
  information-gain arm's at matched context and grain (the author's claim: the best threshold drifts about
  half as much). *Fail:* CMI exceeds information gain by more than one grid step (half a decade), or at
  l / xl its F1 drift is larger by > 0.01.
- **H-ctx-keep-perseq.** At the session grain the keep arm's per-sequence pooled directed F1 ≥ the
  overwrite arm's at matched score (every observed token is a candidate; nothing dropped). *Fail:* keep below
  overwrite by > 0.01 with p < 0.05.
- **H-ctx-keep-context-share.** On the per-trace root-cause axis `n_rc_unreachable_context / n_rc = 0` for
  every keep arm (c = 1 reaches every position), and the hop-0 hit@1 of `rca/trace-ig-keep` ≥ that of
  `rca/trace-ig-overwrite` on xs, s and m. *Fail:* a keep arm with a context-unreachable root-cause position,
  or keep < overwrite on any rung.
- **H-atomic-cmi-f1.** `trace/cli-atomic-cmi-keep/fixed-kl/frozen` directed F1 at the floor 0.05 ≥ the
  landed `trace/cli-atomic/fixed-kl/frozen` on both grains (the author's claim: AP 0.72 → 0.81 on his
  synthetic benchmarks with atomic mediators). *Fail:* below by > 0.01 with p < 0.05.
- **H-atomic-cmi-tau.** The atomic CMI arm's τ drift (as H-ctx-cmi-tau) ≤ the landed `trace/cli-atomic`'s.
  *Fail:* as H-ctx-cmi-tau.
- **H-atomic-cmi-shipped.** `trace/cli-atomic-cmi-keep/fixed-kl/shipped` is non-empty on every seed and its
  directed F1 ≥ the landed `trace/cli-atomic/fixed-kl/shipped`'s. *Fail:* an empty cut on any seed, or
  below the landed shipped cut by > 0.01 with p < 0.05.
Descriptive, not hypotheses: the absolute levels; `fixed` vs `fixed-kl` on the atomic arm (the model draw
with and without the specials); the share of CMI scores at the bottom of the grid.

**§9 Premise checks.** *P1* (prepend at session) — run 2026-10-30 / 31, numbers in §3; closed by the
owner's decision. *P3* (label side, seed 0 of xs / s / m, before this registration, no model read): of the
per-trace axis's scored traces C (the 2026-10-31 selection, reproduced exactly: 623 / 48, 1,844 / 1,442 /
7,754, 2,565 / 30 / 889 / 305 / 1,058 traces per case), the share whose root-cause positions are **exactly
{window index 0}** under c = 1 — the earliest-ending span of the end-ordered trace, which the full-staircase
core probe never tests as a cause at any c (the shipped staircase's first row is untested) and which only the
hop-0 reading can name — is xs 8 % / 10 %, s 5 % / 9 % / 5 %, m 11 % / 0 % / 10 % / 24 % / 0 % (the m crash
and edge-service faults: none); the share with **any** root-cause position at index 0 is xs 39 % / 50 %,
s 14 % / 34 % / 74 %, m 39 % / 0 % / 35 % / 30 % / 0 %. The atomic probe tests cause 0, the core probe does
not; the per-trace rows print both shares beside H-ctx-keep-context-share. *Before the first job:* **P1′**
identity check (iii) on the seed-0 backbone over 50 validation sequences per grain on the CPU; **P2** the
parity tests of §7 (iv) and the core CMI parity against `SampleLevelCausalDiscovery.run(causal_strength="CMI")`
on the fixture, CMI ∈ [0, log 2], `nan_to_num` never firing on the core path; **P4′** the D-SB-12 memory
estimate for `cli-atomic-cmi` and `core-keep` at N = 8, c = 1, L = 66 on the xl vocabulary (the same tensors
as the landed probes; under the 20 GiB cap), and the CPU cost per trace of the atomic layout at m, which
sizes the per-trace replica. Results as a dated note under this addendum.

**§10 Deviations (RUN.md).** D-SB-19 (the v0.3.1 merge, recorded 2026-10-31); **D-SB-20** the `core-keep`
probe and the `none` layout, `context_mode` and `window_offset` on every discover record and frozen cell,
the two extra columns of every future `core` pass (tracked pre-change scripts replay at their commit);
**D-SB-21** `scoresweep --arms`, the derived family `context`, the `-context` freeze name, `results-context/`;
**D-SB-22** the CMI corrupted-cell counts (§6 a); **D-SB-23** the `cli-atomic-cmi` probe (two forwards, the
model draw, `fixed` = the specials excluded from the model draw, both cuts).
