# Baseline-arm plan (pre-registered 2026-09-24, before the engine code)

The three shipped read-outs of the same frozen backbone (PRD Arm Register),
exactly the baselines the TRACE paper's Table 2 reports; the benchmark's
scorer supplies its own trivial baselines (`shd_empty_*`,
`shd_topk_directed`) and nothing is built for those.

## 1. Arms

| arm | probe | statistic | paths | sample |
|---|---|---|---|---|
| `baseline/granger` | `core` (the reference arm's tensor, no extra forward) | `calc_granger_score`, mode `diff` (probability gain, in `[−1, 1]`) | `shipped`, `fixed` (the noise draw) | the reference arm's |
| `baseline/saliency` | `saliency`: `calc_neural_saliency` (captum InputXGradient on the input embeddings, L2 over the hidden dimension) | non-negative | `none` | the reference arm's |
| `baseline/shapley` | `shapley`: `calc_neural_shapley` (captum ShapleyValueSampling over input ids, `n_samples 10`, PAD baseline) | signed | `none` | its own: `plans/caps.md` |

## 2. Grids

- `c`: the reference grids (request `{1, 2, 3}`, session `{2, 4, 8}`);
  saliency and Shapley have no particle axis (`N = null` in the freeze).
- τ: granger — half-decades `1e-4 … 1` (12 values); saliency and Shapley —
  label-free quantiles `{p50, p80, p90, p95, p99}` of the pooled validation
  scores, the absolute value recorded at freeze.
- Freeze rule, floor and coverage rule as in `plans/reference-arms.md`.

## 3. Hypotheses

- **H-baselines.** At the frozen cut, every baseline's directed F1 at the
  default floor is **below `trace/core/shipped/frozen`'s** on every rung,
  variant and grain (five-seed mean of the paired difference, same model
  hash). *Fail:* any cell where a baseline ties or wins.
- **H-granger-agree.** `baseline/granger` and `trace/core` on the same
  forward rank the same top-10 % of validation token pairs with overlap
  **≥ 0.5** (Jaccard) on every corpus. *Fail:* any corpus below 0.5.
- **H-shapley-cost.** Shapley's wall clock per sequence at the session grain
  is **≥ 5×** the core probe's at `N = 8`; reported as the cost that
  justifies its own sample.

## Addenda

### 2026-09-25 — after the xs calibration run (owner-reviewed)

- **§2 Grids — granger.** The pre-registered text said "half-decades `1e-4 … 1` (12 values)";
  the xs replica ran the nine half-decades `1e-4 … 1` (the text miscounted). The request-grain
  argmax sat at the bottom (`1e-4`), so the grid widens to **`1e-5 … 3`** (12 values: `1e-5,
  3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3`); a bottom-edge argmax is
  recorded (`at_grid_edge`), never refused (arm plan addendum of the same date).
- **§2 Grids — quantiles.** Unchanged (`p50` won at xs on both grains for saliency and
  Shapley; the quantile family already brackets it).
- **Coverage rule** → reported, per the arm plan's addendum.

### 2026-10-05 — `xl`: the Granger grid and the quantile family reach their floors (owner decision; arm plan addendum of the same date)

- **What the `xl` seed-0 validation tables showed** (request grain, `c = 1`, 48,578 scored
  pairs): Granger has 31,170 positive scores, 1,908 of them below its grid floor `1e-5`, and
  17,408 at or below zero; saliency scores every pair positively and its lowest quantile, `p50`,
  drops half of them. The pairs below each floor carried a marginal precision of 0.074
  (Granger), 0.079 (saliency) and 0.111 (Shapley, on its own sample) against a break-even of
  F1 / 2 ≈ 0.03. So the lowest cut kept 64 % (Granger) and 53 % (saliency) of the scored pairs
  where the reference arms' kept 83 – 100 %: part of the request-grain ordering was the grid.
- **The quantile family had been at its floor since `xs`.** The 2026-09-25 entry above reads
  "`p50` won at xs … the quantile family already brackets it"; `p50` is the family's lowest
  member, so it did not. Saliency froze at `p50` at the request grain, and Shapley at both
  grains, on all five `l` seeds, unflagged — the freeze marked grid edges for absolute grids
  only. From `freeze@3` a quantile-sourced τ at either end of the family is flagged
  (`at_grid_edge`) like any other.
- **§2 Grids — granger.** For `xl`: **`0, 1e-7, 3e-7, 1e-6, 3e-6`** followed by the twelve
  half-decades `1e-5 … 3` (17 values).
- **§2 Grids — quantiles.** For `xl`: **`p0, p10, p20, p30, p40, p50, p80, p90, p95, p99`**
  (10 values). Selection is strict, so `p0` keeps every pair above the smallest score.
- **Reference line, scope and mechanics** as in the arm plan's addendum: no negative τ; the cut
  "every scored pair" is recorded beside each cell, not frozen; `xl` only, `l` stands as landed;
  CPU-only re-score of the five validation sweeps. At the session grain Granger and saliency
  have interior optima on the registered grid and are not expected to move; Shapley sits at
  `p50` there too and may.
