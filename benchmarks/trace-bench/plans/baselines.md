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

(none)

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

