# The per-trace root-cause axis — `rca/trace` and `rca/trace-core` per failed request (xs, s, m × five seeds)

Registered by `plans/rca-arms.md`, addendum 2026-10-31 (D-TBR-14: the benchmark pin moves to the commit carrying
`tracebench.score_rca_trace`; D-TBR-15: the axis, the uncapped selection rule R, the GPU device as a disclosed
replica of the frozen cells with the edge-set tolerance gate, the per-trace files out of git). Scorer
`tracebench.score_rca_trace` (`docs/scorers-service-rca.md` §4 in the benchmark, dated 2026-10-31). The probe runs
on one g5-class EC2 replica (`scripts/jobs/rca-trace-axis-26-10-31.sh`; device `cuda`), the scoring locally in the
`trace-bench-rca` environment (`scoretrace`, B = 2000, bootstrap seed 0; `reporttrace` → `tables/rca-trace/`).
No hit rate of any row was computed before the registration; the model-free rows below marked "(label side)"
were read after it as scale checks of the scorer.

## Premise checks (2026-10-31, seed 0 of xs, s, m; the released labels; before the registration)

| rung | case | kind | root cause | R (failed POST traces) | C (the fault took) | rc span the only failed span | earliest failed span is rc | services / trace |
|---|---|---|---|---|---|---|---|---|
| xs | 0 | degrade | tb-kamko | 1,128 | 623 | 9 | 365 | 3.9 |
| xs | 1 | crash | tb-razase | 758 | 48 | 4 | 29 | 4.6 |
| s | 0 | degrade | tb-razase | 20,390 | 1,844 | 138 | 1,178 | 6.2 |
| s | 1 | crash | tb-visliput | 13,838 | 1,442 | 28 | 1,126 | 5.0 |
| s | 2 | pool_exhaust | tb-tagan | 12,441 | 7,754 | 1,078 | 7,457 | 4.4 |
| m | 0 | degrade | tb-sitizo | 17,837 | 2,565 | 748 | 1,973 | 5.2 |
| m | 1 | crash | tb-tuzelsom | 12,092 | **30** | 5 | 22 | 5.2 |
| m | 2 | pool_exhaust | tb-zitaron | 8,508 | 889 | 46 | 740 | 6.7 |
| m | 3 | cache_flush | tb-batopet | 8,352 | 305 | 114 | 211 | 5.8 |
| m | 4 | degrade | tb-edge | 4,133 | 1,058 | 722 | 722 | 4.7 |

- P1 — R's phase test agrees with the labels' POST phase on every trace of the three corpora (0 mismatches); the
  label's `rc_took` counts 4xx, which the axis does not: 3 / 0 (xs), 3 / 2 / 6 (s), 3 / 0 / 0 / 3 / 8 (m) traces per
  case differ, hence the recomputed `took`. Splits of C at xs seed 0: 79 % train, 10 % val, 11 % test.
- P2 — `rc_positions_end` names exactly the root-cause service's operations on every label row (0 exceptions).
- P3 — `parent_pos` points to a later position on 97–98 % of links (the callee ends first); inversions kept.
- The m crash fault scores 30 traces; the `tb-edge` degrade at m has the root span as the only failed span in 68 %
  of its scored traces (its only caller is the synthetic client), so the hop-≥1 reading cannot name it there.
- Rule R over the 15 corpora (label-free, from the views): xs 1,347–1,886, s 46,669–65,880, m 50,922–58,437 failed
  POST traces per corpus; 546,573 in all, probed under two constructions ≈ 1.09 M probes.
- Scorer scale (seed 0, model-free rows, the view and the labels loaded once per cell): select 0.1 / 3.7 / 12 s,
  load 50 s at m, then under a second per row.

## Tolerance check (P4) — the GPU re-probe of the landed case-level selections against the landed CPU work files

2026-11-01, `tolerance` (records `results/<rung>/latent/seed=<k>/<arm slug>/trace/tolerance/`). Gate: mean per-trace
edge-set Jaccard ≥ 0.80 per run, positions identical. Mean Jaccard per run (the 500 window traces per case that both
sides probed: 755–1,000 per corpus at xs, where c = 3 skips short rows, 1,468–1,500 at s, 2,500 at m; 49,310 in all):

| rung | arm | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| xs | `rca/trace` | 0.913 | 0.940 | 0.980 | 0.965 | 0.934 |
| xs | `rca/trace-core` | 0.847 | 0.894 | 0.962 | 0.899 | 0.861 |
| s | `rca/trace` | 0.887 | 0.944 | 0.918 | 0.865 | 0.932 |
| s | `rca/trace-core` | 0.830 | 0.859 | 0.914 | 0.888 | **0.793** |
| m | `rca/trace` | 0.951 | 0.831 | 0.942 | 0.821 | 0.897 |
| m | `rca/trace-core` | 0.933 | 0.900 | 0.915 | 0.914 | 0.928 |

- Window positions are identical on all 49,310 compared traces (the check refuses otherwise): the selection and the
  view are device-free; only the probe's draws differ.
- **The gate fails on one run of 30: `s` seed 4, `rca/trace-core`, mean 0.793** (identical edge sets on 34 % of its
  traces against 33–94 % elsewhere); its case-level top-1 service agrees GPU vs CPU in all three cases. Under the
  registration a failed gate is a disclosed deviation and an **owner decision** (keep the run with this disclosure, or
  turn the gate into a diagnostic); the run's per-trace rows are in the tables as scored, nothing was re-run.
- **2026-11-02, owner decision: the gate is a diagnostic.** The 0.80 threshold (`TRACE_TOLERANCE_JACCARD`) stays as the
  reported reference value of the P4 check, not a pass/fail condition on a run; the `s` seed 4 `rca/trace-core` run
  stands as scored with the 0.793 disclosed beside it (positions identical, case top-1 unchanged), no re-run, no
  number moved. The paper's A5.5 names the value and the one run below it.
- The case-level top-1 service of the GPU re-probe equals the landed CPU one in 99 of 100 cases (the exception: m
  seed 2, `rca/trace`, case 0, mean Jaccard 0.937). The recomputed Avg@5 named beside it in the plan is not computed
  by `tolerance` (top-1 only).

## The axis (pooled over five seeds; cases = injected faults; the point is the mean over cases)

2026-11-01, `reporttrace` → `tables/rca-trace/` (`summary.json` sha256 `4a265160…`; B = 2000, bootstrap seed 0;
`rca-trace.md`). Cases 10 / 15 / 25 at xs / s / m; scored traces (C, the fault took) 3,641 / 50,979 / 16,758 of the
8,005 / 267,487 / 271,081 probed failed POST traces (R); no scored trace unprobed; traces whose root-cause span took
with a 4xx only (excluded from C) 4 / 80 / 50; non-client services per scored trace 3.87 / 6.09 / 5.96. Cell = hit@1
[95 % interval over cases] / hit@3 / exact set (mean set size):

| row | `xs` | `s` | `m` | pooled `xs`–`m` |
|---|---|---|---|---|
| `rca/trace/hop1` | 0.378 [0.280, 0.483] / 0.566 / 0.063 (1.99) | 0.401 [0.356, 0.445] / 0.725 / 0.081 (3.46) | 0.143 [0.090, 0.197] / 0.458 / 0.028 (3.90) | 0.267 [0.219, 0.317] / 0.560 / 0.051 (3.39) |
| `rca/trace/hop0` | 0.398 [0.266, 0.527] / 0.632 / 0.035 (2.66) | 0.678 [0.615, 0.740] / 0.887 / 0.062 (4.47) | 0.625 [0.548, 0.685] / 0.863 / 0.099 (4.67) | 0.595 [0.536, 0.647] / 0.824 / 0.075 (4.21) |
| `rca/trace-core/hop1` | 0.432 [0.330, 0.533] / 0.621 / 0.073 (2.06) | 0.451 [0.410, 0.490] / 0.730 / 0.104 (3.39) | 0.258 [0.189, 0.327] / 0.494 / 0.089 (3.11) | 0.351 [0.302, 0.401] / 0.590 / 0.090 (2.98) |
| `rca/trace-core/hop0` | 0.538 [0.386, 0.637] / 0.868 / 0.083 (3.00) | 0.740 [0.693, 0.790] / 0.929 / 0.066 (4.55) | 0.654 [0.584, 0.711] / 0.903 / 0.146 (4.03) | 0.656 [0.606, 0.701] / 0.904 / 0.110 (3.98) |
| `floor/trace-topology/hop1` | 0.589 [0.471, 0.706] / 0.830 / 0.146 (2.14) | 0.446 [0.286, 0.604] / 0.738 / 0.093 (3.71) | 0.191 [0.092, 0.301] / 0.390 / 0.052 (3.34) | 0.347 [0.263, 0.434] / 0.582 / 0.083 (3.21) |
| `floor/trace-topology/hop0` | 0.817 [0.755, 0.880] / 1.000 / 0.126 (3.17) | 0.808 [0.739, 0.866] / 0.982 / 0.066 (4.91) | 0.695 [0.629, 0.755] / 0.956 / 0.133 (4.46) | 0.753 [0.710, 0.798] / 0.972 / 0.112 (4.34) |
| `control/trace-random` | 0.246 [0.229, 0.263] / 0.737 / n/a | 0.175 [0.162, 0.189] / 0.523 / n/a | 0.199 [0.177, 0.225] / 0.569 / n/a | 0.201 [0.186, 0.217] / 0.589 / n/a |
| `control/trace-callees/hop1` | 0.407 [0.351, 0.459] / 0.790 / n/a | 0.296 [0.243, 0.345] / 0.635 / n/a | 0.174 [0.120, 0.229] / 0.384 / n/a | 0.257 [0.217, 0.298] / 0.540 / n/a |
| `control/trace-callees/hop0` | 0.402 [0.345, 0.462] / 0.868 / n/a | 0.283 [0.252, 0.318] / 0.667 / n/a | 0.351 [0.312, 0.387] / 0.723 / n/a | 0.341 [0.313, 0.368] / 0.735 / n/a |
| `broadcast/control/rule` | 0.405 [0.137, 0.678] / 0.751 / n/a | 0.620 [0.408, 0.818] / 0.809 / n/a | 0.399 [0.246, 0.558] / 0.716 / n/a | 0.466 [0.343, 0.583] / 0.751 / n/a |
| `broadcast/rca/baro` | 0.370 [0.100, 0.670] / 0.842 / n/a | 0.402 [0.179, 0.630] / 0.726 / n/a | 0.381 [0.233, 0.530] / 0.767 / n/a | 0.385 [0.276, 0.494] / 0.770 / n/a |
| `broadcast/rca/baro-bocpd` | 0.169 [0.000, 0.408] / 0.281 / n/a | 0.327 [0.139, 0.530] / 0.617 / n/a | 0.204 [0.097, 0.323] / 0.477 / n/a | 0.234 [0.143, 0.329] / 0.480 / n/a |
| `broadcast/rca/causalrca/latency-50` | 0.233 [0.090, 0.423] / 0.738 / n/a | 0.136 [0.042, 0.263] / 0.554 / n/a | 0.164 [0.082, 0.262] / 0.562 / n/a | 0.169 [0.106, 0.242] / 0.594 / n/a |
| `broadcast/rca/causalrca/latency-95` | 0.312 [0.138, 0.508] / 0.668 / n/a | 0.155 [0.049, 0.301] / 0.448 / n/a | 0.116 [0.043, 0.202] / 0.506 / n/a | 0.167 [0.103, 0.241] / 0.521 / n/a |
| `broadcast/rca/circa` | 0.238 [0.000, 0.477] / 0.622 / n/a | 0.282 [0.106, 0.496] / 0.763 / n/a | 0.139 [0.061, 0.227] / 0.636 / n/a | 0.202 [0.122, 0.296] / 0.671 / n/a |
| `broadcast/rca/circa-topology` | 0.100 [0.000, 0.300] / 0.500 / n/a | 0.133 [0.000, 0.333] / 0.371 / n/a | 0.254 [0.146, 0.369] / 0.708 / n/a | 0.187 [0.105, 0.280] / 0.565 / n/a |
| `broadcast/rca/pc-pagerank` | 0.200 [0.000, 0.500] / 0.800 / n/a | 0.317 [0.140, 0.514] / 0.542 / n/a | 0.160 [0.067, 0.269] / 0.627 / n/a | 0.215 [0.130, 0.316] / 0.636 / n/a |
| `broadcast/rca/rcd` | 0.665 [0.417, 0.897] / 0.884 / n/a | 0.820 [0.620, 1.000] / 0.945 / n/a | 0.477 [0.292, 0.661] / 0.732 / n/a | 0.617 [0.494, 0.743] / 0.826 / n/a |
| `broadcast/rca/tracerca` | 0.265 [0.000, 0.531] / 0.700 / n/a | 0.412 [0.158, 0.667] / 0.660 / n/a | 0.107 [0.027, 0.203] / 0.446 / n/a | 0.231 [0.128, 0.344] / 0.561 / n/a |

- Position hit@1 / hit@3 of the arms in their own window (hop1): `rca/trace` 0.387 / 0.537, 0.431 / 0.702, 0.130 /
  0.409; `rca/trace-core` 0.430 / 0.590, 0.465 / 0.712, 0.267 / 0.479 (xs / s / m).
- Per fault kind, pooled xs–m (hit@1, hop1): `rca/trace` cache_flush 0.137 · crash 0.322 · degrade 0.272 ·
  pool_exhaust 0.242; `rca/trace-core` 0.180 · 0.387 · 0.333 · 0.417; the floor 0.171 · 0.329 · 0.424 · 0.307;
  broadcast RCD 0.000 · 0.378 · 0.760 · 1.000 (5 / 15 / 20 / 10 cases).
- The broadcast rows are a case-level ranking read per trace: a row is as good as its case top-1 on the case's
  traces, so their intervals are as wide as Table 1's; RCD's trace-weighted hit@1 is 0.92–0.96 at xs / s and 0.73 at
  m — a few large cases it ranks right carry most traces.
- The m floor at hop ≥ 1 sits at random (0.191 against 0.199). The `tb-edge` degrade is case 4 of every m seed (5
  of 25 cases) and scores 0.00 on all five: its only caller is the synthetic client, so the root span is the only
  failed non-client span in most of its traces (premise check) and no rule that excludes a failed span's own service
  can name it. Twelve further m cases score ≤ 0.09 at hop ≥ 1; with hop 0 every m case rises (0.33–0.92) and the
  floor reaches 0.695.

## Hypotheses

Mechanical, `summary.json`; all four fail.

- **H-trace-perseq-random — fail (at m).** Every trace-reading row's hit@1 interval lies above random at xs and s; at
  m none does: `rca/trace` 0.143 [0.090, 0.197], `rca/trace-core` 0.258 [0.189, 0.327], the floor 0.191 [0.092,
  0.301], against 0.199.
- **H-trace-perseq-topology — fail (2 of 12, in the arm's favour).** The call-tree floor's hit@1 is at least each
  TRACE row's at the same reading except `rca/trace-core` at hop1 on m (+0.067) and s (+0.004); at xs the floor leads
  by 0.16–0.42.
- **H-trace-perseq-broadcast — fail (every rung, both arms).** The best broadcast row is RCD on every rung (0.665 /
  0.820 / 0.477), above both TRACE rows at hop1; the broadcast rule control (0.405 / 0.620 / 0.399) is above
  `rca/trace` on every rung and above `rca/trace-core` at s and m.
- **H-trace-perseq-self — fail (at m).** With hop 0 admitted both TRACE rows pass the best broadcast row at m (0.625 /
  0.654 against 0.477) where hop1 falls below it; at xs and s the sign is negative under both readings.

## Disclosures that travel with the rows

- Device: the probes ran on a GPU (`cuda`); CPU and CUDA draw different noise under the same seed, so this read is a
  disclosed replica of the frozen cells, not a bit-identical reproduction of the landed CPU case-level rows (which
  stay as landed). The per-trace seed index is the trace's position in the selected list, so the uncapped selection
  gives a trace a different draw than the capped one.
- The backbone saw the train split of the same corpus; 80 % of the scored traces are train-split traces on every rung
  (2,931 / 40,725 / 13,413). The training-overlap audit shows no gap: trace-weighted hit@1, train against val + test,
  differs by at most 0.03 on every row and rung (e.g. `rca/trace-core` hop1 0.409 / 0.385, 0.429 / 0.427, 0.286 /
  0.273); the per-split rows are in the score records (`by_split`).
- A root-cause position inside the arm's context (the first c − 1 spans; c = 1–3) or beyond the 64-token cap is
  unreachable for the arm and counted (`n_rc_unreachable_context`, `n_rc_unreachable_truncation`); the floor and the
  controls are defined on the whole trace. The end view puts the callee first, so the root span tends to open the row:
  **the root-cause span lies in the context in 74.8 % of the scored traces for `rca/trace` at xs** (c = 2–3), 18.2 %
  at s (c = 2 on seeds 1–4) and none at m (c = 1); 15.1 % for `rca/trace-core` at xs (c = 1–2), none at s and m.
  There the arm can name the root-cause service only through another of its spans inside the window. No root-cause
  position lies past the 64-token cap.
- The shared rows (floor, controls, broadcasts) ride both arms' cells; their position metrics and context counts are
  in the first named arm's window (`rca/trace`); their service metrics are identical in both cells (checked on all
  225 shared row pairs). `reporttrace` first refused the landed records because it compared those window-coordinate
  fields too; fixed in `e7a167d9` before any table was written (both refused attempts failed before pooling a row;
  their records are kept untracked in `out/reporttrace-failed-26-11-01/`).
- The box records name code `ed5f71eb` with `dirty: true`: the job's untracked `input/` (the backbones) and
  `exports/` on the box (neither is gitignored); the replica cloned the branch at that commit. The pinned benchmark
  reports its version as 0.3.0 at `df4a345f`.
- The addendum and D-TBR-15 speak of "eleven" broadcast rows; their own list names ten (nine landed arms, CausalRCA
  as p50 and p95), and those ten are the rows scored — a miscount in the registration text, no row missing.
- The per-trace rankings and per-trace score rows are not tracked (up to ≈ 60 MB per m run); their sha256 ride the
  `rank-trace` and `score-trace` records and the files live on the run's prefix and in the local scratch.
