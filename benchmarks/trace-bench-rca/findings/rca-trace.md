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


## The context family on the per-trace axis, landed 2026-11-08: `rca/trace-{ig,cmi}-{overwrite,keep}`, `rca/trace-atomic-cmi-keep`, `control/trace-earliest` per failed request (xs, s, m × five seeds)

The five context arms (`plans/rca-arms.md`, addendum "2026-10-31, later"; D-TBR-16 … 17) and the earliest-error control
(D-TBR-19) on this axis, run on one EC2 replica (`seq2cause-bench-26-11-06-rca-ctx-trace.tfvars`, g5.xlarge; job script
`scripts/jobs/rca-ctx-trace-26-11-06.sh`, PR #88; code `3bdba66f`; the chain 7.5 h, synced on the first attempt) with the probes
on the A10G (`--device cuda`). Same selection as the landed axis — rule R, no cap: 8,005 / 267,487 / 271,081 failed POST traces
per rung, the landed run's selections exactly (R is arm-independent); scored traces C = 3,641 / 50,979 / 16,758, as landed. The
probes read the frozen request cells of the committed context freezes (`freezes/2026-11-01-<rung>-latent-s<k>-context.json`, the
`fixed-kl` path; seq2cause v0.3.1) in the probe layouts of each seed's freeze: the information-gain and CMI cells of one core layout
share one forward (`--probe-arms`; the second arm re-ranks the first arm's work files, `runner_reused`, their sha256 recorded), the
two keep cells likewise, the atomic CMI cell its own — three probes per corpus, four on s seeds 2 and 4 where the CMI overwrite cell
froze at c = 2 (47 probes, 28 re-ranks; 638–987 s per s probe, 711–892 s per m probe, 13–15 ms per trace). Every reading is written per trace
(`ranking-trace@2`: `hop1`, `hop0`, the `source` rule; `hop0` the primary). The earliest control ranks the same traces by the start of
each service's earliest non-OK span (one reading, `earliest`). Staging: xs smallest rung, then the previous rung's `rank-trace`
record of the same arm made earlier in the chain. **No CPU row of these cells exists, so the tolerance check of D-TBR-15 is not
applicable: the read is disclosed as a GPU replica of the frozen cells** (the per-trace seed stream of 2026-10-30; CPU and CUDA draw
differently under one seed). Scored locally (`scoretrace`, B = 2000, bootstrap seed 0; `--rows all` for the arms, `arm-only` for the
control); `reporttrace` → `tables/rca-trace-context/` (31 rows × 4 tables). The landed `tables/rca-trace/` are untouched — a re-run
of their call reproduces all 76 pooled files byte-equal (its `summary.json` would gain only the `H-trace-earliest: not run` key the
code acquired after that landing) — and the shared rows (floor, controls, broadcasts) of the five new cells equal the landed cells'
on every service metric (same C). Cell = hit@1 [95 % interval over cases] / hit@3 / exact set (mean set size):

| row | `xs` | `s` | `m` | pooled `xs`–`m` |
|---|---|---|---|---|
| `rca/trace-ig-overwrite/hop1` | 0.394 [0.283, 0.511] / 0.568 / 0.059 (2.00) | 0.459 [0.408, 0.502] / 0.729 / 0.098 (3.38) | 0.258 [0.188, 0.327] / 0.494 / 0.093 (3.05) | 0.345 [0.294, 0.397] / 0.579 / 0.088 (2.94) |
| `rca/trace-ig-overwrite/hop0` | 0.492 [0.340, 0.603] / 0.834 / 0.075 (2.97) | 0.741 [0.696, 0.789] / 0.927 / 0.063 (4.55) | 0.659 [0.594, 0.709] / 0.904 / 0.149 (3.99) | 0.650 [0.598, 0.696] / 0.897 / 0.109 (3.95) |
| `rca/trace-ig-overwrite/source` | 0.532 [0.356, 0.670] / 0.738 / 0.323 (1.50) | 0.571 [0.502, 0.644] / 0.954 / 0.299 (2.05) | 0.619 [0.559, 0.674] / 0.850 / 0.483 (1.56) | 0.587 [0.539, 0.635] / 0.858 / 0.396 (1.70) |
| `rca/trace-cmi-overwrite/hop1` | 0.387 [0.285, 0.494] / 0.555 / 0.056 (2.00) | 0.323 [0.267, 0.374] / 0.605 / 0.065 (3.38) | 0.174 [0.115, 0.235] / 0.430 / 0.054 (3.34) | 0.261 [0.214, 0.310] / 0.507 / 0.058 (3.08) |
| `rca/trace-cmi-overwrite/hop0` | 0.490 [0.344, 0.597] / 0.828 / 0.075 (2.98) | 0.659 [0.583, 0.728] / 0.862 / 0.060 (4.55) | 0.627 [0.555, 0.687] / 0.856 / 0.110 (4.27) | 0.609 [0.556, 0.655] / 0.852 / 0.088 (4.10) |
| `rca/trace-cmi-overwrite/source` | 0.538 [0.362, 0.676] / 0.744 / 0.314 (1.48) | 0.635 [0.574, 0.703] / 0.894 / 0.359 (1.80) | 0.631 [0.570, 0.684] / 0.806 / 0.520 (1.43) | 0.613 [0.563, 0.659] / 0.820 / 0.430 (1.55) |
| `rca/trace-ig-keep/hop1` | 0.499 [0.410, 0.587] / 0.732 / 0.088 (2.20) | 0.459 [0.408, 0.502] / 0.729 / 0.098 (3.38) | 0.258 [0.188, 0.327] / 0.494 / 0.093 (3.05) | 0.366 [0.315, 0.418] / 0.612 / 0.094 (2.98) |
| `rca/trace-ig-keep/hop0` | 0.615 [0.446, 0.723] / 0.939 / 0.089 (3.07) | 0.741 [0.696, 0.789] / 0.927 / 0.063 (4.55) | 0.659 [0.594, 0.709] / 0.904 / 0.149 (3.99) | 0.675 [0.628, 0.717] / 0.918 / 0.111 (3.97) |
| `rca/trace-ig-keep/source` | 0.617 [0.440, 0.744] / 0.772 / 0.443 (1.46) | 0.571 [0.502, 0.644] / 0.954 / 0.299 (2.05) | 0.619 [0.559, 0.674] / 0.850 / 0.483 (1.56) | 0.604 [0.558, 0.649] / 0.865 / 0.420 (1.69) |
| `rca/trace-cmi-keep/hop1` | 0.512 [0.436, 0.584] / 0.745 / 0.087 (2.24) | 0.340 [0.283, 0.391] / 0.635 / 0.064 (3.55) | 0.174 [0.115, 0.235] / 0.430 / 0.054 (3.34) | 0.291 [0.242, 0.343] / 0.555 / 0.063 (3.18) |
| `rca/trace-cmi-keep/hop0` | 0.628 [0.462, 0.734] / 0.930 / 0.088 (3.09) | 0.675 [0.609, 0.732] / 0.886 / 0.053 (4.71) | 0.627 [0.555, 0.687] / 0.856 / 0.110 (4.27) | 0.641 [0.592, 0.685] / 0.880 / 0.089 (4.16) |
| `rca/trace-cmi-keep/source` | 0.623 [0.444, 0.746] / 0.774 / 0.441 (1.42) | 0.620 [0.559, 0.687] / 0.898 / 0.355 (1.83) | 0.631 [0.570, 0.684] / 0.806 / 0.520 (1.43) | 0.626 [0.580, 0.670] / 0.827 / 0.454 (1.55) |
| `rca/trace-atomic-cmi-keep/hop1` | 0.496 [0.381, 0.606] / 0.878 / 0.083 (2.60) | 0.213 [0.155, 0.276] / 0.593 / 0.032 (3.69) | 0.117 [0.070, 0.169] / 0.377 / 0.021 (4.07) | 0.222 [0.170, 0.277] / 0.542 / 0.037 (3.66) |
| `rca/trace-atomic-cmi-keep/hop0` | 0.627 [0.449, 0.750] / 0.963 / 0.064 (3.21) | 0.635 [0.576, 0.690] / 0.866 / 0.047 (4.92) | 0.614 [0.534, 0.676] / 0.821 / 0.071 (4.88) | 0.623 [0.574, 0.668] / 0.863 / 0.063 (4.55) |
| `rca/trace-atomic-cmi-keep/source` | 0.622 [0.441, 0.750] / 0.684 / 0.558 (1.14) | 0.639 [0.583, 0.703] / 0.944 / 0.350 (1.91) | 0.624 [0.561, 0.677] / 0.780 / 0.528 (1.37) | 0.628 [0.582, 0.671] / 0.810 / 0.481 (1.48) |
| `control/trace-earliest/earliest` | 0.206 [0.130, 0.281] / 0.809 / 0.000 (4.20) | 0.132 [0.090, 0.188] / 0.751 / 0.000 (6.13) | 0.428 [0.310, 0.557] / 0.881 / 0.008 (5.92) | 0.295 [0.224, 0.374] / 0.827 / 0.004 (5.64) |
| `floor/trace-topology/hop1` | 0.589 [0.471, 0.706] / 0.830 / 0.146 (2.14) | 0.446 [0.286, 0.604] / 0.738 / 0.093 (3.71) | 0.191 [0.092, 0.301] / 0.390 / 0.052 (3.34) | 0.347 [0.263, 0.434] / 0.582 / 0.083 (3.21) |
| `floor/trace-topology/hop0` | 0.817 [0.755, 0.880] / 1.000 / 0.126 (3.17) | 0.808 [0.739, 0.866] / 0.982 / 0.066 (4.91) | 0.695 [0.629, 0.755] / 0.956 / 0.133 (4.46) | 0.753 [0.710, 0.798] / 0.972 / 0.112 (4.34) |
| `control/trace-random` | 0.246 [0.229, 0.263] / 0.737 / n/a | 0.175 [0.162, 0.189] / 0.523 / n/a | 0.199 [0.177, 0.225] / 0.569 / n/a | 0.201 [0.186, 0.217] / 0.589 / n/a |
| `control/trace-callees/hop1` | 0.407 [0.351, 0.459] / 0.790 / n/a | 0.296 [0.243, 0.345] / 0.635 / n/a | 0.174 [0.120, 0.229] / 0.384 / n/a | 0.257 [0.217, 0.298] / 0.540 / n/a |
| `control/trace-callees/hop0` | 0.402 [0.345, 0.462] / 0.868 / n/a | 0.283 [0.252, 0.318] / 0.667 / n/a | 0.351 [0.312, 0.387] / 0.723 / n/a | 0.341 [0.313, 0.368] / 0.735 / n/a |
| `broadcast/control/rule` | 0.405 [0.137, 0.678] / 0.751 / n/a | 0.620 [0.408, 0.818] / 0.809 / n/a | 0.399 [0.246, 0.558] / 0.716 / n/a | 0.466 [0.343, 0.583] / 0.751 / n/a |
| `broadcast/rca/rcd` | 0.665 [0.417, 0.897] / 0.884 / n/a | 0.820 [0.620, 1.000] / 0.945 / n/a | 0.477 [0.292, 0.661] / 0.732 / n/a | 0.617 [0.494, 0.743] / 0.826 / n/a |

- The landed arms on this axis, for comparison (2026-11-01; `hop1` / `hop0` hit@1, xs / s / m): `rca/trace` 0.378 / 0.398, 0.401 /
  0.678, 0.143 / 0.625; `rca/trace-core` 0.432 / 0.538, 0.451 / 0.740, 0.258 / 0.654. `rca/trace-ig-overwrite` reproduces
  `rca/trace-core`'s `hop1` at m exactly (0.258) and is within 0.01 of it elsewhere: the same construction at c = 1 on the
  `fixed-kl` path against the `shipped` path (P1′ of the reference harness: bit-identical matrices at equal cells; the two freeze at
  different τ).
- Position hit@1 / hit@3 in the arm's own window (`hop0`): information gain 0.52 / 0.79 (xs, overwrite; 0.64 / 0.90 keep), 0.76 /
  0.91 (s), 0.66 / 0.87 (m); CMI 0.52 / 0.79 (0.64 / 0.90), 0.67 / 0.83, 0.63 / 0.82; atomic CMI 0.63 / 0.91, 0.62 / 0.83, 0.61 / 0.78.
  Under `source` 0.61–0.63 / 0.68–0.76 (xs), 0.69–0.70 / 0.88–0.94 (s), 0.63 / 0.78–0.85 (m).
- Per fault kind, pooled xs–m (hit@1): `rca/trace-ig-keep/hop0` cache-flush 0.58 · crash 0.63 · degrade 0.67 · pool-exhaust 0.80; `…/source`
  cache-flush 0.58 · crash 0.52 · degrade 0.64 · pool-exhaust 0.67; the atomic CMI arm `hop0` cache-flush 0.57 · crash 0.57 · degrade 0.63 · pool-exhaust 0.72; the floor `hop0`
  cache-flush 0.59 · crash 0.73 · degrade 0.76 · pool-exhaust 0.85; the earliest control cache-flush 0.42 · crash 0.19 · degrade 0.37 · pool-exhaust 0.25; broadcast RCD
  cache-flush 0.00 · crash 0.38 · degrade 0.76 · pool-exhaust 1.00 (5 / 15 / 20 / 10 cases).
- Per seed (`hop0` hit@1): `rca/trace-ig-keep` xs 0.72 / 0.67 / 0.34 / 0.76 / 0.58, s 0.79 / 0.70 / 0.75 / 0.77 / 0.69, m 0.71 / 0.73 / 0.58 / 0.69 / 0.59;
  `rca/trace-cmi-keep` xs 0.71 / 0.64 / 0.34 / 0.77 / 0.68, s 0.74 / 0.53 / 0.71 / 0.71 / 0.68, m 0.69 / 0.71 / 0.52 / 0.66 / 0.55;
  the earliest control xs 0.13 / 0.06 / 0.28 / 0.39 / 0.17, s 0.10 / 0.14 / 0.20 / 0.12 / 0.10, m 0.49 / 0.42 / 0.40 / 0.48 / 0.36.

### Hypotheses (`plans/rca-arms.md`, addendum "2026-10-31, later" and 2026-10-31; mechanical in `reporttrace` except the context share, read from the score records)

| hypothesis | verdict | xs | s | m |
|---|---|---|---|---|
| **H-trace-earliest** — each TRACE arm's `hop0` hit@1 ≥ `control/trace-earliest`'s | **pass** (15 / 15) | +0.28 … +0.42 | +0.50 … +0.61 | +0.19 … +0.23 |
| **H-ctx-keep-context-share** — `n_rc_unreachable_context / n_rc = 0` for every keep arm; `rca/trace-ig-keep` `hop0` ≥ `rca/trace-ig-overwrite` | **pass** | 0 / 0 / 0 unreachable (keep); overwrite 988 of 3,641; keep − overwrite +0.123 (IG), +0.138 (CMI) | 0 / 0 / 0; CMI overwrite 4,362 of 50,979 (seeds 2, 4 at c = 2); +0.000 (IG), +0.016 (CMI) | 0 / 0 / 0; overwrite 0; 0.000 / 0.000 |
| H-trace-perseq-random — every trace-reading row's `hop1` interval above random | **fail** | pass (all six rows) | fail (the atomic CMI arm 0.213 [0.155, 0.276] against 0.175) | fail (no row: 0.117–0.258 against 0.199, intervals reaching 0.07–0.19) |
| H-trace-perseq-topology — the floor's hit@1 ≥ each TRACE row's at the same reading | **fail** (3 of 30 in the arm's favour) | pass (floor leads by 0.08–0.33) | fail: IG `hop1` +0.013 over the floor (both layouts); `hop0` −0.07 … −0.17 | fail: IG `hop1` +0.067; `hop0` −0.04 … −0.08 |
| H-trace-perseq-broadcast — each TRACE row's `hop1` ≥ every broadcast row's | **fail** | RCD 0.665 above all (0.39–0.51) | RCD 0.820 above all (0.21–0.46) | RCD 0.477 above all (0.12–0.26) |
| H-trace-perseq-self — `hop0` changes the sign of (TRACE − best broadcast) on no rung | **fail** (at m) | sign negative under both readings | negative under both | `hop1` below RCD, `hop0` above (0.61–0.66 against 0.477) |

### Reading notes (descriptive, not re-tuned)

- **Per trace, the family's own rule works where the case level did not.** At the case level the family read `hop0` 0.50 / 0.24 /
  0.22 on xs / s / m (section 2026-11-06 of `rca-latent.md`); per failed request the same arms read `hop0` 0.49–0.63 / 0.63–0.74 /
  0.61–0.66 and `source` 0.53–0.62 / 0.57–0.64 / 0.62–0.63, two to three times random (0.25 / 0.18 / 0.20) and 0.05–0.20 under the
  call-tree floor at `hop0` (0.82 / 0.81 / 0.70). The decay with the rung that the case level showed is absent per trace. The source
  rule names about one and a half services per trace (mean set size 1.4–2.1 against 3.0–4.9 under `hop0`) and gets the exact set right
  on 30–56 % of traces (`hop0`: 5–15 %), at a hit@1 within 0.07 of `hop0` at s and above it at m (0.62–0.63 against 0.61–0.66 is a tie
  at the interval width); it remains the reading the author would sign.
- **The earliest-error control is the weakest row of the axis.** "The first thing that failed" names the root-cause service in 21 %
  / 13 % / 43 % of traces — below random at hit@1 on xs and s (0.25 / 0.18; it ranks every service, so hit@3 is 0.75–0.88) and above it
  only at m, where the `tb-edge` degrade and the other degrade cases make the root span the earliest failed span. The sample graph
  adds 0.2–0.6 over it on every rung — the answer to the author's question is yes, per trace.
- **Score separates per trace: information gain beats the CMI.** On the shared forward of a core layout the two scores differ only in
  the cut, and per trace that cut matters: at s `hop0` 0.741 (IG) against 0.659 (CMI) and `hop1` 0.459 against 0.323; at m 0.659
  against 0.627 and 0.258 against 0.174; at xs equal within 0.002. The CMI cells freeze at much smaller τ (3e-4 … 1e-4 against 0.03
  at s) and admit more edges, which the upstream closure turns into larger reached sets (mean 4.6–4.7 against 4.5 at `hop0`; the
  exact-set rate 0.06 against 0.06–0.10) and lower precision at the top. The case level, which pooled the per-trace credits into one
  ranking per fault, did not show this (identical `hop0` per seed across the four core arms at s).
- **Keep = overwrite where both froze at c = 1; keep wins where the overwrite cell froze at c = 2.** At s and m the keep arms
  reproduce the overwrite arms' rows exactly (the same forward and cut; P1′) except the CMI arm at s (+0.016), whose overwrite cell
  froze at c = 2 on seeds 2 and 4 and left 4,362 root-cause positions (8.6 %) inside the context. At xs the overwrite cells of seeds 0,
  3 and 4 froze at c = 2 and 988 of 3,641 root-cause positions (27 %) are unreachable for them; the keep arms test every position and
  read +0.12 (IG) / +0.14 (CMI) at `hop0`. The context share is the whole difference: this is what H-ctx-keep-context-share asked.
- **The atomic CMI arm is the weakest TRACE row at hop ≥ 1 on s and m** (`hop1` 0.213 / 0.117, at or under random) and within 0.05 of
  the core arms at `hop0` (0.635 / 0.614) and `source` (0.639 / 0.624): its per-position CMI matrix (every position a candidate, one
  forward per position) places the failed position's own service first but orders the upstream services worse. Against the landed
  `rca/trace` (atomic information gain, `shipped` path; c = 3 / 2 / 1 at xs / s / m) it reads `hop0` +0.23 / −0.04 / −0.01.
- **Hop 0 is the primary reading on this axis as registered, and it leads `hop1` by 0.10–0.46 on every arm and rung** (the landed
  arms: +0.02 … +0.48). At m every TRACE row at `hop1` sits at or below random while `hop0` reads 0.61–0.66: the m faults sit in
  the failed service itself (the `tb-edge` degrade has no non-client caller), as the landed section found.
- **Where the family stands against the landed arms per trace:** at `hop0` the core keep arms match or exceed `rca/trace-core`
  (0.615–0.628 against 0.538 at xs; 0.675–0.741 against 0.740 at s; 0.627–0.659 against 0.654 at m) and the whole family exceeds
  `rca/trace` at xs (+0.09 … +0.23) and matches it at s and m (±0.05); at `hop1` the keep arms lead `rca/trace-core` at xs by
  +0.06 … +0.08. Nothing on this axis reaches the floor at `hop0` or RCD's case-level broadcast at s (0.82).

### Disclosures that travel with the rows

- The landed-axis disclosures (device, the training split, the context and truncation counts) hold here; the records carry
  `by_split`. The per-trace rankings (`ranking-trace.json`) and the per-trace score rows (`score-trace-traces.json`) are not tracked
  (sha256 in the `rank-trace` and `score-trace` records; the files live on the run's prefix and in the local `out/`); the 250 uncapped
  per-case work documents (3.9 GB) stay on the prefix.
- The box records name code `3bdba66f` with `dirty: true` (the job's untracked `input/` and `exports/`); the local score records name
  the same commit with `dirty: true` (the landing's untracked result files, committed with them).
- The pre-fault subtraction has no per-trace definition (addendum "2026-10-31, later") and no row here; the family's `…-subpre` rows
  are case level only.
- The ledger gained 90 `rank-trace` entries (75 arms, 15 control); `results/<rung>/latent/seed=<k>/pull-trace-ctx/` records the local
  re-pull of the score-side tier (the corpora had been removed after the s / m landing); the box's pull and export records are
  `pull-rca-ctx-trace/` and `export-ctx-trace/`.

## The l rung, landed 2026-11-09: `rca/trace-ig-keep`, `rca/trace-cmi-keep`, `control/trace-earliest` per failed request, with `rca/rcd`, `rca/baro` and `control/rule` as broadcast sources (l × five seeds; a harness run under a relaxed bin rule)

The l rung of the root-cause work (`plans/rca-arms.md`, addendum 2026-11-08; D-TBR-21 … 23; owner decisions 2026-11-08) is a
**harness result**: the paper's outcome of 2026-10-26 ("no root-cause arm runs at l at this submission") stands as the paper's. The
export uses the relaxed rule (180 s bins, N = 5; three of l's ten fault boundaries floored to their bin, offsets recorded per case:
case-02 end +60 s, case-03 inject +120 s, case-04 inject +60 s on every seed) through the benchmark at the pinned commit; the
case-level broadcast sources are `rca/rcd` (10 seeded runs per case in the pinned environment), `rca/baro` and `control/rule`
only (D-TBR-23: CIRCA, PC-PageRank, BOCPD, TraceRCA and CausalRCA are not run at 152 services; their l rows are absent, not
failed). Five EC2 replicas, one g5.xlarge per seed (`seq2cause-bench-26-11-08-rca-l-s<k>.tfvars`; job scripts
`scripts/jobs/rca-l-s<k>-26-11-08.sh`, PRs #90 … #92; code `1b59e52a`; the chains 21–23 min each after the environment builds;
two earlier applies died in setup and at `rca/baro` — RUN.md run register), the probes on the A10G (`--device cuda`; a disclosed
GPU replica of the frozen cells, D-TBR-15's class; no CPU row). Selection as on the other rungs — rule R, no cap: 38 771 / 39 074 /
42 898 / 41 089 / 35 655 failed POST traces per seed (197 487; seed 0 = the premise count exactly; 0 skipped short); the probes read
the keep cells of the committed l context freezes (`freezes/2026-11-02-l-latent-s<k>-context.json`, `fixed-kl` request cells, every
seed c = 1, N = 8, g = 1; τ 1e-5 for the information gain and 1e-7 for the CMI on seed 0) in one forward per seed (the CMI arm re-ranks
the information-gain arm's work files, `runner_reused`; 482–578 s per probe, 16.8–26.0 M token positions). Staging: every l row on the
m ledger entry of the same arm and stage (form ledger). **Scored traces C = 5 234 over the five seeds (645 / 789 / 1 801 / 851 /
1 148 per seed; m: 16 758): at l the fault takes in few failed requests (the crash in 6–20 and the cache flush in 4–158 traces per corpus), so
the intervals below are wider than m's and the per-kind rows for `crash` (n = 61) and `cache_flush` (n = 220) rest on small sets.**
Scored locally (`scoretrace`, B = 2000, bootstrap seed 0; `--rows all` for the arms, `arm-only` for the control; `score` / `scorercd`
for the three broadcast sources first); `reporttrace --rungs m l` → `tables/rca-trace-l/` (22 rows × 3 tables: m, l, pooled; the m rows
re-pool identically to `tables/rca-trace-context/` on every metric and interval). The landed `tables/rca-trace/`, `tables/rca-trace-context/`
and `tables/rca-latent/` re-run byte-equal (76 / 76, 124 / 124 and 66 / 66 pooled files; the two per-trace summaries gain only the "not run"
keys of the hypotheses the code acquired since). Cell = hit@1 [95 % interval over cases] / hit@3 / exact set (mean set size):

| row | `m` (landed 2026-11-08) | `l` |
|---|---|---|
| `rca/trace-ig-keep/hop1` | 0.258 [0.188, 0.327] / 0.494 / 0.093 (3.05) | 0.149 [0.088, 0.217] / 0.444 / 0.057 (4.08) |
| `rca/trace-ig-keep/hop0` | 0.659 [0.594, 0.709] / 0.904 / 0.149 (3.99) | **0.603 [0.538, 0.663] / 0.824** / 0.054 (4.87) |
| `rca/trace-ig-keep/source` | 0.619 [0.559, 0.674] / 0.850 / 0.483 (1.56) | 0.612 [0.552, 0.671] / 0.711 / 0.526 (1.27) |
| `rca/trace-cmi-keep/hop1` | 0.174 [0.115, 0.235] / 0.430 / 0.054 (3.34) | 0.119 [0.069, 0.174] / 0.368 / 0.045 (3.89) |
| `rca/trace-cmi-keep/hop0` | 0.627 [0.555, 0.687] / 0.856 / 0.110 (4.27) | **0.559 [0.484, 0.626] / 0.774** / 0.049 (4.82) |
| `rca/trace-cmi-keep/source` | 0.631 [0.570, 0.684] / 0.806 / 0.520 (1.43) | 0.594 [0.527, 0.652] / 0.684 / 0.486 (1.32) |
| `control/trace-earliest/earliest` | 0.428 [0.310, 0.557] / 0.881 / 0.008 (5.92) | 0.404 [0.282, 0.544] / 0.815 / 0.000 (6.28) |
| `floor/trace-topology/hop1` | 0.191 [0.092, 0.301] / 0.390 / 0.052 (3.34) | 0.169 [0.068, 0.290] / 0.395 / 0.025 (3.67) |
| `floor/trace-topology/hop0` | 0.695 [0.629, 0.755] / 0.956 / 0.133 (4.46) | 0.693 [0.621, 0.763] / 0.954 / 0.106 (4.86) |
| `control/trace-random` | 0.199 [0.177, 0.225] / 0.569 / n/a | 0.181 [0.168, 0.194] / 0.540 / n/a |
| `control/trace-callees/hop1` | 0.174 [0.120, 0.229] / 0.384 / n/a | 0.151 [0.108, 0.196] / 0.366 / n/a |
| `control/trace-callees/hop0` | 0.351 [0.312, 0.387] / 0.723 / n/a | 0.314 [0.266, 0.362] / 0.681 / n/a |
| `broadcast/control/rule` | 0.399 [0.246, 0.558] / 0.716 / n/a | 0.493 [0.328, 0.666] / 0.688 / n/a |
| `broadcast/rca/baro` | 0.381 [0.233, 0.530] / 0.767 / n/a | 0.436 [0.261, 0.605] / 0.750 / n/a |
| `broadcast/rca/rcd` | 0.477 [0.292, 0.661] / 0.732 / n/a | 0.398 [0.212, 0.587] / 0.718 / n/a |

- Position hit@1 / hit@3 in the arm's own window at l: information gain `hop0` 0.60 / 0.80, `source` 0.59 / 0.71; CMI `hop0`
  0.53 / 0.75, `source` 0.59 / 0.69 (m: 0.66 / 0.87 and 0.63 / 0.82 at `hop0`).
- Per seed at l (`hop0` hit@1 / hit@3): `rca/trace-ig-keep` 0.62 / 0.76, 0.61 / 0.81, 0.56 / 0.82, 0.60 / 0.84, 0.63 / 0.90;
  `rca/trace-cmi-keep` 0.52 / 0.70, 0.62 / 0.82, 0.47 / 0.73, 0.55 / 0.76, 0.63 / 0.86; the earliest control 0.34 / 0.79, 0.46 / 0.85,
  0.44 / 0.80, 0.27 / 0.77, 0.51 / 0.87; broadcast RCD 0.41 / 0.66, 0.44 / 0.85, 0.40 / 0.75, 0.34 / 0.61, 0.40 / 0.72; the floor 0.70 / 0.98,
  0.71 / 0.95, 0.66 / 0.92, 0.75 / 0.95, 0.65 / 0.97.
- Per fault kind at l (hit@1; n = 220 cache-flush, 61 crash, 3 916 degrade, 1 037 pool-exhaust traces): `rca/trace-ig-keep/hop0`
  cache-flush 0.45 · crash 0.59 · degrade 0.62 · pool-exhaust 0.73 (m 0.58 · 0.57 · 0.68 · 0.78); `rca/trace-cmi-keep/hop0` 0.41 · 0.46 · 0.61 ·
  0.70; `…/source` 0.45 · 0.62 · 0.62 · 0.74 (IG), 0.45 · 0.53 · 0.62 · 0.76 (CMI); the floor `hop0` 0.56 · 0.73 · 0.65 · 0.87; the earliest control 0.23 ·
  0.13 · 0.64 · 0.39; random 0.16 · 0.17 · 0.20 · 0.18; broadcast RCD 0.00 · 0.02 · 0.50 · 0.98, BARO 0.62 · 0.11 · 0.24 · 0.96, the rule 0.47 · 0.14 ·
  0.43 · 0.99.
- Case level at l (descriptive; the rows are the broadcast sources, no case-level table or hypothesis is registered at l — owner
  2026-11-08): `rca/rcd` AC@1 0.40 / 0.40 / 0.40 / 0.26 / 0.34 per seed (AC@3 0.36–0.40; the run-averaged aggregation; m pooled 0.40 /
  0.55), `rca/baro` 0.00 / 0.00 / 0.20 / 0.00 / 0.00, `control/rule` 0.20 / 0.20 / 0.20 / 0.20 / 0.00; random AC@1 0.0066 over 152
  candidates. The three case-level rankings are in `results/l/latent/seed=<k>/<slug>/` with their `score-rca.json`.

### Hypotheses (`plans/rca-arms.md`, addendum 2026-11-08; mechanical in `reporttrace.l_hypotheses`, reading `hop0`, pooled over five seeds)

| hypothesis | threshold | `rca/trace-ig-keep` | `rca/trace-cmi-keep` | verdict |
|---|---|---|---|---|
| **H-l-top3** — hit@3 ≥ 0.80 | 0.80 | 0.824 | 0.774 | **fail** (the CMI arm 0.026 under) |
| **H-l-over-free** — hit@1 ≥ best free baseline + 0.15 | best = earliest 0.404 (callees `hop0` 0.314, random 0.181) | +0.198 | +0.154 | **pass** (the CMI arm by 0.004) |
| **H-l-hold** — \|l − m\| ≤ 0.10 on hit@1 and hit@3 | 0.10 | −0.056 / −0.081 | −0.068 / −0.082 | **pass** |
| **H-l-vs-rcd** — hit@1 and hit@3 ≥ broadcast RCD's | RCD 0.398 / 0.718 | +0.205 / +0.106 | +0.161 / +0.056 | **pass** |
| **H-l-ig-ge-cmi** — IG ≥ CMI on both | — | +0.044 / +0.050 | | **pass** |
| **H-l-no-blind-kind** — each kind's hit@1 ≥ 2 × the rung's random hit@1 | 0.363 | min cache-flush 0.446 | min cache-flush 0.408 | **pass** |
| **H-l-floor-gap** — hit@1 ≥ floor `hop0` − 0.15 | floor 0.693 | −0.090 | −0.134 | **pass** |

The landed axis's hypotheses on the l rows read as on m: H-trace-earliest passes (both arms, +0.15 … +0.20), H-trace-perseq-random,
-topology, -broadcast and -self fail (`hop1` at or below random; the floor leads at `hop0`; RCD's broadcast above every `hop1` row;
`hop0` flips the sign against the best broadcast at l as at m).

### Reading notes (descriptive, not re-tuned)

- **Three times the services, no collapse — and a measurable price.** Per failed request the keep arms read `hop0` 0.60 / 0.56 at l
  against 0.66 / 0.63 at m (−0.06 / −0.07 at hit@1, −0.08 at hit@3, inside the registered 0.10), while the call-tree floor is flat (0.69
  on both rungs) and random falls 0.20 → 0.18. The gap to the floor widens from −0.04 / −0.07 (m) to −0.09 / −0.13 (l): the learned
  graph loses a little of what the instrumented call tree keeps. The exact-set rate under `hop0` falls to 0.05 (m 0.11–0.15) with sets
  of 4.8–4.9 services; the `source` rule keeps its shape (1.3 services named, exact on 49–53 %, hit@1 0.59–0.61 ≈ `hop0`'s, hit@3 0.68–0.71
  because it names so few).
- **H-l-top3 fails on the CMI arm only.** The information-gain arm clears the 0.80 bar (0.824); the CMI arm reads 0.774. The
  difference between the two arms is the cut on one shared forward (τ 1e-5 against 1e-7 on seed 0), as at s and m, and the gap is the
  registered direction (H-l-ig-ge-cmi passes: +0.044 / +0.050). Per seed the CMI arm's hit@3 is below 0.80 on three seeds (0.70 / 0.73 /
  0.76) and above on two (0.82 / 0.86); the information-gain arm is below on one (seed 0, 0.76).
- **The free heuristics stay well behind, the margin is thinner at l.** The earliest-error control is the best free baseline at l as
  at m (0.404; random 0.181, callees `hop0` 0.314) and the arms lead it by +0.20 (IG) and +0.15 (CMI) — the CMI arm clears the 0.15
  margin by 0.004, which at C = 5 234 is not a distinction. On `degrade` faults (3 916 of the 5 234 traces) the earliest control reads
  0.64 and matches or beats both arms (0.62 / 0.61): the degraded service's own span is the first non-OK span of the request. The arms
  earn their margin on the other kinds (pool-exhaust 0.73 / 0.70 against 0.39; cache-flush 0.45 / 0.41 against 0.23; crash 0.59 / 0.46
  against 0.13).
- **Against RCD per failed request the arms lead by more at l than at m.** RCD's case ranking applied to the request reads 0.398 /
  0.718 (m 0.477 / 0.732); the arms +0.21 / +0.16 at hit@1. RCD's broadcast is a kind-level pattern: pool-exhaust 0.98 and degrade 0.50,
  cache-flush 0.00 and crash 0.02 — the 180 s export gives its post-fault sample 5–14 bins, and on the two kinds that take in few
  requests it names other services. BARO's broadcast (0.436) is the only row that reads the cache flush (0.62) and the rule's
  broadcast (0.493) is the strongest case-level row at l at hit@1 — both still 0.07–0.17 under the arms. The case-level RCD numbers
  themselves (AC@1 0.26–0.40 per seed) are recorded for the broadcast, not read as a case-level result (owner, 2026-11-08).
- **No blind fault kind, with a caveat on the sample.** Every (arm, kind) hit@1 is above twice the rung's random (0.363): the weakest is
  the CMI arm on the cache flush (0.408, n = 220) and the information-gain arm on the same kind (0.446). The crash rows rest on 61
  traces across the five seeds (the 900 s crash takes in 6–20 failed requests per corpus at l's error multiplier) — the per-kind
  intervals are not computed and these two rows would not survive a stricter bound.
- **Hop 1 is at random at l as at m** (0.149 / 0.119 against 0.181; the floor's `hop1` 0.169): the l faults sit in the failed service
  itself or one hop upstream of the failed position, and `hop0` remains the primary reading as registered.

### Disclosures that travel with the rows

- **The l run is a harness result under a disclosed departure from §3's bin rule (D-TBR-21):** 180 s bins, N = 5, three boundaries
  floored to their bin (offsets recorded per case in the export record); the exporter is the pinned benchmark commit (D-TBR-22). The
  paper's outcome of 2026-10-26 stands; nothing here enters the frozen paper.
- **The broadcast set at l is three arms (D-TBR-23):** the other six case-level arms are not run at 152 services; their rows are absent.
  **No case-level table and no case-level hypothesis at l** (owner, 2026-11-08); the three case-level `score-rca.json` are landed as the
  broadcast sources' records.
- **GPU replica:** the probes ran on an A10G (`--device cuda`); no CPU row of these cells exists, so D-TBR-15's tolerance check is not
  applicable (the per-trace seed stream of 2026-10-30; CPU and CUDA draw differently under one seed). The runner names seq2cause 0.3.1
  and harness commit `1b59e52a`; the backbone sha of every probe equals its freeze's.
- **Small scored sets:** C = 645 … 1 801 per seed (5 234 pooled) against 16 758 at m; the intervals are over the 25 cases; two fault
  kinds rest on 61 and 220 traces. Rule R itself selects 197 487 traces; the 192 253 in which the fault did not take are not scored
  (`n_took_4xx_only` 26).
- **RCD's post-fault sample under the floored boundaries is 5–14 bins per case** (72–91 pre-fault bins); 50 seeded runs per seed, 0
  failed; the run-averaged aggregation of D-TBR-7; the rankings reproduce byte-equal across two boxes (the aborted second apply and the
  landed third).
- The box records name code `1b59e52a` with `dirty: true` (the job's untracked `input/` and `exports/`); the local score records name
  the same commit with `dirty: true` (the landing's untracked result files, committed with them); `rca/baro` ran in the method
  environment built on the box by `scripts/envs/rca-env.sh` (Python 3.12), the rest of the chain in the provisioning environment
  (Python 3.11) — RUN.md "Environment".
- The per-trace rankings (`ranking-trace.json`, 12–16 MB per arm and seed) and the per-trace score rows (`score-trace-traces.json`)
  are not tracked (sha256 in the records; the files live on the run's prefixes and in the local `out/`); the 62 RCD work documents per
  seed stay on the prefix. The ledger gained 15 `rank` and 15 `rank-trace` entries; `results/l/latent/seed=<k>/pull-trace-l/` records
  the local pull of the score-side tier, `pull-rca-l/` and `export-l/` the box's.

## The xl rung, landed 2026-11-11: the l runs one rung up (xl × five seeds; a harness run at 180 s bins with N = 2; rows reported, no hypothesis)

The xl rung of the root-cause work (`plans/rca-arms.md`, addendum 2026-11-10; D-TBR-24 … 26; owner decisions 2026-11-10) repeats the
l runs on the 351-service corpora: `rca/rcd`, `rca/baro` and `control/rule` at case level as broadcast sources, `rca/trace-ig-keep`,
`rca/trace-cmi-keep` and `control/trace-earliest` per failed request. **No hypothesis is registered at xl**: the rows stand beside m
and l, and `reporttrace` reads no xl row into any verdict (the l hypotheses re-read identically with xl named). Export: 180 s bins,
the widest that keep five bins in the 900 s crash, with **N = 2 as measured** (the sparsest candidate service's pre-fault median over
the five seeds; p10 0; l had 5 and 2) and the same three boundaries floored as at l (case-02 end +60 s, case-03 inject +120 s,
case-04 inject +60 s on every seed), through the benchmark at `d86d822a`. Five EC2 replicas, one g5.2xlarge per seed
(`seq2cause-bench-26-11-10-rca-xl-s<k>.tfvars`; `scripts/jobs/rca-xl-s<k>-26-11-10.sh`; code `6b14c77c`; one apply, the chains 35–36
min each after the environment builds), the probes on the A10G (`--device cuda`). Selection rule R, no cap: 40 882 / 38 723 / 39 526 /
38 909 / 36 984 failed POST traces per seed (195 024; seed 0 = the premise count; 0 skipped short); the probes read the keep cells of
`freezes/2026-11-03-xl-latent-s<k>-context.json` (`fixed-kl` request, c = 1, N = 8, g = 1) in one forward per seed (489–539 s,
19.6–22.0 M token positions). Staging: every xl row on the l ledger entry of the same arm and stage (form ledger). **Scored traces
C = 2 560 over the five seeds (388 / 730 / 453 / 693 / 296; l 5 234, m 16 758).** Scored locally (`scoretrace`, B = 2000, bootstrap
seed 0); `reporttrace --rungs m l xl` → `tables/rca-trace-xl/`; the m and l rows re-pool identically to `tables/rca-trace-l/` (0 metric
or interval differences) and the hypothesis block equals the landed one. Cell = hit@1 [95 % interval over cases] / hit@3 / exact set
(mean set size):

| row | `m` (landed 2026-11-08) | `l` (landed 2026-11-09) | `xl` |
|---|---|---|---|
| `rca/trace-ig-keep/hop1` | 0.258 [0.188, 0.327] / 0.494 / 0.093 (3.05) | 0.149 [0.088, 0.217] / 0.444 / 0.057 (4.08) | 0.137 [0.081, 0.192] / 0.429 / 0.047 (3.32) |
| `rca/trace-ig-keep/hop0` | 0.659 [0.594, 0.709] / 0.904 / 0.149 (3.99) | 0.603 [0.538, 0.663] / 0.824 / 0.054 (4.87) | 0.620 [0.524, 0.705] / 0.813 / 0.054 (4.12) |
| `rca/trace-ig-keep/source` | 0.619 [0.559, 0.674] / 0.850 / 0.483 (1.56) | 0.612 [0.552, 0.671] / 0.711 / 0.526 (1.27) | 0.603 [0.509, 0.686] / 0.640 / 0.562 (1.11) |
| `rca/trace-cmi-keep/hop1` | 0.174 [0.115, 0.235] / 0.430 / 0.054 (3.34) | 0.119 [0.069, 0.174] / 0.368 / 0.045 (3.89) | 0.137 [0.081, 0.192] / 0.429 / 0.047 (3.32) |
| `rca/trace-cmi-keep/hop0` | 0.627 [0.555, 0.687] / 0.856 / 0.110 (4.27) | 0.559 [0.484, 0.626] / 0.774 / 0.049 (4.82) | 0.620 [0.524, 0.705] / 0.813 / 0.054 (4.12) |
| `rca/trace-cmi-keep/source` | 0.631 [0.570, 0.684] / 0.806 / 0.520 (1.43) | 0.594 [0.527, 0.652] / 0.684 / 0.486 (1.32) | 0.603 [0.509, 0.687] / 0.640 / 0.562 (1.10) |
| `control/trace-earliest/earliest` | 0.428 [0.310, 0.557] / 0.881 / 0.008 (5.92) | 0.404 [0.282, 0.544] / 0.815 / 0.000 (6.28) | 0.420 [0.290, 0.553] / 0.913 / 0.000 (5.92) |
| `floor/trace-topology/hop1` | 0.191 [0.092, 0.301] / 0.390 / 0.052 (3.34) | 0.169 [0.068, 0.290] / 0.395 / 0.025 (3.67) | 0.155 [0.070, 0.240] / 0.461 / 0.073 (3.02) |
| `floor/trace-topology/hop0` | 0.695 [0.629, 0.755] / 0.956 / 0.133 (4.46) | 0.693 [0.621, 0.763] / 0.954 / 0.106 (4.86) | 0.700 [0.620, 0.775] / 0.975 / 0.118 (4.20) |
| `control/trace-random` | 0.199 [0.177, 0.225] / 0.569 / n/a | 0.181 [0.168, 0.194] / 0.540 / n/a | 0.194 [0.176, 0.215] / 0.570 / n/a |
| `control/trace-callees/hop1` | 0.174 [0.120, 0.229] / 0.384 / n/a | 0.151 [0.108, 0.196] / 0.366 / n/a | 0.209 [0.138, 0.287] / 0.421 / n/a |
| `control/trace-callees/hop0` | 0.351 [0.312, 0.387] / 0.723 / n/a | 0.314 [0.266, 0.362] / 0.681 / n/a | 0.352 [0.305, 0.408] / 0.750 / n/a |
| `broadcast/control/rule` | 0.399 [0.246, 0.558] / 0.716 / n/a | 0.493 [0.328, 0.666] / 0.688 / n/a | 0.440 [0.259, 0.621] / 0.710 / n/a |
| `broadcast/rca/baro` | 0.381 [0.233, 0.530] / 0.767 / n/a | 0.436 [0.261, 0.605] / 0.750 / n/a | 0.454 [0.308, 0.600] / 0.793 / n/a |
| `broadcast/rca/rcd` | 0.477 [0.292, 0.661] / 0.732 / n/a | 0.398 [0.212, 0.587] / 0.718 / n/a | 0.417 [0.249, 0.590] / 0.723 / n/a |

- **At xl the two keep arms are one row.** The CMI cells' frozen τ is 0 on every xl seed and the information-gain cells' is 1e-8 …
  1e-6, so both cuts keep nearly every positive entry of the one shared forward: the two arms' per-trace entries are identical on
  99.3–99.97 % of the selected traces per seed, and their pooled `hop0` rows agree to three decimals. The xl table has one learned
  reading, not two; the information-gain / CMI comparison of s, m and l is not available here.
- Position hit@1 / hit@3 at xl: `hop0` 0.61 / 0.77, `source` 0.58 / 0.64 (l: 0.60 / 0.80 and 0.59 / 0.71 for the information gain).
- Per seed at xl (`hop0` hit@1 / hit@3, either arm): 0.48 / 0.70, 0.76 / 0.94, 0.54 / 0.71, 0.66 / 0.90, 0.68 / 0.84; the earliest
  control 0.32 / 0.93, 0.36 / 0.91, 0.56 / 0.90, 0.43 / 0.91, 0.43 / 0.92; broadcast RCD 0.38 / 0.65, 0.39 / 0.70, 0.47 / 0.64, 0.51 /
  0.97, 0.36 / 0.70; the floor 0.71 / 0.97, 0.80 / 0.98, 0.68 / 0.97, 0.73 / 0.99, 0.59 / 0.97.
- Per fault kind at xl (hit@1; n = 212 cache-flush, 17 crash, 1 277 degrade, 1 054 pool-exhaust traces): the arms' `hop0` cache-flush
  0.58 · crash 0.54 · degrade 0.59 · pool-exhaust 0.78; `source` 0.56 · 0.52 · 0.58 · 0.76; the floor `hop0` 0.60 · 0.85 · 0.61 · 0.85; the
  earliest control 0.34 · 0.17 · 0.60 · 0.34; random 0.17 · 0.16 · 0.22 · 0.19; broadcast RCD 0.00 · 0.00 · 0.59 · 0.82, BARO 0.51 · 0.33 ·
  0.29 · 0.83, the rule 0.27 · 0.25 · 0.35 · 0.95.
- Case level at xl (descriptive; the rows are the broadcast sources): `rca/rcd` AC@1 0.32 / 0.30 / 0.28 / 0.24 / 0.34 per seed,
  `rca/baro` 0.00 on every seed, `control/rule` 0.20 / 0.20 / 0.00 / 0.20 / 0.00; random AC@1 0.0028 over 352 candidates.

### Reading notes (descriptive, not re-tuned)

- **Per failed request the reading holds from l to xl.** `hop0` reads 0.620 / 0.813 at xl against 0.603 / 0.824 (information gain)
  and 0.559 / 0.774 (CMI) at l and 0.659 / 0.904 at m; the xl interval [0.524, 0.705] is wider than l's and covers both. The call-tree
  floor is flat across the three rungs (0.695 / 0.693 / 0.700 at hit@1), so the gap to it at xl is −0.08, inside l's −0.09 … −0.13.
  Random is 0.19. The `source` rule names 1.1 services, is exact on 56 % of traces and reads 0.603 at hit@1.
- **The free heuristic is closer at hit@3 than at hit@1.** The earliest-error control reads 0.420 / 0.913: 0.20 under the arms at
  hit@1, 0.10 above them at hit@3 (it ranks every service of the trace, 5.9 on average, where `hop0` names 4.1).
- **Against the case-level rankings applied to the request** the arms lead by +0.17 … +0.20 at hit@1 (BARO 0.454, the rule 0.440,
  RCD 0.417) and by +0.02 … +0.10 at hit@3; the three intervals are wide (0.25–0.62) and overlap the arms'. RCD ranks one to ten of
  the 352 services per run, and BARO's case-level AC@1 is 0 on every seed: at N = 2 the 180 s export gives the sparsest services
  near-empty bins, so these three rows are a weaker comparison at xl than at l.
- **Fault kinds rest on small sets.** The crash rows are 17 traces over the five seeds (one on seed 0) and are not a reading; the
  cache flush has 212. Degrade and pool-exhaust carry 91 % of the scored traces.
- **Seeds differ more than at l**: `hop0` hit@1 runs 0.48–0.76 across seeds on 296–730 scored traces each.

### Disclosures that travel with the rows

- **The bin rule is relaxed further than at l (D-TBR-24):** 180 s, N = 2 (p10 0 on two seeds; 5–34 of 351 candidate services per seed
  have a pre-fault median under 5 requests per bin), three boundaries floored; the width was fixed before the measurement and N
  recorded as found. It affects the case-level broadcast sources only; the per-trace arms and controls read the fault windows, not bins.
- **No hypothesis at xl (D-TBR-26; owner, 2026-11-10)**; the broadcast set is the l set; the other case-level arms are not run.
- **One learned reading:** the two keep arms coincide at xl (above).
- **GPU replica:** the probes ran on an A10G; no CPU row of these cells exists. The runner names seq2cause 0.3.1 and harness commit
  `6b14c77c`; the backbone sha of every probe equals its freeze's.
- **Small scored sets:** C = 296 … 730 per seed (2 560 pooled); rule R selects 195 024 traces and the 192 464 in which the fault did not
  take are not scored (`n_took_4xx_only` 3). Intervals are over the 25 cases.
- The pull checks 595 files on seeds 0, 2, 3 and 594 on seeds 1, 4 (the published end-request view has one part file fewer there);
  each pull passes against its own manifest.
- The per-trace rankings (`ranking-trace.json`) and per-trace score rows (`score-trace-traces.json`) are not tracked (sha256 in the
  records); the 62 RCD work documents per seed stay on the prefix. The ledger gained 15 `rank` and 15 `rank-trace` entries (1 899).
