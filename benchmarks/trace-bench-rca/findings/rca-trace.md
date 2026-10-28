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

*Pending the replica.* Gate: mean per-trace edge-set Jaccard ≥ 0.80 per run, positions identical; the per-case top-1
service of the GPU re-probe against the landed one is reported beside it.

## The axis (pooled over five seeds; cases = injected faults; the point is the mean over cases)

*Pending the runs; the table is `tables/rca-trace/rca-trace.md`.* Label-side reads of the model-free rows at seed 0
(c = 1, no skipped traces; a scale check, not a landed cell): the call-tree floor's hit@1 0.704 / 0.416 / 0.180 at
hop ≥ 1 and 0.702 at m with hop 0 admitted; random over the trace's services 0.243 / 0.206 / 0.200.

## Hypotheses

*Pending.* H-trace-perseq-random · H-trace-perseq-topology · H-trace-perseq-broadcast · H-trace-perseq-self.

## Disclosures that travel with the rows

- Device: the probes ran on a GPU (`cuda`); CPU and CUDA draw different noise under the same seed, so this read is a
  disclosed replica of the frozen cells, not a bit-identical reproduction of the landed CPU case-level rows (which
  stay as landed). The per-trace seed index is the trace's position in the selected list, so the uncapped selection
  gives a trace a different draw than the capped one.
- The backbone saw the train split of the same corpus; ~79 % of the scored traces are train-split traces at xs seed 0;
  the per-split hit rates are in the score records (`by_split`).
- A root-cause position inside the arm's context (the first c − 1 spans; c = 1–3) or beyond the 64-token cap is
  unreachable for the arm and counted (`n_rc_unreachable_context`, `n_rc_unreachable_truncation`); the floor and the
  controls are defined on the whole trace. At xs seed 0 with c = 3, 159 of 1,745 probed traces hold no failed
  non-client position inside the window at all.
- The per-trace rankings and per-trace score rows are not tracked (up to ≈ 60 MB per m run); their sha256 ride the
  `rank-trace` and `score-trace` records and the files live on the run's prefix and in the local scratch.
