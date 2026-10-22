# Root-cause arms on the latent corpora — BARO rows (xs, s, m × five seeds)

Landed 2026-10-21. Registered by `plans/rca-arms.md` (2026-10-14, with the addenda of 2026-10-19
and 2026-10-21) and `plans/controls.md`. Every read below ran from commit `c0513701` in the
`trace-bench-rca` environment (Python 3.12, `fse-baro` at `e35f4ec`, `RCAEval` 1.8.0 at
`259ea41`, the benchmark at `ace384b4`), corpora `chadyuk/trace-bench@v0.3.0`, latent variant;
the export at the registered widths (xs 30 s, s 15 s, m 60 s); scoring by the benchmark's
`tracebench.score_rca` (B = 2000, bootstrap seed 0). Tables: `tables/rca-latent/`
(`summary.json`, `rca-latent.md`, one pooled file per rung and arm, one across rungs). No value
was chosen on any read: both rows run at the methods' published defaults.

## Ranking axis (pooled over five seeds; cases = injected faults)

| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 (n services) | misses (no ranking) |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/baro` | 10 | 0.300 | 0.800 | 1.000 | 0.720 [0.540, 0.880] | 0.600 (5) | 0 |
| s | `rca/baro` | 15 | 0.267 | 0.467 | 0.667 | 0.467 [0.253, 0.680] | 0.250 (12) | 0 |
| m | `rca/baro` | 25 | 0.120 | 0.160 | 0.240 | 0.176 [0.048, 0.320] | 0.058 (52) | 0 |
| xs | `rca/baro-bocpd` | 10 | 0.100 | 0.100 | 0.200 | 0.140 [0.000, 0.380] | 0.600 (5) | 7 |
| s | `rca/baro-bocpd` | 15 | 0.133 | 0.267 | 0.600 | 0.333 [0.173, 0.533] | 0.250 (12) | 0 |
| m | `rca/baro-bocpd` | 25 | 0.040 | 0.040 | 0.040 | 0.040 [0.000, 0.120] | 0.058 (52) | 8 |

Across rungs (50 cases): `rca/baro` Avg@5 0.372 [0.260, 0.488]; `rca/baro-bocpd` 0.148 [0.072, 0.232].
`control/random` is the analytic column (AC@k = min(k, n) / n); `control/rule` is not run
(`plans/controls.md` §2 awaits the owner, publication plan Q8).

Per fault kind, `rca/baro` Avg@5: xs crash 0.52 / degrade 0.92; s crash 0.16 / degrade 0.36 /
pool exhaustion 0.88; m cache flush 0.04 / crash 0.00 / degrade 0.00 (10 cases) / pool exhaustion
0.84. The full pipeline: xs crash 0.00 / degrade 0.28; s 0.28 / 0.24 / 0.48; m 0.00 / 0.00 / 0.00 /
0.20.

## The full pipeline's no-ranking cases and change points

- **Misses without a ranking (registered, counted, scored as misses):** xs 7 of 10 "change point
  at the table's end" (the detector's only change point falls after the last row); m 7 of 25
  "detector raised LinAlgError" (scipy's default `eigh` driver on the multivariate-t shape matrix,
  addendum 2026-10-21) and 1 of 25 at the table's end; s none. One further xs case returned an empty
  ranking from the pinned scorer (a one-row fault segment, BARO's per-segment constant-column drop),
  scored as a miss by the benchmark's scorer.
- **Change-point offset from the labelled injection** (cases whose first change point lies inside
  the table): xs 3 cases, +270 …
  +450 s (median +300, all after the injection); **s 15 of 15 before the injection** (−18,405 …
  −10,155 s, median −14,805); **m 17 of 17 before the injection** (−9,900 … −660 s, median −3,420).
  At s and m the detector places its first change point hours before the fault, inside the
  pre-fault segment, so RobustScorer fits its baseline on a prefix that ends well before the
  injection.

## Hypotheses (`plans/rca-arms.md` §6)

| hypothesis | verdict | the numbers |
|---|---|---|
| **H-rca-random** — `rca/baro`'s Avg@5 interval above the random expectation on every rung | **fail** (xs, m) | xs low 0.540 vs 0.600; s low 0.253 vs 0.250 (pass); m low 0.048 vs 0.058 |
| **H-rca-rule** — `rca/baro` at or above `control/rule` | **not run** | `control/rule` awaits Q8 |
| **H-rca-bocpd** — the full pipeline within 0.10 of the injection-time row on every rung | **fail** (every rung) | gaps −0.580 (xs), −0.133 (s), −0.136 (m) |
| **H-rca-scale** — `rca/baro`'s Avg@5 does not rise from xs to m | **pass** | m − xs = −0.544; wider half-width 0.170 |
| **H-graph-topology** — CausalRCA's service graph at or below `floor/topology` | **not run** | stream B5 |

H-rca-random fails on intervals, not on point estimates: every `rca/baro` point estimate is above
random (0.72 vs 0.60, 0.47 vs 0.25, 0.18 vs 0.06), but xs has 10 cases over 5 services and m's
interval reaches the expectation. At m the row finds pool exhaustion (0.84) and almost nothing
else (crash 0.00, degrade 0.00, cache flush 0.04 over 20 cases).

## Reading notes (descriptive, not re-tuned)

- The registered score is the **signed** maximum of the robust z-score after the split (§2). A
  fault that lowers a series (a crash's request count) scores only through the series it raises.
  Why the m crash and degrade cases score zero is not analysed here; the rule is the pinned code's
  and is not changed.
- The full pipeline is the published BARO algorithm with no knowledge of the injection; on these
  corpora its detector fires either after the fault's last bin (xs) or hours before the fault (s,
  m), which the offsets above make visible. The injection-time row is the suite's "BARO" (the
  author's confirmation of 2026-10-14).
- One series per service and metric is all either row reads: request count, 5xx share, p50, p95
  from the raw access-log lines; no CPU, memory or trace features exist in this export.

## Outreach record

- BARO: test plan sent to Luan Pham 2026-10-14; reply the same day confirming both rows and the
  naming of the injection-time configuration as "BARO", assistance offered, no objection to the
  adapter.
- CausalRCA: test plan sent to Peng Chen and Zhiming Zhao 2026-10-14; no reply as of 2026-10-21
  (its row is stream B5).

## Deviations in force

D-TBR-1 (benchmark pin `ace384b4`), D-TBR-2 (the suite's `main.py` loader not run; its fill rule
applied), D-TBR-3 (positional case directories) — `RUN.md`. Addendum 2026-10-21: Python 3.12; the
end-of-table and LinAlgError misses; the scorer's own preprocessing on the full-pipeline row.
