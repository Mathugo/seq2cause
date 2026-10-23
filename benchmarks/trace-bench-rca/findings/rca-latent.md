# Root-cause arms on the latent corpora — BARO, `control/rule` and CausalRCA rows (xs, s, m × five seeds)

Landed 2026-10-21. Registered by `plans/rca-arms.md` (2026-10-14, with the addenda of 2026-10-19
and 2026-10-21) and `plans/controls.md`. Every read below ran from commit `c0513701` in the
`trace-bench-rca` environment (Python 3.12, `fse-baro` at `e35f4ec`, `RCAEval` 1.8.0 at
`259ea41`, the benchmark at `ace384b4`), corpora `chadyuk/trace-bench@v0.3.0`, latent variant;
the export at the registered widths (xs 30 s, s 15 s, m 60 s); scoring by the benchmark's
`tracebench.score_rca` (B = 2000, bootstrap seed 0). Tables: `tables/rca-latent/`
(`summary.json`, `rca-latent.md`, one pooled file per rung and arm, one across rungs). No value
was chosen on any read: both rows run at the methods' published defaults.

The `control/rule` row landed 2026-10-23 from commit `c8c4551c`, on the same exports (every case
file re-verified against its export record), registered by `plans/controls.md` §2 with its
addenda of 2026-10-22 (owner confirmation) and 2026-10-23 (the details §2 leaves open, fixed
before the code). The `rca/baro` and `rca/baro-bocpd` pooled files of the regenerated table are
byte-identical to the 2026-10-21 ones.

## Ranking axis (pooled over five seeds; cases = injected faults)

| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 (n services) | misses (no ranking) |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/baro` | 10 | 0.300 | 0.800 | 1.000 | 0.720 [0.540, 0.880] | 0.600 (5) | 0 |
| s | `rca/baro` | 15 | 0.267 | 0.467 | 0.667 | 0.467 [0.253, 0.680] | 0.250 (12) | 0 |
| m | `rca/baro` | 25 | 0.120 | 0.160 | 0.240 | 0.176 [0.048, 0.320] | 0.058 (52) | 0 |
| xs | `rca/baro-bocpd` | 10 | 0.100 | 0.100 | 0.200 | 0.140 [0.000, 0.380] | 0.600 (5) | 7 |
| s | `rca/baro-bocpd` | 15 | 0.133 | 0.267 | 0.600 | 0.333 [0.173, 0.533] | 0.250 (12) | 0 |
| m | `rca/baro-bocpd` | 25 | 0.040 | 0.040 | 0.040 | 0.040 [0.000, 0.120] | 0.058 (52) | 8 |
| xs | `control/rule` | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.820] | 0.600 (5) | 0 |
| s | `control/rule` | 15 | 0.467 | 0.667 | 0.800 | 0.680 [0.480, 0.867] | 0.250 (12) | 0 |
| m | `control/rule` | 25 | 0.160 | 0.280 | 0.320 | 0.272 [0.120, 0.440] | 0.058 (52) | 0 |

Across rungs (50 cases): `rca/baro` Avg@5 0.372 [0.260, 0.488]; `rca/baro-bocpd` 0.148 [0.072, 0.232];
`control/rule` 0.468 [0.344, 0.588]. `control/random` is the analytic column (AC@k = min(k, n) / n);
`control/rule` is the three-rule alert counter (`plans/controls.md` §2). Alerting services per
case, mean [min, max]: xs 2.5 [0, 5] of 5, s 8.8 [3, 12] of 12, m 29.0 [3, 51] of 52.

Per fault kind, `rca/baro` Avg@5: xs crash 0.52 / degrade 0.92; s crash 0.16 / degrade 0.36 /
pool exhaustion 0.88; m cache flush 0.04 / crash 0.00 / degrade 0.00 (10 cases) / pool exhaustion
0.84. The full pipeline: xs crash 0.00 / degrade 0.28; s 0.28 / 0.24 / 0.48; m 0.00 / 0.00 / 0.00 /
0.20. `control/rule`: xs crash 0.52 / degrade 0.76; s crash 0.32 / degrade 0.72 / pool exhaustion
1.00; m cache flush 0.00 / crash 0.08 / degrade 0.18 / pool exhaustion 0.92.

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
| **H-rca-rule** — `rca/baro` at or above `control/rule` | **fail** (s, m) | `rca/baro` − `control/rule` Avg@5: xs +0.080 (pass), s −0.213, m −0.096 |
| **H-rca-bocpd** — the full pipeline within 0.10 of the injection-time row on every rung | **fail** (every rung) | gaps −0.580 (xs), −0.133 (s), −0.136 (m) |
| **H-rca-scale** — `rca/baro`'s Avg@5 does not rise from xs to m | **pass** | m − xs = −0.544; wider half-width 0.170 |
| **H-graph-topology** — CausalRCA's service graph at or below `floor/topology` | **pass** (every rung, both series; four test seeds 1–4) | CausalRCA − topology directed F1: xs −0.73 / −0.86, s −0.82 / −0.77, m −0.74 / −0.77 (`latency-50` / `latency-95`) |

H-rca-random fails on intervals, not on point estimates: every `rca/baro` point estimate is above
random (0.72 vs 0.60, 0.47 vs 0.25, 0.18 vs 0.06), but xs has 10 cases over 5 services and m's
interval reaches the expectation. At m the row finds pool exhaustion (0.84) and almost nothing
else (crash 0.00, degrade 0.00, cache flush 0.04 over 20 cases).

H-rca-rule fails on the registered point estimates: the three-rule alert counter is above
`rca/baro` at s (0.680 vs 0.467) and m (0.272 vs 0.176) and below it at xs (0.640 vs 0.720).
Every pair of intervals overlaps, so no rung separates the two rows; across rungs the rule control
is at 0.468 [0.344, 0.588] against BARO's 0.372 [0.260, 0.488]. By fault kind the rule control is
at or above `rca/baro` on every s and m kind but the m cache flush (0.00 vs 0.04).

## Reading notes (descriptive, not re-tuned)

- The registered score is the **signed** maximum of the robust z-score after the split (§2). A
  fault that lowers a series (a crash's request count) scores only through the series it raises.
  Why the m crash and degrade cases score zero is not analysed here; the rule is the pinned code's
  and is not changed.
- The full pipeline is the published BARO algorithm with no knowledge of the injection; on these
  corpora its detector fires either after the fault's last bin (xs) or hours before the fault (s,
  m), which the offsets above make visible. The injection-time row is the suite's "BARO" (the
  author's confirmation of 2026-10-14).
- The rule control alerts in either direction (a crash's falling request count alerts), where
  `rca/baro`'s registered score is signed (the first note above). The m crash and degrade cases
  are where BARO scores zero and the rule control does not (0.08, 0.18). Neither difference is
  analysed further here.
- One series per service and metric is all either row reads: request count, 5xx share, p50, p95
  from the raw access-log lines; no CPU, memory or trace features exist in this export.

## Outreach record

- BARO: test plan sent to Luan Pham 2026-10-14; reply the same day confirming both rows and the
  naming of the injection-time configuration as "BARO", assistance offered, no objection to the
  adapter.
- CausalRCA: test plan sent to Peng Chen and Zhiming Zhao 2026-10-14; no reply as of 2026-10-23
  (its row is stream B5).

## Deviations in force

D-TBR-1 (benchmark pin `ace384b4`), D-TBR-2 (the suite's `main.py` loader not run; its fill rule
applied), D-TBR-3 (positional case directories) — `RUN.md`. Addendum 2026-10-21: Python 3.12; the
end-of-table and LinAlgError misses; the scorer's own preprocessing on the full-pipeline row.

## CausalRCA (`rca/causalrca`, both series; landed 2026-10-23)

Registered by `plans/rca-arms.md` (the 2026-10-22 decisions Q7 / Q16 / Q17, D-TBR-5 and the adapter
addendum of 2026-10-23). The authors' `train_latency.py` at `21f4eb0` with its data block substituted
(the case's fault-window `latency-50` or `latency-95` columns in `services.json` order, min-max scaled
per column; every other line unchanged; `CONFIG.epochs = 1000`, D-TBR-4), in the pinned environment
`causalrca-21f4eb0` (the repository's requirements exactly, Python 3.9, x86_64 under Rosetta 2), ten
trainings per case and series seeded with the run index, from commit `88b3b325`, locally on CPU (eight
single-threaded workers): 1,000 trainings, 18.2 CPU-hours (xs 1.6, s 3.7, m 12.9; the longest m training
311 s). Every rung ran locally: the plan's GPU box for m (Q7) was not needed — no case approached the
~30-minute threshold of §4.

**Failed runs: 6 of 1,000** (xs 2, m 4), all one cause: the learned graph was empty after the script's
own `|w| < 0.3` cut, and `sknetwork`'s PageRank raised "The input matrix is empty". Each is a run without
a hit on the ranking axis and is left out of its case's mean graph, as registered; none was refilled.

### Ranking axis (pooled over five seeds; per case, the share of the ten runs placing the root cause in the first k)

| rung | series | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 |
|---|---|---|---|---|---|---|---|
| xs | `latency-50` | 10 | 0.220 | 0.620 | 0.990 | 0.596 [0.470, 0.734] | 0.600 |
| s | `latency-50` | 15 | 0.080 | 0.227 | 0.487 | 0.248 [0.113, 0.400] | 0.250 |
| m | `latency-50` | 25 | 0.016 | 0.036 | 0.068 | 0.040 [0.007, 0.084] | 0.058 |
| xs | `latency-95` | 10 | 0.280 | 0.630 | 0.990 | 0.622 [0.474, 0.780] | 0.600 |
| s | `latency-95` | 15 | 0.107 | 0.193 | 0.333 | 0.204 [0.083, 0.356] | 0.250 |
| m | `latency-95` | 25 | 0.012 | 0.036 | 0.060 | 0.034 [0.003, 0.080] | 0.058 |

Across rungs (50 cases): `latency-50` 0.214 [0.138, 0.296], `latency-95` 0.203 [0.128, 0.281]. Every
point estimate is at or below the analytic random expectation of its rung, and every interval contains
it. Per fault kind (Avg@5, `latency-50` / `latency-95`): xs crash 0.62 / 0.49, degrade 0.58 / 0.75; s
crash 0.32 / 0.37, degrade 0.19 / 0.21, pool exhaustion 0.23 / 0.04; m cache flush 0.14 / 0.01, crash
0.02 / 0.02, degrade 0.02 / 0.07, pool exhaustion 0.01 / 0.00 — pool exhaustion, which `rca/baro` and
`control/rule` find at 0.84–1.00, is near zero here at m.

### Graph axis (four test seeds 1–4; seed 0 is the graph axis's validation split)

τ frozen per rung and series on the seed-0 cases (`freezes/2026-10-22-<rung>-rca-causalrca-<series>-graph.json`):
xs 0.1 / 0.15, s and m 0.03 / 0.03 — at s and m every edge learned in any run (F1 at 0.03 equals F1 at 0).
Mean over the test cases of the per-case scores against the service view at its default floor, with
`floor/topology` on the same cases:

| rung | series | τ | cases | precision | recall | F1 | AUROC | `floor/topology` F1 / AUROC |
|---|---|---|---|---|---|---|---|---|
| xs | `latency-50` | 0.1 | 8 | 0.249 | 0.304 | 0.266 | 0.516 | 1.000 / 1.000 |
| s | `latency-50` | 0.03 | 12 | 0.258 | 0.153 | 0.181 | 0.500 | 1.000 / 1.000 |
| m | `latency-50` | 0.03 | 20 | 0.185 | 0.431 | 0.257 | 0.507 | 0.997 / 0.999 |
| xs | `latency-95` | 0.15 | 8 | 0.132 | 0.148 | 0.136 | 0.433 | 1.000 / 1.000 |
| s | `latency-95` | 0.03 | 12 | 0.319 | 0.198 | 0.233 | 0.517 | 1.000 / 1.000 |
| m | `latency-95` | 0.03 | 20 | 0.163 | 0.380 | 0.226 | 0.479 | 0.997 / 0.999 |

**H-graph-topology passes** on every rung and series: the learned service graph is 0.73–0.86 F1 below
the deployment topology, whose service-level coarsening is the service view's truth almost exactly
(F1 ≥ 0.997). The learned graphs' directed AUROC sits at 0.43–0.52: the mean absolute weight does not
order true service edges above the rest. Licence line: author repository, no licence file; authors
contacted 2026-10-14, no reply as of 2026-10-23.

