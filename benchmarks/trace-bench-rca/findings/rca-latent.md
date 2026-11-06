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
- CIRCA (both rows, the port's window facts stated): test plan sent to Dan Pei 2026-10-26; reply the
  same day forwarding it to the first author (Mingjie Li); no corrections as of 2026-10-27. Code BSD-3-Clause.
- RCD: test plan sent to Azam Ikram, Saurabh Bagchi and Murat Kocaoglu 2026-10-26; no reply as of
  2026-10-27. Code MIT.
- TraceRCA (the suite's implementation, labelled so): test plan sent to Junjie Chen 2026-10-26; no
  reply as of 2026-10-27. Licence line below.

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


## Stretch arms, landed 2026-10-25: `rca/circa` and `rca/rcd` (xs, s, m × five seeds)

Registered by `plans/rca-arms.md`, addenda 2026-10-25 (`rca/circa`, D-TBR-8, with its correction) and
2026-10-25, later (`rca/rcd`, D-TBR-6, D-TBR-7). Reads on the 2026-10-21 exports (every case file as
its export record hashed it): `rca/circa` from commit `8c753630` in `trace-bench-rca` (the `circa`
extra: causal-learn 0.1.4.8, networkx 3.7), `rca/rcd` from commit `761f0949` through `rcd_runner.py` in
the pinned environment `rcaeval-rcd-259ea41` (Python 3.10.22, causal-learn 0.1.2.3 with the suite's four
patched files, numpy 1.23.5, pandas 2.0.1). The records' `git_commit.dirty` is true on every read: the
tracked tree was clean at both commits and the flag reflects the reads' own untracked outputs
appearing under `results/` as the chains ran. No value was chosen on any read. The earlier pooled
files of the regenerated table are byte-identical (20 of 20).

| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 (n services) | misses (no ranking) |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/circa` | 10 | 0.100 | 0.500 | 1.000 | 0.620 [0.480, 0.760] | 0.600 (5) | 0 |
| s | `rca/circa` | 15 | 0.133 | 0.467 | 0.733 | 0.427 [0.267, 0.613] | 0.250 (12) | 0 |
| m | `rca/circa` | 25 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.058 (52) | 0 |
| xs | `rca/rcd` | 10 | 0.630 | 0.880 | 0.880 | 0.818 [0.638, 0.960] | 0.600 (5) | 0 |
| s | `rca/rcd` | 15 | 0.800 | 0.933 | 0.940 | 0.900 [0.758, 1.000] | 0.250 (12) | 0 |
| m | `rca/rcd` | 25 | 0.404 | 0.552 | 0.584 | 0.518 [0.340, 0.687] | 0.058 (52) | 0 |

Across rungs (50 cases): `rca/circa` Avg@5 0.252 [0.168, 0.348]; `rca/rcd` 0.692 [0.576, 0.799]
(against `rca/baro` 0.372 [0.260, 0.488] and `control/rule` 0.468 [0.344, 0.588]). `rca/rcd`'s
AC@k and Avg@5 are run-averaged: per case the share of its ten seeded runs that place the root
cause in the first k (the 2026-10-23 aggregation); 0 of 500 runs failed.

Per fault kind, Avg@5 — `rca/circa`: xs crash 0.48 / degrade 0.76; s crash 0.32 / degrade 0.32 /
pool exhaustion 0.64; m every kind 0.00. `rca/rcd`: xs crash 0.65 / degrade 0.99; s crash 0.70 /
degrade 1.00 / pool exhaustion 1.00; m cache flush 0.15 / crash 0.27 / degrade 0.58 / pool
exhaustion 1.00.

### Hypotheses (`plans/rca-arms.md`, addenda 2026-10-25)

| hypothesis | verdict | the numbers |
|---|---|---|
| **H-rca-circa-random** — `rca/circa`'s Avg@5 interval above the random expectation on every rung | **fail** (xs, m) | xs low 0.480 vs 0.600; s low 0.267 vs 0.250 (pass); m 0.000 vs 0.058 |
| **H-rca-circa-rule** — `rca/circa` at or above `control/rule` | **fail** (every rung) | `rca/circa` − `control/rule` Avg@5: xs −0.020, s −0.253, m −0.272 |
| **H-rca-rcd-random** — `rca/rcd`'s Avg@5 interval above the random expectation on every rung | **pass** | lows 0.638 / 0.758 / 0.340 vs 0.600 / 0.250 / 0.058 |
| **H-rca-rcd-rule** — `rca/rcd` at or above `control/rule` | **pass** (every rung) | `rca/rcd` − `control/rule` Avg@5: xs +0.178, s +0.220, m +0.246 |
| H-circa-topology-gain | **fail** (xs, s) | landed 2026-10-26 — see the section below |

### Reading notes (descriptive, not re-tuned)

- **`rca/rcd` is the first root-cause arm above the rule control on every rung**, and the only
  one whose interval clears the random expectation at m. It does so while ranking few services: a
  run ranks 2.3 / 6.9 / 6.1 services of 5 / 12 / 52 on average (xs / s / m; 1–13 at m), and every
  service it does not rank is a miss for that run (`n_services_unranked_per_run`, mean 2.7 / 5.1 /
  45.9). The F-node's Ψ-PC neighbours are a short list that contains the root cause in most runs
  (m AC@1 0.404 over ten runs per case). The suite's named-dataset preprocessing removed 0.5 / 0.2 /
  0.9 columns per case on average (20 / 48 / 208 in); no constant column in either segment on most
  cases. Wall clock: ≈ 0.1 s per run at xs, 11 s at s (max 52 s), 4.5 s at m (max 20 s); the runs of
  a rung sum to 0.2 / 27 / 19 CPU-min; peak resident 0.12 GB.
- **`rca/circa` finds nothing at m** (0 of 25 cases in the top five of 52, every fault kind) and
  sits at the random expectation at xs. The head ranked every service on every case (no fallback,
  no pruned service: the 2026-10-25 correction's rules did not bite on these exports), on PC graphs
  of 16 / 104 / 362 undirected edges per case (20 / 48 / 208 columns). The registered reading
  stands: the port's hypothesis-testing head resamples the table to 1 s and fits on
  `[inject − 120 s, inject)`, which holds 4 / 8 / 2 bins at xs / s / m, interpolated, and tests the
  10 s ending at injection + 300 s; at m that reference window is two 60 s bins. PC took 0.1 s per
  case at xs, ≈ 3 s at s and ≈ 70–90 s at m (412 s for the slowest five-case corpus); peak resident
  0.70 GB. Whether a head parameterised for the bin width would do better is not a question this
  read answers (the port is run verbatim, D-TBR-8); the paper states the window facts beside the
  row.
- Both arms' records carry the per-case counts the addenda name (`n_edges`, `n_services_unranked`,
  the fallback flag; `n_services_unranked_per_run`, the kept-column counts, every run's wall clock).

## Stretch arms, landed 2026-10-26: `rca/circa-topology` and `rca/tracerca` (xs, s, m × five seeds)

Registered by `plans/rca-arms.md`, addendum 2026-10-25 (`rca/circa-topology`, D-TBR-9) and addenda
2026-10-25, later (2) and 2026-10-26, later (`rca/tracerca`, D-TBR-10; the synthetic client's spans
dropped before the port). Both read the **2026-10-26 exports**: after the owner merged the exporter's
extras (trace-bench PR #14) the benchmark pin moved to `fcb7ddb3` (D-TBR-11), the `rca` tier was
re-pulled from the release (now with `views/end-request/**`) and every corpus re-exported with
`--extras topology spans`; before any read, each case's `data.csv`, `window.json` and `services.json`
were asserted byte-identical (sha256) to the 2026-10-21 export records, and every input of the earlier
export hashed the same (the four view files are the only new inputs). `rca/circa-topology` ran from
tree `41c0f9e3` at xs and `9c7b0438` at s and m (its adapter unchanged since `8c753630`); `rca/tracerca`
from `9c7b0438` (the correction commit). The records' `git_commit.dirty` is true on every read (the
reads' own untracked outputs). No value was chosen on any read. The earlier pooled files of the
regenerated table are byte-identical (28 of 28).

| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 (n services) | misses (no ranking) |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/circa-topology` | 10 | 0.100 | 0.500 | 1.000 | 0.500 [0.360, 0.680] | 0.600 (5) | 0 |
| s | `rca/circa-topology` | 15 | 0.133 | 0.267 | 0.333 | 0.253 [0.067, 0.453] | 0.250 (12) | 0 |
| m | `rca/circa-topology` | 25 | 0.000 | 0.000 | 0.080 | 0.024 [0.000, 0.056] | 0.058 (52) | 0 |
| xs | `rca/tracerca` | 10 | 0.200 | 0.700 | 0.900 | 0.640 [0.440, 0.820] | 0.600 (5) | 0 |
| s | `rca/tracerca` | 15 | 0.400 | 0.600 | 0.600 | 0.520 [0.280, 0.747] | 0.250 (12) | 0 |
| m | `rca/tracerca` | 25 | 0.000 | 0.080 | 0.200 | 0.088 [0.024, 0.168] | 0.058 (52) | 0 |

Across rungs (50 cases): `rca/circa-topology` Avg@5 0.188 [0.112, 0.272]; `rca/tracerca` 0.328
[0.220, 0.436] (against `rca/circa` 0.252 [0.168, 0.348], `rca/baro` 0.372 [0.260, 0.488],
`control/rule` 0.468 [0.344, 0.588] and `rca/rcd` 0.692 [0.576, 0.799]).

Licence line (`rca/tracerca`): the standard suite's implementation at `259ea41` (MIT) is run; the
authors' repository carries no licence file and is cited, not run; authors contacted 2026-10-26, no reply
as of 2026-10-27.

Per fault kind, Avg@5 — `rca/circa-topology`: xs crash 0.28 / degrade 0.72; s crash 0.40 / degrade
0.00 / pool exhaustion 0.36; m degrade 0.06, every other kind 0.00. `rca/tracerca`: xs crash 0.40 /
degrade 0.88; s crash 0.24 / degrade 0.40 / pool exhaustion 0.92; m cache flush 0.08 / degrade 0.18 /
crash 0.00 / pool exhaustion 0.00.

### Hypotheses (`plans/rca-arms.md`, addenda 2026-10-25)

| hypothesis | verdict | the numbers |
|---|---|---|
| **H-rca-circa-topology-random** — interval above the random expectation on every rung | **fail** (every rung) | lows 0.360 / 0.067 / 0.000 vs 0.600 / 0.250 / 0.058 |
| **H-rca-circa-topology-rule** — at or above `control/rule` | **fail** (every rung) | arm − rule Avg@5: xs −0.140, s −0.427, m −0.248 |
| **H-circa-topology-gain** — the structural-graph cell at or above the PC cell on every rung | **fail** (xs, s) | topology − PC Avg@5: xs −0.120, s −0.173, m +0.024 |
| **H-rca-tracerca-random** — interval above the random expectation on every rung | **fail** (xs, m) | lows 0.440 / 0.280 / 0.024 vs 0.600 / 0.250 / 0.058 (s passes) |
| **H-rca-tracerca-rule** — at or above `control/rule` | **fail** (s, m) | arm − rule Avg@5: xs 0.000 (tie, passes), s −0.160, m −0.184 |

### Reading notes (descriptive, not re-tuned)

- **The structural graph does not carry CIRCA's head on these bins.** `rca/circa-topology` hands the
  port's own `rht` head a graph of 43.8 / 248 / 3,061 edges per case (xs / s / m; from 4.8 / 35 / 484
  deployed-service call edges over 20 / 48 / 208 metric columns) against PC's 16 / 104 / 362, ranks
  every service on every case (no pruned service, no fallback — PC is not run), and lands below the PC
  cell at xs and s and at 0.024 beside its 0.000 at m, both under the random expectation. Both cells
  share the head's registered window (the 1 s resample, 4 / 8 / 2 interpolated reference bins, the
  10 s test window five minutes after the injection); the graph only decides which parents each metric
  is regressed on inside that window, so the gain the paper attributes to the structural graph is not
  measurable here. Wall clock 1–3 s per corpus; peak resident 0.24 GB.
- **`rca/tracerca` ties the rule control at xs and sits near random at m.** The suite's
  reimplementation flagged 1.6 / 1.5 / 1.3 % of post-fault spans (per case 115 / 2,237 / 1,409
  abnormal spans of 7,103 / 147,230 / 108,753), ranked 10.9 of 14 / 54.0 of 59 / 275 of 521
  operations per case and left 2 / 0 / 24 service slots unranked (services with no abnormal span —
  misses at every k); neither registered miss rule fired (no `KeyError`, no empty ranking) on any of
  the 50 cases. At s it clears the random expectation (0.520 vs 0.250) but trails `control/rule`'s
  0.680; at m its interval [0.024, 0.168] contains the random 0.058. Its strongest kind is pool
  exhaustion at s (0.92) and its weakest the m faults of every kind but degrade. The input is the
  reconstructed request view — the correlator's measured loss (paper §3) is in it — and the port
  counts spans, not traces, with no trace-level anomaly-detection stage. The synthetic client's root
  spans, 4.1 / 3.8 / 3.2 % of the file, were dropped before the port (correction 2026-10-26; recorded
  per case as `n_spans_client`). Wall clock 2 / 13 / 26 s per corpus (the m fear of minutes per case
  did not materialise: ≈ 1.2 M spans per case in pandas), peak resident 1.75 GB.
- The two 2026-10-26 export stages wrote `topology.json` (4 / 35 / 501 deployed-service edges) and
  `spans.csv` (xs 6,243 / 4,307 traces per case; s ≈ 310–430 k; m ≈ 105–316 k) beside the three
  unchanged files; their records hash every input, including the four view files the spans come from.


## Stretch arm, landed 2026-10-28: `rca/pc-pagerank` (xs, s, m × five seeds)

Registered by `plans/rca-arms.md`, addendum 2026-10-28 (owner go of 2026-10-28; D-TBR-12, the named-dataset
preprocessing). The standard suite's PC-PageRank: PC (causal-learn defaults) over every metric column of
the whole case table, pre- and post-fault rows pooled — the graph `rca/circa`'s port builds — then
scikit-network PageRank (α 0.85, power iteration, ten iterations) against the edges, toward causes, the
head CausalRCA uses; one deterministic run per case; the port never reads the injection time, so the arm
declares no window. Reads on fresh exports of the three registered files (2026-10-27/28, no extras; each
case's `data.csv`, `window.json` and `services.json` sha256-equal to the 2026-10-26 export records), from
commit `ab6d6995` in `trace-bench-rca` (the `pagerank` extra: scikit-network 0.33.5, whose `__version__`
prints 0.33.0; causal-learn 0.1.4.8, networkx 3.7, numpy 2.5.4 unchanged). The records' `git_commit.dirty`
is true on every read (the reads' own untracked outputs; the xs and m seed-0 pull and export records
name `15a6b5f2`, the trunk the branch was cut from, the others `ab6d6995`). No value was chosen on any
read. The earlier pooled files of the regenerated table are byte-identical (36 of 36).

| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 (n services) | misses (no ranking) |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/pc-pagerank` | 10 | 0.200 | 0.800 | 1.000 | 0.660 [0.480, 0.820] | 0.600 (5) | 0 |
| s | `rca/pc-pagerank` | 15 | 0.067 | 0.333 | 0.467 | 0.293 [0.120, 0.493] | 0.250 (12) | 0 |
| m | `rca/pc-pagerank` | 25 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.058 (52) | 0 |

Across rungs (50 cases): `rca/pc-pagerank` Avg@5 0.220 [0.128, 0.320] (against `rca/circa` 0.252
[0.168, 0.348], `rca/causalrca` 0.214 / 0.203 (p50 / p95 latency), `rca/baro` 0.372 [0.260, 0.488],
`control/rule` 0.468 [0.344, 0.588] and `rca/rcd` 0.692 [0.576, 0.799]).

Per fault kind, Avg@5 — xs crash 0.56 / degrade 0.76; s crash 0.36 / degrade 0.36 / pool exhaustion
0.16; m every kind 0.00 (cache flush, crash, degrade, pool exhaustion).

### Hypotheses (`plans/rca-arms.md`, addendum 2026-10-28)

| hypothesis | verdict | the numbers |
|---|---|---|
| **H-rca-pc-pagerank-random** — interval above the random expectation on every rung | **fail** (every rung) | lows 0.480 / 0.120 / 0.000 vs 0.600 / 0.250 / 0.058 |
| **H-rca-pc-pagerank-rule** — at or above `control/rule` | **fail** (s, m) | arm − rule Avg@5: xs +0.020, s −0.387, m −0.272 |

### Reading notes (descriptive, not re-tuned)

- **The two gaps the row was added for.** *The learner under the same head* — PC-PageRank − CausalRCA
  (Avg@5): +0.064 / +0.045 / −0.040 against the p50-latency series and +0.038 / +0.089 / −0.034 against
  the p95 series (xs / s / m); every pair of intervals overlaps, and at m both learners are at or below
  the random expectation (0.000 and 0.040 / 0.034 vs 0.058). Under the PageRank head the classical
  learner on all four series is no better than the neural one on one latency series. *The head under
  the same graph* — PC-PageRank − CIRCA: +0.040 / −0.134 / 0.000; the two cells share the PC graph case
  by case (P5; 16.4 / 103.7 / 362.1 undirected edges per case here against CIRCA's 16 / 104 / 362), so
  the s gap is the head alone: CIRCA's regression-based hypothesis test on its 1 s window, for all its
  2–8 interpolated reference bins, does better at s than the stationary centrality of the pooled graph;
  at m both score 0.000.
- **PC-PageRank finds nothing at m** (0 of 25 cases in the top five of 52; the root cause's rank is 8–50,
  median 23) and sits at the random expectation at xs (0.660 vs 0.600, the interval containing it) and
  at s (0.293 vs 0.250; AC@1 0.067, the root cause's median rank 7 of 12). The ranking is the
  stationary centrality of a graph learned on the pooled table — the fault window is never read — so a
  service that is central before the fault stays central after it; a root cause has to be central in
  the pooled PC graph to rank, which the injected faults of the larger rungs do not make it.
- **Rankings are complete and nearly tie-free.** Every service ranked on every case (no fallback, no
  empty ranking, 0 unranked service slots); the named-dataset path dropped 0 constant columns on every
  case, so the PC graph is exactly CIRCA's. Distinct PageRank scores per case: 12–17 of 20 (xs), 41–43 of
  48 (s), 146–172 of 208 (m); the ties are the isolated metrics of the PC graph (equal centrality) and sit
  at the tail. The root cause's best-ranked metric shares its score with another metric on 3 cases of 50,
  all at xs (a pair; on one of them the pair spans two services, so the root cause's position is the
  export's column order) and on none at s or m. A diagnostic only; nothing was re-ranked.
- **Cost.** PC per case 1.8–2.2 s / 7.6–9.4 s / 51–57 s per corpus-average at xs / s / m (rank stage wall
  10 s / 43 s / 23 min for the five corpora of each rung, longest corpus 288 s); peak resident 0.28 /
  0.33 / 0.68 GB; PageRank negligible. Pull + export per corpus: xs 4–52 s + 1 s, s 118–220 s + 38–49 s,
  m 86–200 s + 62–69 s, raw shards (≈ 3 GB at m) deleted after each export.
- The records carry the per-case counts the addendum names (`n_edges`, `n_metrics_dropped`,
  `n_distinct_scores`, `tie_groups`, `n_services_unranked`, the fallback flag, `n_filled`), and the
  score records the root cause's tie group (`root_tie_groups`, `n_cases_root_in_tie_group`).

## Stretch arms, landed 2026-10-30: `rca/trace` and `rca/trace-core` (xs, s, m × five seeds)

Registered by `plans/rca-arms.md`, addendum 2026-10-30 (owner go of 2026-10-30; D-TBR-13). The reference
algorithm under its own root-cause rule — the rule the owner proposed to the method's author on 2026-10-21
as the lab's reading of his ask of 2026-10-19 (the Markov boundary of the outcome, then the upstream
traversal of the sample graph); the author neither confirmed nor declined it and reviews the draft. Per
case, every trace of the end-ordered request view whose start falls in the fault window (all three
splits; the first 500 by start time) is probed once with the seed's frozen backbone and the arm's frozen
request cell of the committed freeze (`trace/cli-atomic/shipped/frozen/request` for `rca/trace`,
`trace/core/shipped/frozen/request` for `rca/trace-core`) through the reference harness's engine adapter
(bit-identical to a CPU `discover` read of the same cell on six stored traces, P4); the sample graph is
the frozen cut; the parents of each non-OK token (`5xx`, `err`, `slow`) and their upstream closure map
to services; a service's score is the number of probed traces in which it is upstream of a non-OK
token (ties by mean hop, then `services.json` order; never-upstream services after). Reads on fresh
2026-10-30 exports of the three registered files (each case's `data.csv`, `window.json` and
`services.json` sha256-equal to the 2026-10-28 export records), from commit `e76c2ad5` in
`trace-bench-rca`, the probe in `seq2cause-trace-bench` (torch 2.14.1, CPU). The records' `git_commit.dirty`
is true on every read (the reads' own untracked outputs). No value was chosen on any read; no case was
a miss. The earlier pooled files of the regenerated table are byte-identical (40 of 40).

| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 (n services) | misses (no ranking) |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/trace` | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 (5) | 0 |
| s | `rca/trace` | 15 | 0.133 | 0.267 | 0.600 | 0.333 [0.160, 0.520] | 0.250 (12) | 0 |
| m | `rca/trace` | 25 | 0.000 | 0.040 | 0.200 | 0.080 [0.016, 0.160] | 0.058 (52) | 0 |
| xs | `rca/trace-core` | 10 | 0.100 | 0.600 | 1.000 | 0.560 [0.380, 0.740] | 0.600 (5) | 0 |
| s | `rca/trace-core` | 15 | 0.133 | 0.267 | 0.467 | 0.307 [0.120, 0.507] | 0.250 (12) | 0 |
| m | `rca/trace-core` | 25 | 0.000 | 0.200 | 0.240 | 0.144 [0.048, 0.256] | 0.058 (52) | 0 |

Across rungs (50 cases): `rca/trace` Avg@5 0.248 [0.160, 0.340], `rca/trace-core` 0.276 [0.184, 0.372]
(against `rca/tracerca` 0.328 [0.220, 0.436], `rca/baro` 0.372 [0.260, 0.488], `control/rule` 0.468
[0.344, 0.588] and `rca/rcd` 0.692 [0.576, 0.799]).

Per fault kind, Avg@5 — `rca/trace`: xs crash 0.28 / degrade 0.80; s crash 0.24 / degrade 0.04 / pool
exhaustion 0.72; m cache flush 0.00 / crash 0.08 / degrade 0.16 / pool exhaustion 0.00. `rca/trace-core`:
xs 0.28 / 0.84; s 0.20 / 0.00 / 0.72; m 0.00 / 0.12 / 0.30 / 0.00.

### Hypotheses (`plans/rca-arms.md`, addendum 2026-10-30)

| hypothesis | verdict | the numbers |
|---|---|---|
| **H-rca-trace-random** — interval above the random expectation on every rung | **fail** (every rung) | lows 0.360 / 0.160 / 0.016 vs 0.600 / 0.250 / 0.058 |
| **H-rca-trace-rule** — at or above `control/rule` | **fail** (every rung) | arm − rule Avg@5: xs −0.100, s −0.347, m −0.192 |
| **H-rca-trace-core-random** | **fail** (every rung) | lows 0.380 / 0.120 / 0.048 vs 0.600 / 0.250 / 0.058 |
| **H-rca-trace-core-rule** | **fail** (every rung) | arm − rule Avg@5: xs −0.080, s −0.373, m −0.128 |

### Reading notes (descriptive, not re-tuned)

- **The three gaps the rows were added for.** *The two trace-reading arms* — `rca/trace` − `rca/tracerca`
  (Avg@5): −0.100 / −0.187 / −0.008 (xs / s / m); every pair of intervals overlaps. *The construction* —
  `rca/trace` − `rca/trace-core`: −0.020 / +0.027 / −0.064; the two rows are within 0.07 of each other on
  every rung and their intervals overlap. *The best published arm* — `rca/trace` − `rca/rcd`: −0.278 /
  −0.567 / −0.438. Against `rca/baro`: −0.180 / −0.133 / −0.096.
- **Where the rule finds the root cause.** Pool exhaustion at s (0.72 on both rows: first on two seeds of
  five, within the top four on the other three) and the degrade faults at xs (0.80 / 0.84: second of five
  on every seed but one, where the full construction places it first). The crash faults sit last of five
  at xs on four seeds (0.28), and at m cache flush and pool exhaustion are never in the top five (0.00);
  the m degrade faults are the rows' best m cells (0.16 / 0.30). The root cause's median rank is 2.5 / 5 /
  14 (`rca/trace`) and 2.5 / 6 / 15 (`rca/trace-core`) of 5 / 12 / 52.
- **The sample graphs reach almost every service.** With 500 probed traces per case the upstream closures
  touch every deployed service at xs and s and 51.2 / 51.0 of 52 at m on average, so the ranking is a
  count over traces, not a short list: 106,866 / 70,742 cut edges over the 12,500 probed m traces, 1.4 / 1.4
  non-OK tokens per probed trace at s (0.7 at xs, 1.5 at m), and a non-OK token with at least one
  upstream service in 47 % (s) and 48 % / 44 % (m) of the probed traces. The non-OK token's own service is
  never scored by its own failure (hop 0 is a diagnostic), which is where a crashed leaf service's
  evidence sits: the crash rows are the weakest at xs.
- **What the rows read.** The fault windows hold 1,046–2,144 (xs), 17,429–43,836 (s) and 10,722–32,016 (m)
  traces per case; the registered cap probes the first 500 by start time, i.e. 23–48 % of an xs window and
  1.1–4.8 % of an s or m window — the first minute or so after the injection at the larger rungs. Traces
  with fewer than two positions after the context are skipped (`rca/trace` at xs: 648 of 5,000 at c = 3;
  s 37; m 0; `rca/trace-core` xs 5, s and m 0); no trace was truncated at 64 tokens; 6 / 0 / 17 folded
  positions per rung; 200–674 client positions dropped per rung. The backbone saw the train split of the
  same corpus in pretraining: the root-cause read is not a held-out read on the discovery axis.
- **Device.** The probe ran on the CPU of the laptop. A CPU re-probe of six stored traces of the landed xs
  seed-0 test read with the test read's own per-sequence seeds reproduces the stored matrices bit for bit
  when the stored read is itself a CPU `discover` read, and differs from the landed GPU read by the
  device's noise stream (N = 2 particles; 3 of 6 edge sets equal), so the rows here are a CPU replica of
  the frozen cells with their own noise draws, as the addendum allows and discloses.
- **Cost.** Probe wall clock per corpus 5–38 s (xs), 9–86 s (s), 20–138 s (m) for `rca/trace` and 5–12 / 10–79 /
  118–138 s for `rca/trace-core` (the N = 32 cells are the slow ones); rank stage ≤ 2.5 min per corpus;
  peak resident 0.1 GB; the 30 rank + score runs in 29 min of wall clock. Pull + export per corpus: xs 3–51 s
  + ≤ 1 s, s 97–137 s + 22–28 s, m 129–186 s + 34–38 s, raw shards (≈ 3 GB at m) deleted after each export.
- The records carry the per-case counts the addendum names (`n_traces_window`, `n_traces_probed`,
  `n_skipped_short`, `n_truncated`, `n_folded`, `n_nonok_positions`, `n_traces_with_nonok`,
  `n_traces_with_upstream`, `n_edges`, `n_client_positions_dropped`, `n_services_reached`, the per-service
  `counts` and `mean_hops`, `self_counts`, the miss reason), the cell's `(c, N, g, tau)`, the model and
  freeze hashes, both environments' `pip freeze` and the runner's import paths and commit.


## The context family's root-cause rows, landed 2026-11-02: `rca/trace-{ig,cmi}-{overwrite,keep}`, `rca/trace-atomic-cmi-keep`, the `hop0` / `source` readings, the pre-fault subtraction, `control/trace-earliest` (xs × five seeds; case level)

Registered by `plans/rca-arms.md`, addendum "2026-10-31, later" (owner decisions 2026-10-30 / 31; D-TBR-16 … 19).
The five arms read the request cells of the committed context freezes (`freezes/2026-11-01-xs-latent-s<k>-context.json`
of the reference harness; seq2cause v0.3.1, the `fixed-kl` path) with the same case rule as `rca/trace` (the window
traces of the end-ordered request view, the first 500 by start time, all splits; 995–1,000 probed per corpus): one
forward per layout (the information-gain and CMI cells of the core overwrite layout share a probe, as do the two keep
cells; the atomic CMI arm is its own forward), re-ranked under the three readings. `hop0` is the primary reading
(the failed position's own service credited at hop 0); `hop1` the upstream-closure rule as landed; `source` the
author's rule (only the sources of the non-OK subgraph). The post − pre rows subtract the per-trace credit rate on the
pre-fault stretch (`[pre_start, inject_time)`: the corpus start for the first case of a corpus, the preceding case's
end for the second; 997–1,000 traces probed, nearest the fault first; every case has a pre stretch, `n_cases_no_pre`
0). The landed `rca/trace` and `rca/trace-core` gain `hop0` and `source` rows from a **CPU re-probe of the landed
selection** (their 2026-10-30 work files were purged before the re-rank — the addendum's disclosed fallback; the same
cells, selection, cap and seed stream; their `hop1` rows stand as landed). `control/trace-earliest` ranks the service
holding each selected trace's earliest non-OK span. Code `8052ecc9`; the probe in `seq2cause-trace-bench` (torch 2.15,
CPU); records `dirty: true` (the reads' own untracked outputs). xs only: the s and m case-level rows and the per-trace
axis (the GPU replica of the three layouts) follow; `report` requires the family's `hop0` row on every rung it is
asked for, so the all-arm table is `tables/rca-latent-context-xs/` and the landed `tables/rca-latent/` (regenerated,
48 of 48 landed pooled files byte-identical) carries only the landed arms' new `hop0` / `source` xs rows beside them.

| arm (xs, 10 cases) | hop1 | **hop0** | source | hop1 − pre | **hop0 − pre** | source − pre |
|---|---|---|---|---|---|---|
| `rca/trace-ig-overwrite` (c 2, τ 0.001) | 0.540 [0.360, 0.720] | **0.500 [0.340, 0.660]** | 0.640 [0.440, 0.840] | 0.800 [0.620, 0.940] | **0.840 [0.660, 0.980]** | 0.880 [0.740, 0.980] |
| `rca/trace-cmi-overwrite` (c 2, τ 0.001) | 0.520 [0.360, 0.680] | **0.500 [0.340, 0.660]** | 0.660 [0.460, 0.840] | 0.800 [0.620, 0.940] | **0.820 [0.620, 0.980]** | 0.880 [0.740, 0.980] |
| `rca/trace-ig-keep` (c 1, τ 0.0003) | 0.600 [0.420, 0.800] | **0.560 [0.380, 0.740]** | 0.680 [0.480, 0.880] | 0.800 [0.620, 0.940] | **0.820 [0.620, 0.980]** | 0.880 [0.740, 0.980] |
| `rca/trace-cmi-keep` (c 1, τ 0.0003) | 0.580 [0.400, 0.760] | **0.540 [0.360, 0.720]** | 0.700 [0.500, 0.880] | 0.800 [0.620, 0.940] | **0.820 [0.620, 0.980]** | 0.880 [0.740, 0.980] |
| `rca/trace-atomic-cmi-keep` (c 1, τ 1e-7) | 0.600 [0.440, 0.740] | **0.540 [0.380, 0.700]** | 0.700 [0.520, 0.880] | 0.780 [0.560, 0.940] | **0.780 [0.560, 0.940]** | 0.880 [0.740, 0.980] |
| `rca/trace` (landed; hop0 / source re-probed) | 0.540 [0.360, 0.720] | 0.460 [0.320, 0.620] | 0.580 [0.380, 0.780] | — | — | — |
| `rca/trace-core` (landed; hop0 / source re-probed) | 0.560 [0.380, 0.740] | 0.540 [0.360, 0.720] | 0.640 [0.440, 0.840] | — | — | — |
| `control/trace-earliest` | 0.440 [0.340, 0.540] | | | | | |
| `control/rule` · `rca/rcd` · random (5 services) | 0.640 · 0.818 · 0.600 | | | | | |

Avg@5 with the bootstrap 95 % interval (B = 2000, seed 0); the frozen (c, τ) are those of seed 0 (τ varies by seed:
1e-5 … 1e-3 on the core cells; the atomic arm froze at the grid bottom 1e-7 on seeds 0 and 2). Per seed, `hop0`:
ig-overwrite 0.5 / 0.5 / 0.5 / 0.6 / 0.4, cmi-overwrite 0.5 / 0.5 / 0.5 / 0.6 / 0.4, ig-keep 0.5 / 0.5 / 0.5 / 0.7 /
0.6, cmi-keep 0.5 / 0.5 / 0.5 / 0.6 / 0.6, atomic 0.5 / 0.5 / 0.5 / 0.6 / 0.6; `source`: 0.6 / 0.6 / 0.5 / 0.8 /
0.7, 0.6 / 0.7 / 0.5 / 0.8 / 0.7, 0.7 / 0.6 / 0.5 / 0.9 / 0.7, 0.7 / 0.7 / 0.5 / 0.9 / 0.7, 0.7 / 0.6 / 0.6 / 0.9 /
0.7; `hop0 − pre`: 0.8 / 1.0 / 0.6 / 0.9 / 0.9 (ig-overwrite), 0.7 / 1.0 / 0.6 / 0.9 / 0.9 (the other core arms),
0.5 / 0.9 / 0.6 / 0.9 / 1.0 (atomic); `source − pre`: 0.9 / 1.0 / 0.7 / 0.9 / 0.9 on every arm.

### Hypotheses (`plans/rca-arms.md`, addendum "2026-10-31, later"; mechanical in `report`)

| hypothesis | verdict (xs) | the numbers |
|---|---|---|
| **H-rca-hop0** — `hop0` ≥ `hop1` for every TRACE arm | **fail** (all seven arms) | hop0 − hop1: −0.04 / −0.02 / −0.04 / −0.04 / −0.06 (the five), −0.08 (`rca/trace`), −0.02 (`rca/trace-core`) |
| **H-rca-ctx-rule** — each new arm's `hop0` ≥ `control/rule` | **fail** (all five) | −0.14 / −0.14 / −0.08 / −0.10 / −0.10 |
| **H-rca-ctx-rcd** — each new arm's `hop0` ≥ `rca/rcd` | **fail** (all five) | −0.32 / −0.32 / −0.26 / −0.28 / −0.28 |
| **H-rca-source** — `source` ≥ `hop1` for every TRACE arm | **pass** (all seven) | +0.10 / +0.14 / +0.08 / +0.12 / +0.10; +0.04 (`rca/trace`), +0.08 (`rca/trace-core`) |
| **H-rca-presub** — post − pre ≥ post under `hop0` for every new arm | **pass** (all five) | +0.34 / +0.32 / +0.26 / +0.28 / +0.24 |
| **H-rca-atomic-cmi** — the atomic CMI arm's `hop0` ≥ `rca/trace`'s `hop0` | **pass** | 0.54 vs 0.46 (+0.08) |
| H-trace-earliest (per trace, hit@1) | not computed | the per-trace axis of the family is the GPU replica; at case level `control/trace-earliest` 0.44 sits under every TRACE `hop0` row (0.46–0.56) |
| the per-arm random / rule hypotheses on the `hop0` rows | **fail** (all five) | interval lows 0.34–0.38 vs random 0.60; arm − rule −0.08 … −0.14 |

### Reading notes (descriptive, not re-tuned)

- **Hop 0 costs at xs.** Crediting the failed position's own service lowers Avg@5 on every arm (−0.02 … −0.08):
  at xs the non-OK token sits in the client-facing or edge service, which hop 0 then ranks above the upstream
  cause; `rca/trace`'s crash cases fall from 0.28 to 0.24 on the fault-kind split, its degrade cases from 0.80 to
  0.68. The author's intuition that hop 0 is "the natural reading" does not hold on this rung; the `source` rule he
  would sign does — it is the best post-only reading on all seven arms (0.58–0.70) and lifts the degrade cases to
  0.88–0.92 and the top-1 rate to 0.2–0.4.
- **The pre-fault subtraction is the large effect.** Subtracting the pre-fault credit rate raises every arm by
  0.24–0.34 under `hop0` (to 0.78–0.84, around `rca/rcd`'s 0.82) and to 0.88 under `source` on every arm — above
  `control/rule` (0.64) and `rca/rcd` (0.82), with AC@1 0.5–0.6 against 0.0–0.1 for the post-only rows. The pre stretch holds as many probed
  traces as the window (997–1,000 at the cap), its credit rates run at a quarter to a half of the post rates on the
  services the post rows over-credit (seed 0 case-00: `tb-limle` 0.40 post / 0.20 pre, `tb-edge` 0.24 / 0.08,
  `tb-kamko` 0.29 / 0.06), and the subtraction promotes the service whose rate rose most. The intervals of the
  post − pre rows still overlap `rca/rcd`'s. Alternatives were fixed before any read (addendum); the definition is
  flagged to the author.
- **Score and context do not separate at case level.** Information gain vs CMI at matched layout and reading:
  0.00 … +0.02 (`hop0`), −0.02 … +0.02 (`source`), 0.00 … +0.02 (`hop0 − pre`); keep vs overwrite: +0.04 … +0.06
  (`hop0`, `hop1`, `source`), −0.02 … 0.00 (post − pre) — every pair of intervals overlaps. The atomic CMI arm reads
  as the keep arms do (0.54 / 0.70 / 0.78 / 0.88); against its landed sibling `rca/trace` it is +0.06 at `hop1`
  and +0.08 at `hop0` (H-rca-atomic-cmi passes), +0.12 at `source`.
- **The earliest-error control** (0.44) ranks below every TRACE reading at case level and below random: at xs the
  earliest non-OK span of a window trace is mostly the edge or client-facing service, which the control places first
  on every case; the sample graphs add 0.06–0.26 over "the first thing that failed" at `hop0` and `source`, more
  with the subtraction.
- **The landed arms' re-probe.** `rca/trace` probed 851 / 980 / 1,000 / 766 / 755 traces (the c = 3 cell skips the
  short traces as landed; the counts equal the landed records'), `rca/trace-core` 995 / 1,000 × 4; a `hop1` re-rank
  of the re-probed work files is not landed (the landed `hop1` rows stand); the device's draws differ from the
  landed CPU run's only by the torch version (2.15 against 2.14.1), undisclosed in magnitude — the P4 tolerance
  check of 2026-11-01 (GPU vs CPU, Jaccard 0.79–0.98) bounds the kind of drift.
- **Cost.** Probe ≤ 1 min per layout and phase per corpus (eight probes per seed), rank and score seconds; the
  five seeds in ≈ 25 min of wall clock. Pull 5–8 s and export ≤ 1 s per corpus.

## The context family's root-cause rows, landed 2026-11-06: s and m (case level, five seeds each; the family on xs / s / m complete)

The s and m case-level rows of the five context arms, the landed arms' `hop0` / `source` re-read and `control/trace-earliest`
(`plans/rca-arms.md`, addendum "2026-10-31, later"; D-TBR-16 … 19), run on one EC2 replica (`seq2cause-bench-26-11-06-rca-ctx-s-m.tfvars`,
g5.xlarge; job script `scripts/jobs/rca-ctx-s-m-26-11-06.sh`; code `36eefab3`) with the probes on the A10G (`--device cuda`; the xs rows of
2026-11-02 ran on the CPU). Same case rule as xs: the window traces of the end-ordered request view, the first 500 by start time per case
(1,495–1,500 probed per s corpus, 3 cases; 2,500 per m corpus, 5 cases), the pre-fault stretch under the same cells; one probe per layout
and phase, re-ranked under `hop1` / `hop0` / `source` (`runner_reused`); the probe layouts come from each seed's context freeze — on s seeds
2 and 4 the two overwrite arms froze at different request contexts (c = 1 and c = 2) and were probed separately, elsewhere the information-gain
and CMI cells of a layout share one forward. Every s row stages on the xs row of the same seed, arm and series (ledger), every m row on the s
row made earlier in the same chain. A first replica (2026-11-05) ran s seeds 0 and 1 and was refused at s seed 2 on that layout split; its
records are not landed, and the re-run reproduced its s0 / s1 rankings on all 100 rows. The landed `rca/trace` and `rca/trace-core` gain
`hop0` / `source` rows from a re-probe of the landed selection at the cap (as at xs; the landed `hop1` rows stand). Scored locally (B = 2000,
seed 0). `tables/rca-latent/` regenerated: 52 / 52 landed pooled files byte-identical, the four cross-rung files of the landed arms' `hop0` /
`source` series now pooling xs + s + m instead of xs alone; `tables/rca-latent-context/` holds every arm on xs / s / m (141 rows) and
supersedes the xs-only table of 2026-11-02 (kept).

| arm (s, 15 cases) | hop1 | **hop0** | source | hop1 − pre | **hop0 − pre** | source − pre |
|---|---|---|---|---|---|---|
| `rca/trace-ig-overwrite` (c 1) | 0.31 [0.12, 0.51] | **0.24 [0.07, 0.44]** | 0.47 [0.28, 0.64] | 0.53 [0.33, 0.71] | **0.51 [0.31, 0.69]** | 0.76 [0.60, 0.89] |
| `rca/trace-cmi-overwrite` (c 1; c 2 on seeds 2, 4) | 0.27 [0.11, 0.45] | **0.24 [0.08, 0.43]** | 0.48 [0.29, 0.65] | 0.47 [0.28, 0.65] | **0.55 [0.33, 0.73]** | 0.77 [0.61, 0.91] |
| `rca/trace-ig-keep` (c 1) | 0.31 [0.12, 0.51] | **0.24 [0.07, 0.44]** | 0.47 [0.28, 0.64] | 0.53 [0.33, 0.71] | **0.51 [0.31, 0.69]** | 0.76 [0.60, 0.89] |
| `rca/trace-cmi-keep` (c 1) | 0.25 [0.09, 0.44] | **0.24 [0.07, 0.44]** | 0.47 [0.28, 0.64] | 0.48 [0.29, 0.65] | **0.49 [0.27, 0.71]** | 0.77 [0.61, 0.91] |
| `rca/trace-atomic-cmi-keep` (c 1) | 0.24 [0.08, 0.43] | **0.24 [0.07, 0.43]** | 0.49 [0.31, 0.67] | 0.37 [0.16, 0.59] | **0.53 [0.32, 0.73]** | 0.80 [0.61, 0.95] |
| `rca/trace` (landed hop1; hop0 / source re-probed) | 0.33 [0.16, 0.52] | 0.23 [0.05, 0.43] | 0.53 [0.35, 0.71] | — | — | — |
| `rca/trace-core` (landed hop1; hop0 / source re-probed) | 0.31 [0.12, 0.51] | 0.23 [0.07, 0.43] | 0.47 [0.28, 0.64] | — | — | — |
| `control/trace-earliest` | 0.27 [0.11, 0.44] | | | | | |
| `control/rule` · `rca/rcd` · random (3 cases × 5 services) | 0.68 · 0.90 · 0.25 | | | | | |

| arm (m, 25 cases) | hop1 | **hop0** | source | hop1 − pre | **hop0 − pre** | source − pre |
|---|---|---|---|---|---|---|
| `rca/trace-ig-overwrite` (c 1) | 0.14 [0.05, 0.24] | **0.22 [0.08, 0.39]** | 0.28 [0.14, 0.43] | 0.15 [0.05, 0.27] | **0.20 [0.08, 0.35]** | 0.42 [0.26, 0.59] |
| `rca/trace-cmi-overwrite` (c 1) | 0.14 [0.05, 0.26] | **0.23 [0.09, 0.40]** | 0.25 [0.11, 0.40] | 0.16 [0.06, 0.28] | **0.23 [0.10, 0.38]** | 0.41 [0.26, 0.56] |
| `rca/trace-ig-keep` (c 1) | 0.14 [0.05, 0.24] | **0.22 [0.08, 0.39]** | 0.28 [0.14, 0.43] | 0.15 [0.05, 0.27] | **0.20 [0.08, 0.35]** | 0.42 [0.26, 0.59] |
| `rca/trace-cmi-keep` (c 1) | 0.14 [0.05, 0.26] | **0.23 [0.09, 0.40]** | 0.25 [0.11, 0.40] | 0.16 [0.06, 0.28] | **0.23 [0.10, 0.38]** | 0.41 [0.26, 0.56] |
| `rca/trace-atomic-cmi-keep` (c 1) | 0.10 [0.03, 0.19] | **0.23 [0.09, 0.40]** | 0.25 [0.11, 0.39] | 0.14 [0.04, 0.25] | **0.20 [0.07, 0.35]** | 0.41 [0.25, 0.57] |
| `rca/trace` (landed hop1; hop0 / source re-probed) | 0.08 [0.02, 0.16] | 0.21 [0.07, 0.38] | 0.26 [0.12, 0.40] | — | — | — |
| `rca/trace-core` (landed hop1; hop0 / source re-probed) | 0.14 [0.05, 0.26] | 0.23 [0.09, 0.40] | 0.28 [0.14, 0.43] | — | — | — |
| `control/trace-earliest` | 0.23 [0.09, 0.40] | | | | | |
| `control/rule` · `rca/rcd` · random (5 cases × 5 services) | 0.27 · 0.52 · 0.06 | | | | | |

Avg@5 with the bootstrap 95 % interval (B = 2000, seed 0). Per seed, `hop0` at s: 0.33 / 0.13 / 0.33 / 0.20 / 0.20 on each of the four core
arms (identical), atomic 0.33 / 0.20 / 0.33 / 0.20 / 0.13; `source` (ig-overwrite) 0.47 / 0.67 / 0.40 / 0.33 / 0.47; `hop0 − pre` 0.60 / 0.47 /
0.40 / 0.53 / 0.53. At m: `hop0` 0.24 / 0.20 / 0.20 / 0.28 / 0.20 (information gain), 0.24 / 0.20 / 0.20 / 0.32 / 0.20 (CMI and atomic);
`source` 0.32 / 0.24 / 0.28 / 0.36 / 0.20; `hop0 − pre` 0.40 / 0.20 / 0.12 / 0.20 / 0.08. AC@1 at s: 0.13 under `hop0` and `source`
(atomic 0.20), 0.27 after the subtraction; at m 0.20 / 0.16 / 0.12.

### Hypotheses (`plans/rca-arms.md`, addendum "2026-10-31, later"; mechanical in `report` over xs, s, m)

| hypothesis | verdict | xs | s | m |
|---|---|---|---|---|
| **H-rca-hop0** — `hop0` ≥ `hop1` for every TRACE arm | **fail** | −0.02 … −0.08 (all seven) | −0.11 … 0.00 (six below, one equal) | **+0.09 … +0.13 (all seven above)** |
| **H-rca-source** — `source` ≥ `hop1` for every TRACE arm | **pass** | +0.04 … +0.14 | +0.16 … +0.25 | +0.10 … +0.18 |
| **H-rca-presub** — post − pre ≥ post under `hop0` for every new arm | **fail** | +0.24 … +0.34 | +0.25 … +0.31 | **−0.03 … 0.00** |
| **H-rca-atomic-cmi** — the atomic CMI arm's `hop0` ≥ `rca/trace`'s | **pass** | +0.08 | +0.01 | +0.02 |
| **H-rca-ctx-rule** — each new arm's `hop0` ≥ `control/rule` | **fail** | −0.08 … −0.14 | −0.44 | −0.04 … −0.05 |
| **H-rca-ctx-rcd** — each new arm's `hop0` ≥ `rca/rcd` | **fail** | −0.26 … −0.32 | −0.66 | −0.29 |
| H-ctx-keep-context-share (per-trace axis) | not computed | — | — | — |

### Reading notes (descriptive, not re-tuned)

- **The readings decay with the rung faster than the controls.** `hop0` reads 0.50 / 0.24 / 0.22 and `source` 0.64 / 0.47 / 0.28 on
  xs / s / m, against `rca/rcd` 0.82 / 0.90 / 0.52, `control/rule` 0.64 / 0.68 / 0.27 and random 0.60 / 0.25 / 0.06. Over random the
  sample graphs add nothing at xs (`hop0` −0.10, `source` +0.04), +0.22 at s under `source` (−0.01 under `hop0`) and +0.16 / +0.22 at m;
  the gap to `rca/rcd` is 0.3–0.7 on every rung and to the rule 0.04–0.44. At s the rule and RCD are at their best (0.68, 0.90) while
  every TRACE reading is at 0.24–0.53; at m everything falls, the controls most.
- **Hop 0 costs at xs and s and helps at m** (+0.09 … +0.13 over `hop1` on all seven arms; `rca/trace` 0.08 → 0.21). The fault-kind split
  says why: at m the degrade cases carry the reading (`hop0` 0.52 against `hop1` 0.28 on them; crash 0.08, cache-flush and
  pool-exhaust 0.00), consistent with the failed position sitting in the injected service itself at those faults. At s the pool-exhaust cases carry
  every reading (`hop0` 0.52, `source` 0.84, `hop0 − pre` 0.96) while the degrade cases read 0.00 under `hop0` and `hop1` (0.16 under
  `source`, 0.44 after the subtraction). The hypothesis as registered (every arm, every rung) fails; the direction reverses with the rung.
- **The source rule is the best post-only reading on every rung** (H-rca-source passes 21 / 21): +0.23 … +0.25 over `hop0` at s
  (0.47–0.49), +0.03 … +0.06 at m (0.25–0.28). It is the reading the author would sign.
- **The pre-fault subtraction stops working at m.** It adds +0.25 … +0.31 under `hop0` at s (to 0.49–0.55; the pool-exhaust cases to
  0.96, the degrade cases 0.00 → 0.44) and +0.29 … +0.33 under `source` (to 0.76–0.80, still under `rca/rcd` 0.90), but at m it costs
  0.00–0.03 under `hop0` (the degrade cases 0.52 → 0.34; the pool-exhaust cases 0.00 → 0.24) while it still adds +0.14 … +0.17 under
  `source` (to 0.41–0.42). Why it turns is not established here (the pre stretches shorten with five cases per corpus; the per-service pre
  rates were not read). H-rca-presub fails as registered (m); the `source − pre` row remains the family's best row on every rung (0.88 / 0.76–0.80 /
  0.41–0.42) and is the only row at or near `rca/rcd` anywhere (xs only).
- **Score and context do not separate on any rung.** At s the four core arms read the same `hop0` on every seed (0.33 / 0.13 / 0.33 / 0.20 /
  0.20) and the same `source` within 0.01; at m information gain against CMI is −0.01 (`hop0`) and +0.03 (`source`), keep = overwrite exactly
  (every request cell froze at c = 1, so the layouts coincide; on s seeds 2 and 4 the CMI overwrite cell at c = 2 reads as the c = 1 cells do).
  The atomic CMI arm reads as the core arms (`hop0` 0.24 / 0.23) and 0.01–0.02 above `rca/trace` (H-rca-atomic-cmi passes, by less than at xs).
- **The earliest-error control** sits at the TRACE `hop0` level at s (0.27 against 0.24) and m (0.23 against 0.22), above random on both; at
  xs it was below every reading. On the larger corpora "the first service that failed" is as good a case-level guess as hop 0 of the sample graph.
- **The landed arms' re-probe on cuda** reproduces the landed selection (1,495–1,500 / 2,500 probed); `rca/trace` probed 1,500 / 1,500 / 1,495 / 1,468 / 1,500 at s (its c = 3 cell skips
  short traces, as landed) and 2,500 × 5 at m, `rca/trace-core` every selected trace; `rca/trace` reads `hop0` 0.23 / 0.21 and `source`
  0.53 / 0.26, `rca/trace-core` 0.23 / 0.23 and 0.47 / 0.28; their landed `hop1` rows (0.33 / 0.08; 0.31 / 0.14) stand.
- **Cost.** 8 min per s corpus and 8–9 min per m corpus on the g5.xlarge (pull 21–75 s, export 61–115 s, 8–10 probes of 20–35 s, re-ranks
  seconds); the ten corpora in 1 h 37 min of chain; scoring locally 1.5 min per corpus.
