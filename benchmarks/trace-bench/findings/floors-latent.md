# Findings — the floors (`floor/topology`, `floor/bigram`), xs … xl / latent, five seeds

The three hypotheses of `plans/floors.md` §7 are scored here per rung on the five-seed mean of
the paired per-seed difference, as written; the tables the numbers come from are
`tables/<rung>-latent/` (`report`, regenerated with the two floor cells per grain and the six
floor pairs of `plans/floors.md` §6 beside the arms' cells and pairs — the arm cells are identical
to the landed tables, the arm pairs gain only the key `model_free: false`). Source records:
`results/<rung>/latent/seed={0..4}/floor/<arm>/none/frozen/<grain>/` (the test read's `score.json`,
`score-ranking.json`, `annotate.json`), `test-read/floors-<grain>/` and `floors-val-sweep/`,
under the floor freezes `freezes/2026-10-10-<rung>-latent-s<k>-floors.json` (`freeze@3`,
`family: floor`, `ordering: end`, model `none`). Corpora: the published release
`chadyuk/trace-bench@v0.3.0`, score tier; scorer: trace-bench `a3eb104` (`v0.3.0-score3`,
`score.py` byte-identical to the `8e91aa2` pin the arms landed on). xs, s and m were read locally
on 2026-10-10; l and xl were read on CPU hosts the same day (owner-applied test replicas; the box manifest
is landed as `test-read/floors-artifacts.json`, the pull record as `test-read/floors-pull-score/`).

## Caveats carried beside every number

- **Sample-free (topology).** `floor/topology` reads no sequence: its validation and test reads
  are the same computation and differ only in the τ the freeze fixes, so its "test" number is
  not a held-out number and is not sample-paired with any arm (`plans/floors.md` §5). Every
  topology number below carries the label.
- **What the topology τ selects.** The sweep over `floor_taus` chose, on every seed of every rung
  so far, the largest grid value below the smallest `p_call` among the prior's edges (xs: 0.07 …
  0.15), so the frozen cut keeps every prior edge: the frozen cell is the τ = 0 cell, the
  2026-09-28 probe's construction. Recall 1.0 at request is the prior's coverage of the
  request-grain truth, precision the share of its ordered token pairs that are truth edges. l froze
  at τ 0.02 and xl at τ 0.01 on every seed, the same rule.
- **Grid edge (bigram).** A bigram cell frozen at τ = 0 is the cut "every observed adjacent
  pair" (`plans/floors.md` §3; recorded, not refused). xs: seeds 0, 1 and 2 at request; s: seeds
  0, 2 and 4 at request; m: nine of ten cells; l and xl: all ten cells.
- **Sample size, measured at s.** The bigram floor's test F1 lands above its frozen validation value on every seed and grain: +0.022 … +0.038 at request and +0.037 … +0.052 at session, while its coverage ceiling rises 0.259–0.297 → 0.330–0.387 (request) and 0.302–0.343 → 0.424–0.485 (session) between the 10k / 5k validation sample and the 20k / 10k test sample — the same double-sample coverage effect the arms show (`findings/s-latent.md` … `xl-latent.md`); the arms' test reads probe the same test sample, so the bigram pairs are sample-paired, the topology pairs are not.
- **Five pairs: low power.** Every paired t-test is on five per-seed differences; a p near 0.05
  is read as a direction, not a verdict (`report`'s own note).

## xs (freezes 2026-10-10, local test reads 2026-10-10)

### Floor cells, five-seed mean ± std (test read, default floor 0.05)

| cell | directed F1 | precision | recall | AUROC (directed) | AP (directed) | reachable F1 | F1 per seed 0–4 |
|---|---|---|---|---|---|---|---|
| `floor/topology/none/frozen/request` (sample-free) | 0.820 ± 0.072 | 0.699 | 1.000 | 0.972 | 0.663 | 0.820 | 0.840 / 0.888 / 0.832 / 0.698 / 0.839 |
| `floor/bigram/none/frozen/request` | 0.340 ± 0.027 | 0.358 | 0.324 | 0.636 | 0.277 | 0.521 | 0.361 / 0.330 / 0.317 / 0.316 / 0.375 |
| `floor/topology/none/frozen/session` (sample-free) | 0.643 ± 0.072 | 0.699 | 0.596 | 0.782 | 0.500 | 0.820 | 0.676 / 0.711 / 0.662 / 0.522 / 0.645 |
| `floor/bigram/none/frozen/session` | 0.357 ± 0.022 | 0.408 | 0.319 | 0.642 | 0.302 | 0.564 | 0.362 / 0.364 / 0.343 / 0.329 / 0.387 |

Reference arms and baselines at the frozen cut on the same seeds (directed F1, request / session): `trace/core/shipped/frozen` 0.321 / 0.357; `trace/cli/shipped/frozen` 0.317 / 0.353; `trace/cli-atomic/shipped/frozen` 0.307 / 0.332; `baseline/granger/shipped/frozen` 0.321 / 0.354; `baseline/saliency/none/frozen` 0.279 / 0.324; `baseline/shapley/none/frozen` 0.255 / 0.286.

### Registered pairs (`plans/floors.md` §6; floor − arm, paired per seed, five seeds, paired t-test)

| pair | grain | directed F1 | AUROC (directed) | precision | recall |
|---|---|---|---|---|---|
| `floor/topology` − `trace/core/shipped/frozen` | request | +0.498 (p < 0.001) | +0.326 (p < 0.001) | +0.404 (p < 0.001) | +0.628 (p < 0.001) |
| `floor/topology` − `trace/core/shipped/frozen` | session | +0.286 (p < 0.001) | +0.102 (p 0.003) | +0.344 (p 0.001) | +0.235 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | request | +0.499 (p < 0.001) | +0.340 (p < 0.001) | +0.396 (p < 0.001) | +0.656 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | session | +0.289 (p < 0.001) | +0.111 (p 0.001) | +0.355 (p < 0.001) | +0.231 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | request | +0.541 (p < 0.001) | +0.320 (p < 0.001) | +0.424 (p < 0.001) | +0.710 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | session | +0.319 (p < 0.001) | +0.119 (p 0.001) | +0.433 (p < 0.001) | +0.176 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | request | +0.018 (p 0.278) | -0.010 (p 0.419) | +0.063 (p 0.113) | -0.048 (p 0.297) |
| `floor/bigram` − `trace/core/shipped/frozen` | session | +0.001 (p 0.927) | -0.037 (p 0.002) | +0.053 (p 0.013) | -0.043 (p 0.007) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | request | +0.019 (p 0.280) | +0.004 (p 0.598) | +0.055 (p 0.092) | -0.020 (p 0.401) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | session | +0.003 (p 0.685) | -0.029 (p 0.003) | +0.064 (p 0.012) | -0.046 (p 0.018) |
| `floor/bigram` − `baseline/saliency/none/frozen` | request | +0.061 (p 0.015) | -0.015 (p 0.232) | +0.083 (p 0.027) | +0.034 (p 0.213) |
| `floor/bigram` − `baseline/saliency/none/frozen` | session | +0.033 (p 0.017) | -0.020 (p 0.046) | +0.142 (p < 0.001) | -0.101 (p 0.001) |

**H-floor-topology — PASS.** `floor/topology` request F1 0.820 ± 0.072 against every learned arm on every seed: the smallest five-seed mean paired difference is +0.498 (vs `trace/core/shipped/frozen`, min per-seed +0.443, p < 0.001); the range over the 13 learned cells is +0.498 … +0.565, every p < 0.001. Sample-free: the topology arm's validation and test reads coincide except for τ, so this is not a held-out number and the pairs are not sample-paired (`plans/floors.md` §5).

**H-floor-bigram — PASS.** `floor/bigram` − `trace/core/shipped/frozen` directed F1: request +0.018 ± 0.033 (p 0.278, within 0.02 or n.s.); session +0.001 ± 0.011 (p 0.927, within 0.02 or n.s.).

**H-floor-auroc — FAIL.** Session-grain directed AUROC, learned arm − `floor/bigram` (paired): `trace/core/shipped/frozen` +0.037 (p 0.002); `trace/cli/shipped/frozen` +0.032 (p 0.006); `trace/cli-atomic/shipped/frozen` +0.030 (p 0.002); `baseline/granger/shipped/frozen` +0.029 (p 0.003); `baseline/saliency/none/frozen` +0.020 (p 0.046); `baseline/shapley/none/frozen` +0.002 (p 0.744). Fails by `baseline/shapley/none/frozen` (advantage below 0.02).


**Reading.** At xs the deployment topology alone — the call graph the trace program was
instantiated from, read as callee → caller token pairs — recovers the request-grain target with
recall 1.0 and precision 0.70, an F1 of 0.82 that no learned arm approaches (the arms sit at
0.26 … 0.32 at request, 0.29 … 0.36 at session); at session the same prediction loses the
cross-request (journey / retry) edges and lands at 0.64, still 0.29 above `trace/core`. The
bigram floor — the share of sequences in which `b` immediately follows `a` — ties the learned
arms on directed F1 at both grains (+0.018 / +0.001 vs `trace/core`, n.s.; +0.06 / +0.03 above
saliency, p ≈ 0.02) with higher precision and lower recall; what the learned arms keep is a
session-grain ranking advantage of 0.02 … 0.04 directed AUROC over it, which the Shapley baseline
alone does not have. The paper's floor sentence for xs therefore reads: the topology floor is
above every arm by 0.29 … 0.57 F1, the bigram floor is within 0.02 of `trace/core` at both grains
and above the other reference arms by at most 0.03.

## s (freezes 2026-10-10, local test reads 2026-10-10; validation 10k / 5k, test 20k / 10k head sample)

### Floor cells, five-seed mean ± std (test read, default floor 0.05)

| cell | directed F1 | precision | recall | AUROC (directed) | AP (directed) | reachable F1 | F1 per seed 0–4 |
|---|---|---|---|---|---|---|---|
| `floor/topology/none/frozen/request` (sample-free) | 0.839 ± 0.014 | 0.723 | 1.000 | 0.986 | 0.668 | 0.839 | 0.836 / 0.853 / 0.847 / 0.844 / 0.818 |
| `floor/bigram/none/frozen/request` | 0.306 ± 0.025 | 0.291 | 0.339 | 0.653 | 0.236 | 0.435 | 0.307 / 0.337 / 0.292 / 0.320 / 0.273 |
| `floor/topology/none/frozen/session` (sample-free) | 0.740 ± 0.015 | 0.723 | 0.757 | 0.872 | 0.530 | 0.839 | 0.735 / 0.753 / 0.746 / 0.750 / 0.716 |
| `floor/bigram/none/frozen/session` | 0.342 ± 0.014 | 0.328 | 0.363 | 0.702 | 0.247 | 0.462 | 0.347 / 0.356 / 0.324 / 0.351 / 0.329 |

Reference arms and baselines at the frozen cut on the same seeds (directed F1, request / session): `trace/core/shipped/frozen` 0.289 / 0.342; `trace/cli/shipped/frozen` 0.292 / 0.349; `trace/cli-atomic/shipped/frozen` 0.287 / 0.324; `baseline/granger/shipped/frozen` 0.283 / 0.348; `baseline/saliency/none/frozen` 0.268 / 0.277; `baseline/shapley/none/frozen` 0.158 / 0.151.

### Registered pairs (`plans/floors.md` §6; floor − arm, paired per seed, five seeds, paired t-test)

| pair | grain | directed F1 | AUROC (directed) | precision | recall |
|---|---|---|---|---|---|
| `floor/topology` − `trace/core/shipped/frozen` | request | +0.550 (p < 0.001) | +0.324 (p < 0.001) | +0.449 (p < 0.001) | +0.692 (p < 0.001) |
| `floor/topology` − `trace/core/shipped/frozen` | session | +0.398 (p < 0.001) | +0.135 (p < 0.001) | +0.430 (p < 0.001) | +0.346 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | request | +0.556 (p < 0.001) | +0.334 (p < 0.001) | +0.476 (p < 0.001) | +0.660 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | session | +0.392 (p < 0.001) | +0.141 (p < 0.001) | +0.414 (p < 0.001) | +0.360 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | request | +0.571 (p < 0.001) | +0.328 (p < 0.001) | +0.504 (p < 0.001) | +0.652 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | session | +0.463 (p < 0.001) | +0.138 (p < 0.001) | +0.533 (p < 0.001) | +0.246 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | request | +0.017 (p 0.042) | -0.009 (p 0.017) | +0.017 (p 0.582) | +0.031 (p 0.255) |
| `floor/bigram` − `trace/core/shipped/frozen` | session | -0.001 (p 0.882) | -0.035 (p 0.002) | +0.034 (p 0.102) | -0.047 (p 0.018) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | request | +0.023 (p 0.018) | +0.001 (p 0.543) | +0.043 (p 0.093) | -0.001 (p 0.917) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | session | -0.006 (p 0.205) | -0.030 (p 0.003) | +0.018 (p 0.328) | -0.034 (p 0.054) |
| `floor/bigram` − `baseline/saliency/none/frozen` | request | +0.038 (p < 0.001) | -0.005 (p 0.177) | +0.072 (p 0.027) | -0.009 (p 0.727) |
| `floor/bigram` − `baseline/saliency/none/frozen` | session | +0.065 (p < 0.001) | -0.032 (p 0.003) | +0.137 (p < 0.001) | -0.147 (p < 0.001) |

**H-floor-topology — PASS.** `floor/topology` request F1 0.839 ± 0.014 against every learned arm on every seed: the smallest five-seed mean paired difference is +0.547 (vs `trace/cli/shipped/frozen`, min per-seed +0.541, p < 0.001); the range over the 13 learned cells is +0.547 … +0.682, every p < 0.001. Sample-free: the topology arm's validation and test reads coincide except for τ, so this is not a held-out number and the pairs are not sample-paired (`plans/floors.md` §5).

**H-floor-bigram — PASS.** `floor/bigram` − `trace/core/shipped/frozen` directed F1: request +0.017 ± 0.013 (p 0.042, within 0.02 or n.s.); session -0.001 ± 0.010 (p 0.882, within 0.02 or n.s.).

**H-floor-auroc — FAIL.** Session-grain directed AUROC, learned arm − `floor/bigram` (paired): `trace/core/shipped/frozen` +0.035 (p 0.002); `trace/cli/shipped/frozen` +0.035 (p 0.001); `trace/cli-atomic/shipped/frozen` +0.033 (p 0.002); `baseline/granger/shipped/frozen` +0.030 (p 0.003); `baseline/saliency/none/frozen` +0.032 (p 0.003); `baseline/shapley/none/frozen` -0.115 (p < 0.001). Fails by `baseline/shapley/none/frozen` (advantage below 0.02).


**Reading.** s repeats xs with tighter seeds: the topology floor sits at 0.84 request / 0.74
session (the prior's request-grain recall is 1.0 on every seed; its session recall 0.74–0.77 is
the share of session-target edges that are call edges), 0.55 / 0.40 above `trace/core` and
0.55 … 0.68 above every learned cell; the bigram floor ties `trace/core` at both grains
(+0.017 request, −0.001 session) and is above the saliency and Shapley baselines; the learned
arms' session-AUROC advantage over the bigram floor is 0.030 … 0.035 for every arm but Shapley,
which at s falls 0.115 below it. The sample-size note above is measured here for the first time:
the bigram floor gains +0.02 … +0.05 F1 from the validation to the test sample, as the arms do.

## m (freezes 2026-10-10, local test reads 2026-10-10; validation 10k / 5k, test 20k / 10k head sample)

### Floor cells, five-seed mean ± std (test read, default floor 0.05)

| cell | directed F1 | precision | recall | AUROC (directed) | AP (directed) | reachable F1 | F1 per seed 0–4 |
|---|---|---|---|---|---|---|---|
| `floor/topology/none/frozen/request` (sample-free) | 0.850 ± 0.005 | 0.740 | 1.000 | 0.998 | 0.741 | 0.850 | 0.851 / 0.841 / 0.855 / 0.852 / 0.852 |
| `floor/bigram/none/frozen/request` | 0.193 ± 0.004 | 0.255 | 0.155 | 0.575 | 0.095 | 0.407 | 0.186 / 0.196 / 0.192 / 0.194 / 0.197 |
| `floor/topology/none/frozen/session` (sample-free) | 0.826 ± 0.006 | 0.740 | 0.936 | 0.967 | 0.697 | 0.850 | 0.825 / 0.817 / 0.832 / 0.827 / 0.830 |
| `floor/bigram/none/frozen/session` | 0.228 ± 0.008 | 0.237 | 0.221 | 0.610 | 0.124 | 0.382 | 0.217 / 0.228 / 0.224 / 0.238 / 0.232 |

Reference arms and baselines at the frozen cut on the same seeds (directed F1, request / session): `trace/core/shipped/frozen` 0.204 / 0.270; `trace/cli/shipped/frozen` 0.205 / 0.274; `trace/cli-atomic/shipped/frozen` 0.196 / 0.247; `baseline/granger/shipped/frozen` 0.206 / 0.271; `baseline/saliency/none/frozen` 0.192 / 0.222; `baseline/shapley/none/frozen` 0.070 / 0.070.

### Registered pairs (`plans/floors.md` §6; floor − arm, paired per seed, five seeds, paired t-test)

| pair | grain | directed F1 | AUROC (directed) | precision | recall |
|---|---|---|---|---|---|
| `floor/topology` − `trace/core/shipped/frozen` | request | +0.646 (p < 0.001) | +0.409 (p < 0.001) | +0.462 (p < 0.001) | +0.838 (p < 0.001) |
| `floor/topology` − `trace/core/shipped/frozen` | session | +0.556 (p < 0.001) | +0.323 (p < 0.001) | +0.416 (p < 0.001) | +0.704 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | request | +0.644 (p < 0.001) | +0.415 (p < 0.001) | +0.448 (p < 0.001) | +0.841 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | session | +0.555 (p < 0.001) | +0.332 (p < 0.001) | +0.410 (p < 0.001) | +0.706 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | request | +0.658 (p < 0.001) | +0.406 (p < 0.001) | +0.527 (p < 0.001) | +0.824 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | session | +0.604 (p < 0.001) | +0.322 (p < 0.001) | +0.538 (p < 0.001) | +0.679 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | request | -0.011 (p 0.008) | -0.014 (p < 0.001) | -0.023 (p 0.107) | -0.007 (p 0.097) |
| `floor/bigram` − `trace/core/shipped/frozen` | session | -0.042 (p < 0.001) | -0.035 (p < 0.001) | -0.087 (p 0.002) | -0.011 (p 0.038) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | request | -0.013 (p < 0.001) | -0.008 (p < 0.001) | -0.037 (p < 0.001) | -0.004 (p 0.029) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | session | -0.043 (p < 0.001) | -0.025 (p < 0.001) | -0.092 (p 0.001) | -0.009 (p 0.064) |
| `floor/bigram` − `baseline/saliency/none/frozen` | request | +0.001 (p 0.589) | -0.017 (p < 0.001) | +0.043 (p < 0.001) | -0.021 (p < 0.001) |
| `floor/bigram` − `baseline/saliency/none/frozen` | session | +0.006 (p 0.599) | -0.036 (p < 0.001) | +0.036 (p 0.270) | -0.036 (p 0.035) |

**H-floor-topology — PASS.** `floor/topology` request F1 0.850 ± 0.005 against every learned arm on every seed: the smallest five-seed mean paired difference is +0.644 (vs `baseline/granger/fixed/frozen`, min per-seed +0.637, p < 0.001); the range over the 13 learned cells is +0.644 … +0.780, every p < 0.001. Sample-free: the topology arm's validation and test reads coincide except for τ, so this is not a held-out number and the pairs are not sample-paired (`plans/floors.md` §5).

**H-floor-bigram — FAIL.** `floor/bigram` − `trace/core/shipped/frozen` directed F1: request -0.011 ± 0.005 (p 0.008, within 0.02 or n.s.); session -0.042 ± 0.007 (p < 0.001, beyond 0.02 with p < 0.05).

**H-floor-auroc — FAIL.** Session-grain directed AUROC, learned arm − `floor/bigram` (paired): `trace/core/shipped/frozen` +0.035 (p < 0.001); `trace/cli/shipped/frozen` +0.035 (p < 0.001); `trace/cli-atomic/shipped/frozen` +0.036 (p < 0.001); `baseline/granger/shipped/frozen` +0.025 (p < 0.001); `baseline/saliency/none/frozen` +0.036 (p < 0.001); `baseline/shapley/none/frozen` -0.079 (p < 0.001). Fails by `baseline/shapley/none/frozen` (advantage below 0.02).


**Reading.** m is the first rung where the two floors separate from the arms in opposite
directions. The topology floor tightens to 0.85 request / 0.83 session (seed spread ± 0.005; its
session recall is 0.93–0.94, so the m session target is almost entirely call edges), 0.65 / 0.56
above `trace/core` and 0.64 … 0.78 above every learned cell. The bigram floor collapses with the
sample: on the 20k / 10k test sample it reaches only 0.15–0.16 (request) and
0.21–0.23 (session) of the truth edges at all (0.10–0.11 / 0.13–0.14 on the validation sample) (its coverage ceiling; the arms probe the
same sample and sit under the same ceiling), lands at 0.19 / 0.23 F1, within 0.02 of `trace/core`
at request (−0.011) but 0.04 below it at session (p < 0.001) — **H-floor-bigram fails at m on the
session grain**: the learned arms' session advantage over the adjacency count is real but small
(0.04 F1, 0.025 … 0.036 AUROC), and Shapley again falls below the floor. The bigram test F1 lands
+0.041 … +0.046 (request) / +0.049 … +0.060 (session) above its frozen validation value, the
double-sample effect of the sample-size note; its validation reads froze nine of ten cells at
τ = 0 (the grid edge: every observed adjacent pair).

## l (freezes 2026-10-10, CPU-host test reads 2026-10-10; validation 10k / 5k, test 20k / 10k head sample)

### Floor cells, five-seed mean ± std (test read, default floor 0.05)

| cell | directed F1 | precision | recall | AUROC (directed) | AP (directed) | reachable F1 | F1 per seed 0–4 |
|---|---|---|---|---|---|---|---|
| `floor/topology/none/frozen/request` (sample-free) | 0.846 ± 0.004 | 0.733 | 1.000 | 0.999 | 0.753 | 0.846 | 0.845 / 0.843 / 0.850 / 0.843 / 0.849 |
| `floor/bigram/none/frozen/request` | 0.135 ± 0.005 | 0.281 | 0.089 | 0.544 | 0.054 | 0.439 | 0.127 / 0.138 / 0.140 / 0.139 / 0.133 |
| `floor/topology/none/frozen/session` (sample-free) | 0.837 ± 0.004 | 0.733 | 0.975 | 0.987 | 0.735 | 0.846 | 0.836 / 0.834 / 0.840 / 0.833 / 0.841 |
| `floor/bigram/none/frozen/session` | 0.162 ± 0.005 | 0.258 | 0.118 | 0.559 | 0.065 | 0.410 | 0.153 / 0.164 / 0.167 / 0.165 / 0.159 |

Reference arms and baselines at the frozen cut on the same seeds (directed F1, request / session): `trace/core/shipped/frozen` 0.135 / 0.161; `trace/cli/shipped/frozen` 0.135 / 0.155; `trace/cli-atomic/shipped/frozen` 0.133 / 0.148; `baseline/granger/shipped/frozen` 0.134 / 0.161; `baseline/saliency/none/frozen` 0.139 / 0.153; `baseline/shapley/none/frozen` 0.035 / 0.038.

### Registered pairs (`plans/floors.md` §6; floor − arm, paired per seed, five seeds, paired t-test)

| pair | grain | directed F1 | AUROC (directed) | precision | recall |
|---|---|---|---|---|---|
| `floor/topology` − `trace/core/shipped/frozen` | request | +0.711 (p < 0.001) | +0.445 (p < 0.001) | +0.544 (p < 0.001) | +0.894 (p < 0.001) |
| `floor/topology` − `trace/core/shipped/frozen` | session | +0.676 (p < 0.001) | +0.405 (p < 0.001) | +0.512 (p < 0.001) | +0.847 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | request | +0.712 (p < 0.001) | +0.452 (p < 0.001) | +0.531 (p < 0.001) | +0.900 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | session | +0.676 (p < 0.001) | +0.417 (p < 0.001) | +0.514 (p < 0.001) | +0.848 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | request | +0.707 (p < 0.001) | +0.442 (p < 0.001) | +0.504 (p < 0.001) | +0.900 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | session | +0.684 (p < 0.001) | +0.404 (p < 0.001) | +0.568 (p < 0.001) | +0.833 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | request | -0.000 (p 0.973) | -0.010 (p < 0.001) | +0.092 (p < 0.001) | -0.017 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | session | +0.001 (p 0.903) | -0.023 (p < 0.001) | +0.036 (p 0.124) | -0.011 (p 0.008) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | request | +0.002 (p 0.479) | -0.003 (p 0.050) | +0.079 (p < 0.001) | -0.011 (p < 0.001) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | session | +0.001 (p 0.867) | -0.012 (p < 0.001) | +0.039 (p < 0.001) | -0.010 (p 0.062) |
| `floor/bigram` − `baseline/saliency/none/frozen` | request | -0.003 (p 0.225) | -0.013 (p < 0.001) | +0.052 (p < 0.001) | -0.010 (p 0.005) |
| `floor/bigram` − `baseline/saliency/none/frozen` | session | +0.009 (p 0.038) | -0.025 (p < 0.001) | +0.092 (p < 0.001) | -0.024 (p < 0.001) |

**H-floor-topology — PASS.** `floor/topology` request F1 0.846 ± 0.004 against every learned arm on every seed: the smallest five-seed mean paired difference is +0.707 (vs `baseline/saliency/none/frozen`, min per-seed +0.697, p < 0.001); the range over the 13 learned cells is +0.707 … +0.811, every p < 0.001. Sample-free: the topology arm's validation and test reads coincide except for τ, so this is not a held-out number and the pairs are not sample-paired (`plans/floors.md` §5).

**H-floor-bigram — PASS.** `floor/bigram` − `trace/core/shipped/frozen` directed F1: request -0.000 ± 0.004 (p 0.973, within 0.02 or n.s.); session +0.001 ± 0.011 (p 0.903, within 0.02 or n.s.).

**H-floor-auroc — FAIL.** Session-grain directed AUROC, learned arm − `floor/bigram` (paired): `trace/core/shipped/frozen` +0.023 (p < 0.001); `trace/cli/shipped/frozen` +0.023 (p < 0.001); `trace/cli-atomic/shipped/frozen` +0.024 (p < 0.001); `baseline/granger/shipped/frozen` +0.012 (p < 0.001); `baseline/saliency/none/frozen` +0.025 (p < 0.001); `baseline/shapley/none/frozen` -0.044 (p < 0.001). Fails by `baseline/granger/shipped/frozen`, `baseline/granger/fixed/frozen`, `baseline/shapley/none/frozen` (advantage below 0.02).

**Reading.** l is where the learned arms fall to the bigram floor's level. `trace/core` lands at
0.135 request / 0.161 session and the bigram floor at 0.135 / 0.162 (−0.000 / +0.001, p 0.97 /
0.90), so **H-floor-bigram passes again** — not because the floor recovered (its coverage ceiling
on the 20k / 10k test sample is 0.08–0.09 request / 0.11–0.12 session, down from 0.15–0.16 /
0.21–0.23 at m) but because the arms fell with it: every learned cell but Shapley sits at
0.13–0.14 / 0.15–0.16 F1, the corpus effect of `findings/l-latent.md`. The topology floor is
unmoved at 0.846 ± 0.004 request / 0.837 ± 0.004 session (precision 0.733, session recall
0.97–0.98), 0.71 / 0.68 above `trace/core` and 0.71 … 0.81 above every learned cell — its widest
margin so far. **H-floor-auroc fails by two arms at l:** Granger's session-AUROC advantage over the
bigram floor is +0.012 (p < 0.001, but the criterion is the 0.02 magnitude), Shapley is −0.044
below the floor, and the TRACE arms and saliency hold +0.023 … +0.025. All ten bigram cells froze
at τ = 0 (the grid edge); the bigram test F1 lands +0.042 … +0.046 (request) / +0.047 … +0.053
(session) above its frozen validation value, with the coverage ceiling rising 0.050–0.056 →
0.082–0.094 / 0.066–0.076 → 0.110–0.124 between the 10k / 5k and 20k / 10k samples — the
sample-size note, as at s and m.

## xl (freezes 2026-10-10, CPU-host test reads 2026-10-10; validation 10k / 5k, test 20k / 10k head sample)

### Floor cells, five-seed mean ± std (test read, default floor 0.05)

| cell | directed F1 | precision | recall | AUROC (directed) | AP (directed) | reachable F1 | F1 per seed 0–4 |
|---|---|---|---|---|---|---|---|
| `floor/topology/none/frozen/request` (sample-free) | 0.842 ± 0.002 | 0.728 | 1.000 | 0.999 | 0.752 | 0.842 | 0.844 / 0.841 / 0.840 / 0.842 / 0.845 |
| `floor/bigram/none/frozen/request` | 0.079 ± 0.002 | 0.291 | 0.046 | 0.523 | 0.026 | 0.450 | 0.079 / 0.076 / 0.081 / 0.080 / 0.080 |
| `floor/topology/none/frozen/session` (sample-free) | 0.836 ± 0.002 | 0.728 | 0.983 | 0.991 | 0.740 | 0.842 | 0.838 / 0.835 / 0.833 / 0.836 / 0.839 |
| `floor/bigram/none/frozen/session` | 0.095 ± 0.002 | 0.274 | 0.057 | 0.529 | 0.030 | 0.431 | 0.095 / 0.091 / 0.097 / 0.097 / 0.095 |

Reference arms and baselines at the frozen cut on the same seeds (directed F1, request / session): `trace/core/shipped/frozen` 0.088 / 0.097; `trace/cli/shipped/frozen` 0.088 / 0.093; `trace/cli-atomic/shipped/frozen` 0.089 / 0.088; `baseline/granger/shipped/frozen` 0.083 / 0.095; `baseline/saliency/none/frozen` 0.089 / 0.095; `baseline/shapley/none/frozen` 0.022 / 0.023.

### Registered pairs (`plans/floors.md` §6; floor − arm, paired per seed, five seeds, paired t-test)

| pair | grain | directed F1 | AUROC (directed) | precision | recall |
|---|---|---|---|---|---|
| `floor/topology` − `trace/core/shipped/frozen` | request | +0.754 (p < 0.001) | +0.470 (p < 0.001) | +0.555 (p < 0.001) | +0.941 (p < 0.001) |
| `floor/topology` − `trace/core/shipped/frozen` | session | +0.739 (p < 0.001) | +0.448 (p < 0.001) | +0.590 (p < 0.001) | +0.907 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | request | +0.760 (p < 0.001) | +0.477 (p < 0.001) | +0.533 (p < 0.001) | +0.948 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | session | +0.741 (p < 0.001) | +0.458 (p < 0.001) | +0.563 (p < 0.001) | +0.916 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | request | +0.753 (p < 0.001) | +0.468 (p < 0.001) | +0.575 (p < 0.001) | +0.937 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | session | +0.742 (p < 0.001) | +0.447 (p < 0.001) | +0.582 (p < 0.001) | +0.909 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | request | -0.009 (p < 0.001) | -0.007 (p < 0.001) | +0.118 (p < 0.001) | -0.014 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | session | -0.002 (p 0.522) | -0.015 (p < 0.001) | +0.137 (p < 0.001) | -0.019 (p < 0.001) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | request | -0.004 (p 0.016) | -0.000 (p 0.961) | +0.097 (p < 0.001) | -0.007 (p < 0.001) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | session | -0.000 (p 0.969) | -0.005 (p 0.003) | +0.110 (p < 0.001) | -0.010 (p 0.001) |
| `floor/bigram` − `baseline/saliency/none/frozen` | request | -0.010 (p < 0.001) | -0.008 (p < 0.001) | +0.138 (p < 0.001) | -0.017 (p < 0.001) |
| `floor/bigram` − `baseline/saliency/none/frozen` | session | +0.000 (p 0.952) | -0.016 (p < 0.001) | +0.129 (p 0.006) | -0.017 (p < 0.001) |

**H-floor-topology — PASS.** `floor/topology` request F1 0.842 ± 0.002 against every learned arm on every seed: the smallest five-seed mean paired difference is +0.753 (vs `baseline/saliency/none/frozen`, min per-seed +0.749, p < 0.001); the range over the 13 learned cells is +0.753 … +0.821, every p < 0.001. Sample-free: the topology arm's validation and test reads coincide except for τ, so this is not a held-out number and the pairs are not sample-paired (`plans/floors.md` §5).

**H-floor-bigram — PASS.** `floor/bigram` − `trace/core/shipped/frozen` directed F1: request -0.009 ± 0.001 (p < 0.001, within 0.02 or n.s.); session -0.002 ± 0.007 (p 0.522, within 0.02 or n.s.).

**H-floor-auroc — FAIL.** Session-grain directed AUROC, learned arm − `floor/bigram` (paired): `trace/core/shipped/frozen` +0.015 (p < 0.001); `trace/cli/shipped/frozen` +0.015 (p < 0.001); `trace/cli-atomic/shipped/frozen` +0.015 (p < 0.001); `baseline/granger/shipped/frozen` +0.005 (p 0.003); `baseline/saliency/none/frozen` +0.016 (p < 0.001); `baseline/shapley/none/frozen` -0.022 (p < 0.001). Fails by `trace/core/shipped/frozen`, `trace/core/fixed-kl/frozen`, `trace/core/fixed/frozen`, `trace/cli/shipped/frozen`, `trace/cli/fixed-kl/frozen`, `trace/cli/fixed/frozen`, `trace/cli-atomic/shipped/frozen`, `trace/cli-atomic/fixed-kl/frozen`, `trace/cli-atomic/fixed/frozen`, `baseline/granger/shipped/frozen`, `baseline/granger/fixed/frozen`, `baseline/saliency/none/frozen`, `baseline/shapley/none/frozen` (advantage below 0.02).

**Reading.** xl repeats l with the margins stretched further. The learned arms sit at 0.083–0.089
request / 0.088–0.097 session F1 (Shapley 0.022 / 0.023) and the bigram floor at 0.079 / 0.095:
−0.009 at request (p < 0.001, within the 0.02 band) and −0.002 at session (p 0.52), so
**H-floor-bigram passes** — on a test sample that reaches only 0.044–0.047 / 0.055–0.059 of the
xl truth, the adjacency count and the learned arms are the same cut of the same co-occurrences
(every request arm is within 0.003 of the every-observed-pair reference line,
`findings/xl-latent.md`). The topology floor holds at 0.842 ± 0.002 request / 0.836 ± 0.002
session (precision 0.724–0.731, session recall 0.983–0.984), 0.75 / 0.74 above `trace/core` and
0.75 … 0.82 above every learned cell. **H-floor-auroc fails by every learned arm at xl:** the
session-AUROC advantage over the bigram floor is +0.015 … +0.016 for the TRACE arms and saliency,
+0.005 for Granger, and Shapley is −0.022 below the floor — none reaches the 0.02 the hypothesis
asks for (every p < 0.01; the magnitudes do not). All ten bigram cells froze at τ = 0 (the grid
edge); the bigram test F1 lands +0.031 … +0.032 (request) / +0.036 … +0.039 (session) above its
frozen validation value, with the coverage ceiling rising 0.024–0.027 → 0.044–0.047 /
0.030–0.033 → 0.055–0.059 between the 10k / 5k and 20k / 10k samples. Across the ladder the
picture is now closed: the topology floor's request F1 is 0.82 / 0.84 / 0.85 / 0.85 / 0.84 from
xs to xl while the best learned cell falls 0.32 → 0.09, and the bigram floor ties `trace/core` on
four rungs of five, losing only at m on the session grain; the learned arms' session-AUROC margin
over the adjacency count shrinks from 0.03–0.04 (xs–m) to 0.02 (l) to under 0.02 (xl).

## The floors on the per-sequence axis — xs … xl (registered 2026-10-21; xs, s, m read 2026-10-21, l and xl 2026-10-22)

Registration: `plans/floors.md` and `plans/per-sequence-rules.md`, addenda of 2026-10-21 (committed
before any run), RUN.md D-SB-17. Instrument: `seq2causebench.floorseq`, model-free; tables
`tables/<rung>-latent-perseq-floors/` (their own documents, never a type-level table). Each floor
row is the floor's frozen-τ type-level score **broadcast** onto the sequences, window and candidate
pairs of a landed arm test read — `trace/core/shipped/frozen` (headline) and
`trace/cli-atomic/shipped/frozen` — so floor row and arm row are paired sequence by sequence: on
every one of the 300 runs (5 rungs × 5 seeds × 2 grains × 2 reads × 3 rows) the candidate, truth
and scoreable counts and the predict-all value equal the arm's landed `seqscore.json` exactly
(asserted by the instrument), the floor's recomputed score table equals its landed test read's
`n_pairs` / `n_edges`, and at m the re-derived sequence lists equal the reads' own sequence records
(20 of 20). The bigram floor also gets an **adjacent-only** row (`q = j + 1`). τ = each floor's
frozen type-level τ (`freezes/2026-10-10-<rung>-latent-s<k>-floors.json`); no per-sequence
selection. xs, s and m ran locally; l and xl on CPU hosts (one owner-applied replica per rung and
seed, box at `fede4410`; peak resident equal to the arms' own `seqscore` to 0.1 GiB — 14.9–17.4 GiB
at l session, 69.6–80.1 GiB at xl session).

Caveats beside every number here: the topology floor is sample-free (its score table is the
deployment prior; only the sequences it is broadcast onto are sampled); at the request grain the
per-sequence truth is the call link restricted to target edges, which the topology names by
construction — its row is a statement about the benchmark, not a bar; the session cross-request
rule is an approximation over occurrence pairs (`plans/per-sequence-rules.md` §2); five pairs give
low power, so the effect size (mean ± sd of the paired difference) is the primary readout.

### Pooled per-sequence directed F1, five-seed mean ± sd, beside `trace/core` (paired difference floor − arm, paired t-test)

| rung · grain | `trace/core` | predict-all | `floor/topology` broadcast | `floor/bigram` broadcast | `floor/bigram` adjacent |
|---|---|---|---|---|---|
| xs · request | 0.423 ± 0.061 | 0.456 | 0.800 ± 0.079 (+0.377, p < 0.001) | 0.481 ± 0.065 (+0.058, p 0.29) | 0.496 ± 0.080 (+0.073, p 0.28) |
| s · request | 0.343 ± 0.060 | 0.397 | 0.833 ± 0.051 (+0.490, p < 0.001) | 0.380 ± 0.084 (+0.037, p 0.10) | 0.377 ± 0.074 (+0.033, p 0.010) |
| m · request | 0.444 ± 0.015 | 0.409 | 0.867 ± 0.019 (+0.423, p < 0.001) | 0.453 ± 0.026 (+0.009, p 0.33) | 0.392 ± 0.020 (−0.053, p < 0.001) |
| l · request | 0.429 ± 0.018 | 0.401 | 0.850 ± 0.013 (+0.422, p < 0.001) | 0.465 ± 0.028 (+0.036, p 0.014) | 0.392 ± 0.023 (−0.037, p 0.008) |
| xl · request | 0.380 ± 0.013 | 0.367 | 0.800 ± 0.025 (+0.420, p < 0.001) | 0.417 ± 0.015 (+0.037, p < 0.001) | 0.355 ± 0.005 (−0.025, p 0.028) |
| xs · session | 0.433 ± 0.049 | 0.303 | 0.406 ± 0.037 (−0.028, p 0.40) | 0.382 ± 0.042 (−0.052, p 0.051) | 0.414 ± 0.066 (−0.020, p 0.35) |
| s · session | 0.346 ± 0.063 | 0.173 | 0.493 ± 0.062 (+0.146, p 0.003) | 0.271 ± 0.063 (−0.075, p 0.003) | 0.323 ± 0.067 (−0.023, p 0.12) |
| m · session | 0.412 ± 0.023 | 0.151 | 0.589 ± 0.025 (+0.177, p 0.001) | 0.299 ± 0.029 (−0.113, p < 0.001) | 0.335 ± 0.015 (−0.077, p < 0.001) |
| l · session | 0.383 ± 0.055 | 0.143 | 0.576 ± 0.012 (+0.193, p 0.001) | 0.320 ± 0.025 (−0.063, p 0.032) | 0.343 ± 0.022 (−0.040, p 0.12) |
| xl · session | 0.250 ± 0.042 | 0.132 | 0.572 ± 0.015 (+0.322, p < 0.001) | 0.309 ± 0.009 (+0.059, p 0.028) | 0.317 ± 0.004 (+0.067, p 0.024) |

### The same, beside `trace/cli-atomic` (its own read: same sequences, its own `c`)

| rung · grain | `trace/cli-atomic` | predict-all | `floor/topology` broadcast | `floor/bigram` broadcast | `floor/bigram` adjacent |
|---|---|---|---|---|---|
| xs · request | 0.505 ± 0.052 | 0.500 | 0.807 ± 0.083 (+0.302, p < 0.001) | 0.529 ± 0.057 (+0.024, p 0.55) | 0.538 ± 0.067 (+0.033, p 0.54) |
| s · request | 0.488 ± 0.058 | 0.430 | 0.838 ± 0.043 (+0.350, p < 0.001) | 0.421 ± 0.073 (−0.067, p 0.034) | 0.409 ± 0.090 (−0.079, p 0.042) |
| m · request | 0.491 ± 0.013 | 0.409 | 0.867 ± 0.019 (+0.376, p < 0.001) | 0.453 ± 0.026 (−0.038, p 0.022) | 0.392 ± 0.020 (−0.099, p 0.001) |
| l · request | 0.438 ± 0.030 | 0.409 | 0.852 ± 0.014 (+0.413, p < 0.001) | 0.473 ± 0.023 (+0.035, p 0.035) | 0.398 ± 0.016 (−0.041, p 0.021) |
| xl · request | 0.369 ± 0.014 | 0.367 | 0.800 ± 0.025 (+0.431, p < 0.001) | 0.417 ± 0.015 (+0.048, p 0.001) | 0.355 ± 0.005 (−0.014, p 0.14) |
| xs · session | 0.443 ± 0.047 | 0.304 | 0.426 ± 0.052 (−0.016, p 0.52) | 0.373 ± 0.019 (−0.070, p 0.016) | 0.414 ± 0.067 (−0.028, p 0.13) |
| s · session | 0.423 ± 0.057 | 0.174 | 0.494 ± 0.063 (+0.070, p < 0.001) | 0.269 ± 0.063 (−0.154, p < 0.001) | 0.323 ± 0.067 (−0.101, p 0.001) |
| m · session | 0.470 ± 0.012 | 0.151 | 0.589 ± 0.025 (+0.119, p < 0.001) | 0.299 ± 0.029 (−0.172, p < 0.001) | 0.335 ± 0.015 (−0.136, p < 0.001) |
| l · session | 0.385 ± 0.061 | 0.143 | 0.576 ± 0.012 (+0.191, p 0.002) | 0.320 ± 0.025 (−0.065, p 0.036) | 0.343 ± 0.022 (−0.041, p 0.14) |
| xl · session | 0.230 ± 0.060 | 0.132 | 0.572 ± 0.015 (+0.342, p < 0.001) | 0.309 ± 0.009 (+0.079, p 0.034) | 0.317 ± 0.004 (+0.087, p 0.031) |

(At m and xl both reads use `c = 1` at request and `c = 2` at session on the same sequences, so the
floor rows coincide across the two reads; at xs, s and l the reads' `c` differ on some seeds — at l
only seed 2 at request, where `trace/cli-atomic` reads `c = 2`.)

### Hypotheses (`plans/floors.md` addendum 2026-10-21 §7)

- **H-perseq-floor-topology — passes on every rung.** At request the broadcast topology floor is
  above `trace/core` on every rung (+0.377 / +0.490 / +0.423 / +0.422 / +0.420 xs → xl, p < 0.001),
  recall 1.00 at xs, l and xl; as stated in advance, this is the call tree recovered by construction.
- **H-perseq-floor-bigram — fails on s, m, l and xl; its direction turns over the ladder.** Paired
  differences of the bigram broadcast row against `trace/core`, xs → xl: request +0.058 (p 0.29) /
  +0.037 (p 0.10) / +0.009 (p 0.33) — within the rule through m — then **+0.036 (p 0.014)** /
  **+0.037 (p < 0.001)** at l / xl, in the **floor's** favour on 5/5 seeds each; session −0.052
  (p 0.051) / **−0.075 (p 0.003)** / **−0.113 (p < 0.001)** / **−0.063 (p 0.032)** — in the arm's
  favour from s through l, widest at m — then **+0.059 (p 0.028)** at xl, in the **floor's** favour
  (5/5 seeds positive, seed 2 by +0.000). `trace/cli-atomic` separates by more where it separates
  (−0.154 / −0.172 / −0.065 at s / m / l session) and trails by more at xl (+0.079, p 0.034). The
  adjacent-only row turns with it: session −0.023 / −0.077 / −0.040 (p 0.12) at s / m / l,
  **+0.067 (p 0.024)** at xl.
- **H-perseq-floor-predict-all — reported per row.** The topology rows are above the predict-all
  value on 5/5 seeds on every rung and grain. The bigram rows clear it on 5/5 seeds at the session
  grain everywhere, but not reliably at request: broadcast 4/5 (xs), 2/5 (s), 5/5 (m, l, xl);
  adjacent 2/5, 1/5, 0/5, 1/5, 2/5 — at request the order-count floor sits at the collapsed-threshold
  level, as the arms do (`plans/per-sequence-rules.md` addendum 2026-09-26).

### Why the session grain turns at xl: the arm's frozen cut, not its ranking

The arm row is cut at the arm's frozen **type-level** τ (no per-sequence selection; the registered
rule, not re-opened here), and that τ falls up the ladder: `trace/core`'s session τ is 3e-2 on every
m seed, 3e-4 … 3e-3 at l and 1e-5 … 3e-4 at xl, so its pooled per-sequence directed precision falls
from 0.31–0.36 (m) to 0.18–0.30 (l) and 0.11–0.19 (xl) while recall rises to 0.85–0.91. Seed by
seed the floor's margin follows that cut: at xl +0.107 / +0.078 / +0.063 / +0.048 / +0.000 at τ
1e-5 / 3e-5 / 1e-4 / 1e-4 / 3e-4 (seeds 1, 4, 0, 3, 2); at l the two lowest cuts (seed 2, 3e-4;
seed 0, 1e-3) are the two seeds where the floor comes within 0.03 (−0.008, −0.027), and at 3e-3 it
trails by 0.08–0.11. The ranking does not turn: `trace/core`'s pooled directed AUROC stays above the
bigram broadcast row at session on l and xl (by 0.038, p 0.027; by 0.052, p 0.004) and far above the
adjacent-only row (by 0.19 / 0.18); the topology floor's session AUROC is 0.025 below the arm's at l
(p 0.034) and ties it at xl (+0.001, p 0.93).

### What this adds to the type-level reading

On the type-level axis the bigram floor ties `trace/core` on four rungs of five (H-floor-bigram
above). On the per-sequence axis, the axis TRACE claims, it ties at request through m and **beats**
`trace/core` at request on l and xl (+0.036 / +0.037), where the arm sits 0.01–0.03 above the
predict-all value; at session it **loses from s through l** (−0.075 / −0.113 / −0.063) and **wins at
xl** (+0.059), and the adjacent-only row — immediate succession at the position level — turns the
same way (−0.077 at m, +0.067 at xl). The topology floor remains above every learned arm wherever the
truth is the call tree (request on every rung; session from s up, +0.146 / +0.177 / +0.193 /
+0.322), but **not at xs session** (−0.028, p 0.40: `trace/core` ties it). The session grain is
therefore where this axis discriminates **from s through l** (the reading of 2026-10-21, which had
xs … m); at xl the arm's frozen-cut F1 falls below the order-count floor's while its ranking stays
above it, so at xl the per-sequence advantage is carried by the ranking (AUROC), not by the
registered frozen-cut F1 headline.

## The start-ordered view — the floors on xs and s, and the view-order statistic (registered 2026-10-09 / 2026-10-22; read 2026-10-23)

Request grain only (`plans/floors.md`, addendum 2026-10-09: the session grain carries no clock
artefact). The floors' chain of §4 with `--ordering start`, locally: validation reads at `5b627901`
(xs the whole split, s 10,000 sequences), frozen in `freezes/2026-10-23-<rung>-latent-s<k>-floors-start.json`
(`freeze@3`, `family: floor`, `ordering: start`), committed (`6c9fe9c0`), then the test reads (xs the
whole split, s 20,000 sequences) and `annotate` at that commit. Staging: xs the smallest rung; s the
arms' xs seed-k **end-view** request entries, the floors' own rule (addendum 2026-10-22), so these
reads do not wait on the arms' start-view chain. Records: `results-start/<rung>/latent/seed=<k>/`
(`floors-val-sweep/`, `floors-freeze/`, `test-read/floors-request/`,
`floor/<arm>/none/frozen/request/`); ledger stage `floors`, `ordering: "start"`.

### Floor cells, five-seed mean ± std (test read, request grain, default floor 0.05), start view beside the end view

| rung | cell | directed F1 | precision | recall | directed AUROC | orientation accuracy | end-view F1 / AUROC / orientation |
|---|---|---|---|---|---|---|---|
| xs | `floor/topology` (sample-free) | 0.820 ± 0.072 | 0.699 ± 0.098 | 1.000 | 0.972 ± 0.007 | 1.000 | 0.820 / 0.972 / 1.000 |
| xs | `floor/bigram` | 0.114 ± 0.037 | 0.110 ± 0.033 | 0.120 ± 0.043 | 0.504 ± 0.017 | 0.051 ± 0.024 | 0.340 / 0.632 / 0.796 |
| s | `floor/topology` (sample-free) | 0.839 ± 0.014 | 0.723 ± 0.020 | 1.000 | 0.986 ± 0.002 | 1.000 | 0.839 / 0.986 / 1.000 |
| s | `floor/bigram` | 0.084 ± 0.010 | 0.072 ± 0.009 | 0.101 ± 0.014 | 0.507 ± 0.008 | 0.038 ± 0.010 | 0.306 / 0.645 / 0.830 |

- **Topology** reads no view: its start-view cell equals its end-view cell on every seed (τ, F1 and
  AUROC identical), as registered; it is run and recorded for uniformity and labelled so.
- **Bigram** loses two thirds of its F1 (xs 0.340 → 0.114, s 0.306 → 0.084), its directed AUROC
  falls to chance (0.50–0.51) and its orientation accuracy to 0.04–0.05: on the start view the token
  that follows `a` is, for most truth co-occurrences, its cause, so "b immediately follows a" points
  the wrong way. Its reachable-recall ceiling falls with it (xs 0.339 → 0.123, s 0.366 → 0.101); the
  frozen τ stays at the grid's bottom (0 or 0.005).
- **Pairs and hypotheses** (`plans/floors.md` §6, §7) on the start view: xs below, beside the arms'
  start-view test reads (landed 2026-10-23); s when its start-view Job T lands.

### The view-order statistic (`tables/vieworder-latent/`; test split, the arms' test-read sample)

Share of co-occurrences of a directed truth edge in which the cause token comes first in the view's
row order, five-seed mean [min, max], pooled over truth edges and sequences:

| rung | request, end | request, start | session, end | session, start |
|---|---|---|---|---|
| xs | 0.891 [0.842, 0.927] | 0.102 [0.029, 0.164] | 0.513 [0.500, 0.528] | 0.507 [0.490, 0.544] |
| s | 0.858 [0.753, 0.922] | 0.049 [0.032, 0.071] | 0.505 [0.484, 0.518] | 0.491 [0.483, 0.498] |
| m | 0.947 [0.924, 0.979] | 0.038 [0.032, 0.040] | 0.560 [0.557, 0.563] | 0.455 [0.444, 0.467] |

l and xl are named "not run" in the table: their score tiers were not on disk for these reads. The
statistic describes the view, not an arm; it is reported beside every request-grain orientation
number of both views and gates nothing. At the request grain the end view puts the cause first in
86–95 % of truth co-occurrences and the start view in 4–10 %; the session grain sits near one half on
both views (0.46–0.56), the clock-free grain the 2026-10-09 addendum names. The 2026-10-03 probe
(0.90–0.92 end, 0.05–0.09 start at the request grain; ≈ 0.50 at the session grain) is provenance only;
its counting rule was not written down, and this table supersedes it.

### The arms on the start view — xs (test reads 2026-10-23, five seeds, request grain)

The five xs start-view Job T runs (`freezes/2026-10-22-xs-latent-s<k>-start.json`; one backbone
per seed trained on `start-session`; owner-applied, box at `1d3c4957`), landed in
`results-start/xs/latent/seed=<k>/`; tables `tables/xs-latent-start/` (`report` on `results-start/`,
the twelve xs pairs, every session cell named "not run" through `tables/reasons-start-view.json`).
A start-view cell shares corpus, seed, grids and test sample with its end-view cell
(`tables/xs-latent/`) and differs in the view and the backbone, so start − end below is paired per
seed on the corpus and the sample, not on a model. Five-seed mean ± sd; paired t-test on five
differences. The view-order statistic beside the orientation columns: cause first in 0.102
[0.029, 0.164] of request-grain truth co-occurrences on the start view, 0.891 [0.842, 0.927] on the
end view.

| cell (frozen cut) | directed F1, start | end | start − end (paired) | directed AUROC, start / end | orientation accuracy, start / end | reachable-recall ceiling, start / end |
|---|---|---|---|---|---|---|
| `trace/core`, shipped path | 0.130 ± 0.042 | 0.321 ± 0.043 | -0.192 (p < 0.001; per seed -0.176 … -0.217) | 0.505 / 0.646 | 0.160 / 0.864 | 0.160 / 0.480 |
| `trace/core`, fixed-kl | 0.130 ± 0.042 | 0.321 ± 0.043 | -0.192 (p < 0.001; per seed -0.176 … -0.217) | 0.505 / 0.646 | 0.160 / 0.864 | 0.160 / 0.480 |
| `trace/core`, fixed | 0.129 ± 0.043 | 0.317 ± 0.050 | -0.188 (p < 0.001; per seed -0.165 … -0.213) | 0.507 / 0.646 | 0.169 / 0.852 | 0.160 / 0.475 |
| `trace/cli`, shipped path | 0.128 ± 0.037 | 0.317 ± 0.045 | -0.190 (p < 0.001; per seed -0.165 … -0.223) | 0.505 / 0.646 | 0.158 / 0.854 | 0.160 / 0.480 |
| `trace/cli`, fixed-kl | 0.128 ± 0.037 | 0.317 ± 0.045 | -0.190 (p < 0.001; per seed -0.165 … -0.223) | 0.505 / 0.646 | 0.158 / 0.854 | 0.160 / 0.480 |
| `trace/cli`, fixed | 0.125 ± 0.039 | 0.314 ± 0.043 | -0.189 (p < 0.001; per seed -0.165 … -0.214) | 0.506 / 0.646 | 0.179 / 0.850 | 0.160 / 0.480 |
| `trace/cli-atomic`, shipped path | 0.122 ± 0.034 | 0.307 ± 0.043 | -0.185 (p < 0.001; per seed -0.156 … -0.222) | 0.495 / 0.644 | 0.121 / 0.872 | 0.160 / 0.425 |
| `trace/cli-atomic`, fixed-kl | 0.122 ± 0.034 | 0.307 ± 0.043 | -0.185 (p < 0.001; per seed -0.156 … -0.222) | 0.495 / 0.645 | 0.121 / 0.872 | 0.160 / 0.425 |
| `trace/cli-atomic`, fixed | 0.120 ± 0.036 | 0.312 ± 0.045 | -0.192 (p < 0.001; per seed -0.162 … -0.230) | 0.495 / 0.645 | 0.109 / 0.886 | 0.160 / 0.425 |
| `baseline/granger`, shipped | 0.116 ± 0.038 | 0.321 ± 0.041 | -0.204 (p < 0.001; per seed -0.188 … -0.236) | 0.503 / 0.632 | 0.175 / 0.908 | 0.160 / 0.490 |
| `baseline/granger`, fixed | 0.115 ± 0.036 | 0.318 ± 0.047 | -0.203 (p < 0.001; per seed -0.179 … -0.233) | 0.504 / 0.630 | 0.195 / 0.914 | 0.160 / 0.484 |
| `baseline/saliency` | 0.115 ± 0.055 | 0.279 ± 0.040 | -0.164 (p < 0.001; per seed -0.133 … -0.196) | 0.493 / 0.652 | 0.162 / 0.833 | 0.160 / 0.471 |
| `baseline/shapley` | 0.097 ± 0.031 | 0.255 ± 0.050 | -0.158 (p < 0.001; per seed -0.117 … -0.214) | 0.492 / 0.620 | 0.195 / 0.798 | 0.160 / 0.492 |

- **Every frozen cell loses 0.16–0.20 directed F1 on every seed** (per-seed differences −0.117 …
  −0.236, p < 0.001 on each of the 13 cells) and lands at 0.10–0.13. Directed AUROC falls to chance
  (0.49–0.51, from 0.62–0.65).
- **Orientation accuracy follows the view.** 0.11–0.20 on the start view against 0.80–0.91 on the
  end view, beside a view-order share of 0.10 and 0.89: on both views the arms' request-grain
  orientation is the row order of the view they read.
- **The reachable-recall ceiling is the mechanism.** It falls to 0.160 ± 0.062 for every arm (end view
  0.43–0.49). Every arm scores a pair only as earlier → later, and on the start view a truth edge's
  cause precedes its effect in the sample for about one truth edge in six. Restricted to the edges
  it can reach, each arm stays above "predict every reachable pair" (reachable F1 0.18–0.22 against
  0.16–0.17; Shapley 0.177 against 0.171; on the end view 0.33–0.42 against 0.36–0.39, Shapley
  below it).
- **Shipped cut.** Already near zero on the end view, it does not move: `trace/cli` 0.031 / 0.034 on
  the start view against 0.033 / 0.038 (p 0.84 / 0.62); `trace/cli-atomic` 0.026 / 0.027 against
  0.048 / 0.046 (p 0.25 / 0.29).

### Registered pairs on the start view — xs (`plans/floors.md` §6; floor − arm, paired per seed, five seeds, paired t-test; request grain)

| pair | directed F1 | AUROC (directed) | precision | recall | orientation accuracy |
|---|---|---|---|---|---|
| `floor/topology` − `trace/core/shipped/frozen` | +0.690 (p < 0.001) | +0.467 (p < 0.001) | +0.575 (p < 0.001) | +0.862 (p < 0.001) | +0.840 (p < 0.001) |
| `floor/topology` − `baseline/granger/shipped/frozen` | +0.703 (p < 0.001) | +0.469 (p < 0.001) | +0.571 (p < 0.001) | +0.887 (p < 0.001) | +0.825 (p < 0.001) |
| `floor/topology` − `baseline/saliency/none/frozen` | +0.705 (p < 0.001) | +0.479 (p < 0.001) | +0.577 (p < 0.001) | +0.891 (p < 0.001) | +0.838 (p < 0.001) |
| `floor/bigram` − `trace/core/shipped/frozen` | -0.015 (p 0.020) | -0.001 (p 0.711) | -0.014 (p 0.054) | -0.018 (p 0.064) | -0.109 (p 0.013) |
| `floor/bigram` − `baseline/granger/shipped/frozen` | -0.002 (p 0.700) | +0.000 (p 0.960) | -0.018 (p 0.260) | +0.007 (p 0.482) | -0.124 (p 0.019) |
| `floor/bigram` − `baseline/saliency/none/frozen` | -0.000 (p 0.976) | +0.010 (p 0.102) | -0.012 (p 0.285) | +0.011 (p 0.193) | -0.111 (p 0.034) |

**H-floor-topology (start view, xs) — PASS.** `floor/topology` (sample-free, the same cell as on
the end view) is above every frozen learned cell on every seed: the smallest five-seed mean
difference is +0.690 (vs `trace/core`, shipped and fixed-kl paths; min per seed +0.619, p < 0.001),
the largest +0.722 (vs Shapley). The margin widens from +0.498 … +0.565 on the end view because the
arms fall and the floor does not.

**H-floor-bigram (start view, xs, request half) — PASS.** `floor/bigram` − `trace/core/shipped/frozen`
directed F1 −0.015 ± 0.009 (p 0.020; per seed −0.006 … −0.027): within 0.02, and in the arm's
favour. The session half is not read on the start view.

**H-floor-auroc** is a session-grain hypothesis and is not read on the start view.

**Reading.** At the request grain the end view hands every forward-only scorer the call tree's
direction (cause first in 0.89 of truth co-occurrences); the start view reverses it (0.10). With it
reversed, the learned arms and the bigram floor land together: directed F1 0.10–0.13, AUROC at
chance, orientation at the view's share. `trace/core` keeps a small F1 margin over the bigram floor
(+0.015, p 0.020; Granger and saliency tie it), and the bigram floor keeps a higher skeleton F1
(+0.12, p 0.006). The topology floor reads no order and is unaffected. So the end-view request-grain
results of the arms, orientation included, depend on the view's clock; the session grain (view
order ≈ 0.5 on both views) is where an arm's direction is its own.

### The per-sequence axis on the start view (`plans/per-sequence-rules.md`, addendum 2026-10-23)

The boxes' 95 `seqscore` records re-read the end view (the defect of that addendum) and are not
landed. The landed records are a local re-score at `9d2cf596` from the landed discover reads with
the boxes' arguments; each names `ordering: start`. The request rule's directed truth (§2: the span
at the earlier position is a direct child of the span at the later one) holds **0 or 1 pair per seed
on every read** (seeds 0–4: 1, 0, 1, 0, 0), because the start view puts every caller before its
callees. The directed per-sequence columns are therefore empty by construction and are not
compared with the end view; `tables/xs-latent-start/xs-latent-perseq.md` carries them with their
truth counts. The skeleton columns score the bidirected (confounded) adjacencies almost alone:
skeleton F1 0.04–0.49, skeleton AUROC 0.43–0.55. The floors' per-sequence read is not run on the
start view.
