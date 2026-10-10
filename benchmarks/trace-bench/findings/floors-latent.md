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
on 2026-10-10; l and xl follow on CPU hosts (owner-applied replicas).

## Caveats carried beside every number

- **Sample-free (topology).** `floor/topology` reads no sequence: its validation and test reads
  are the same computation and differ only in the τ the freeze fixes, so its "test" number is
  not a held-out number and is not sample-paired with any arm (`plans/floors.md` §5). Every
  topology number below carries the label.
- **What the topology τ selects.** The sweep over `floor_taus` chose, on every seed of every rung
  so far, the largest grid value below the smallest `p_call` among the prior's edges (xs: 0.07 …
  0.15), so the frozen cut keeps every prior edge: the frozen cell is the τ = 0 cell, the
  2026-09-28 probe's construction. Recall 1.0 at request is the prior's coverage of the
  request-grain truth, precision the share of its ordered token pairs that are truth edges.
- **Grid edge (bigram).** A bigram cell frozen at τ = 0 is the cut "every observed adjacent
  pair" (`plans/floors.md` §3; recorded, not refused). xs: seeds 0, 1 and 2 at request; s: seeds
  0, 2 and 4 at request; m: nine of ten cells.
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
