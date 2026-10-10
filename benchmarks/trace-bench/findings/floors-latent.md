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
`score.py` byte-identical to the `8e91aa2` pin the arms landed on). xs and s were read locally
on 2026-10-10; m, l and xl follow (l and xl on CPU hosts, owner-applied).

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
  pair" (`plans/floors.md` §3; recorded, not refused). xs: seeds 0, 1 and 2 at request.
- **Sample size (from s).** The bigram floor is sample-bound like the arms: the validation read
  probes the arms' 10k / 5k head sample and the test read 20k / 10k (`plans/caps.md`, addendum
  2026-09-27), so from s its test F1 moves against its frozen validation value as a coverage
  effect, not a generalisation check; xs reads the whole split both times.
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
