# Findings — THP with the topology (`thp/topology`), xs and s / latent, five seeds

The hypotheses of `plans/thp.md` §7 are scored here per rung on the five-seed mean of the paired
per-seed difference (same corpus and seed; `report`'s paired t-test, five pairs, low power — the
effect size is the primary readout). Tables: `tables/<rung>-latent-prior-arms/` (`report` over
`results/` with the pairs of `plans/thp.md` §6 and `plans/lmt.md` §6). Source records:
`results/<rung>/latent/seed={0..4}/thp/topology/none/frozen/<grain>/` (annotate's `score.json`,
`score-ranking.json`, `annotate.json`), `test-read/thp-topology-<grain>/`, `thp-val-sweep/`,
`thp-freeze/`, under `freezes/2026-11-04-<rung>-latent-s<k>-external-thp.json` (`freeze@3`,
`family: external`, `method: thp`, model `none`). Method: gCastle's TTPM at `8675af0b` (the pin
is in every fit's `method_version` and in the install's `direct_url.json`; the record's
`method_commit` field holds the harness commit because the pip install sits inside the harness
tree — recorded, not altered mid-rung) with the lab's decomposable-score search patch (D-TH-7,
output-identical to the shipped search on four xs sweeps and a THPSimulation run to convergence)
and the fixed arguments `max_hop=1 penalty=BIC max_iter=5000 epsilon=1`; node = op, the topology
is the hard prior (`mask_mode: refuse`, `priori_knowledge`); the decay (ms) is swept in the `N`
slot. Reads: EC2 replicas (`RUN.md` registry, 2026-11-04), validation on the whole xs split and
the 10k / 5k s sample, test on the whole xs split and the 20k / 10k s sample (head), as every arm.

## Caveats carried beside every number

- **Hard output, τ tie.** TTPM returns a graph, not a score: every pair it returns scores 1.0 and
  the 20 τ rows of each setting are identical (1 distinct F1). The freeze's tie breaks to the
  largest τ, 0.99, flagged `at_grid_edge` on 20 / 20 cells; the flag is the tie, not a selection.
- **Grid edge in the setting slot.** The frozen decay is 100 ms (the BOTTOM of the registered grid
  {100, 1 000, 10 000}) on 10 / 10 corpora at both grains (not flagged: `freeze` flags τ only).
  Shorter decays are untested; extending the grid is a dated-addendum decision (`plans/thp.md` §3).
- **Few edges.** The BIC keeps 82–154 edges at xs and 602–1 156 at s against 241–748 / 2 064–3 475
  truth edges; the reachable recall ceiling is 0.20–0.26 / 0.11–0.15 at xs validation and 0.16–0.18
  at s. Precision is high (0.75–0.85). The reachable-only F1 (0.87–0.88) equals the predict-all
  F1 on every cell: the arm scores exactly the pairs it returns, so the reachable columns say
  nothing beyond precision.
- **Sample-size note (s).** The test session F1 (0.36–0.40) is ABOVE its validation F1 (0.25–0.28):
  the 2× test sample lets the BIC keep more edges (882–1 156 vs 570–670). Validation and test are
  not the same number (the registered note).
- **Cost.** A test fit at s costs ≈ 3× the validation fit for 2× the sample (629–1 269 s request,
  1 559–3 327 s session on one r7i.2xlarge): the search grows faster than linearly in events.

## Headline (directed F1 at the frozen cut, default floor 0.05, five-seed mean ± std; test reads)

| rung | grain | `thp/topology` | `floor/topology` (sample-free) | `floor/bigram` | `trace/core/shipped/frozen` | `baseline/granger/shipped/frozen` |
|---|---|---|---|---|---|---|
| xs | request | **0.356 ± 0.025** (P 0.78, R 0.23; AUROC 0.61) | 0.820 ± 0.072 | 0.340 ± 0.027 | 0.321 ± 0.043 | 0.321 ± 0.041 |
| xs | session | **0.233 ± 0.025** (P 0.77, R 0.14; AUROC 0.57) | 0.643 ± 0.072 | 0.357 ± 0.022 | 0.357 ± 0.021 | 0.354 ± 0.028 |
| s | request | **0.344 ± 0.020** (P 0.79, R 0.22; AUROC 0.61) | 0.839 ± 0.014 | 0.306 ± 0.025 | 0.289 ± 0.013 | 0.283 ± 0.014 |
| s | session | **0.380 ± 0.013** (P 0.78, R 0.25; AUROC 0.62) | 0.740 ± 0.015 | 0.342 ± 0.014 | 0.342 ± 0.018 | 0.348 ± 0.015 |

## Pairs (Δ directed F1 = `thp/topology` − other, five-seed mean ± std of the difference, paired t-test)

| rung | pair | request | session |
|---|---|---|---|
| xs | − `floor/topology` | −0.463 ± 0.075, p 0.0002 | −0.410 ± 0.059, p 0.0001 |
| xs | − `floor/bigram` | +0.017 ± 0.013, p 0.050 | −0.124 ± 0.018, p 0.0001 |
| xs | − `trace/core/shipped/frozen` | +0.035 ± 0.042, p 0.14 | −0.124 ± 0.018, p 0.0001 |
| xs | − `baseline/granger/shipped/frozen` | +0.036 ± 0.042, p 0.13 | −0.121 ± 0.027, p 0.0005 |
| s | − `floor/topology` | −0.496 ± 0.023, p < 0.0001 | −0.360 ± 0.016, p < 0.0001 |
| s | − `floor/bigram` | +0.038 ± 0.032, p 0.059 | +0.038 ± 0.017, p 0.008 |
| s | − `trace/core/shipped/frozen` | +0.055 ± 0.022, p 0.005 | +0.037 ± 0.015, p 0.006 |
| s | − `baseline/granger/shipped/frozen` | +0.061 ± 0.024, p 0.004 | +0.032 ± 0.012, p 0.004 |

## Hypotheses (`plans/thp.md` §7)

- **H-thp-topology-request — FAIL on xs and s.** −0.46 and −0.50 against `floor/topology`
  (required ≥ +0.02): the hard prior caps recall at the floor's and the BIC keeps a fifth of it.
- **H-thp-topology-session — FAIL on xs and s.** −0.41 and −0.36.
- **H-thp-vs-core — FAIL on xs, PASS on s.** xs: request +0.035 is not significant (p 0.14; 2 / 5
  seeds within ±0.01) and session is −0.12 (p 0.0001, 5 / 5 seeds below the core arm). s: +0.055
  (p 0.005) and +0.037 (p 0.006), 5 / 5 seeds above at both grains.
- The pair against `cascade/topology` of §6 is not reported: CASCADE has no landed cell.

## Reading

TTPM under the hard prior is a high-precision, low-recall reader: three quarters of what it keeps
is true, but it keeps a fifth of the truth, so its F1 lands on the data-driven arms' level
(0.32–0.36) rather than near the prior's (0.82 / 0.84). Its only clear win is at s, where the
second-rung arms fall to 0.28–0.35 and THP holds 0.34–0.38; at xs it is level with them at request
and below them at session. The frozen decay sits at the grid bottom on every corpus and the BIC
penalty is the registered one; neither is varied here. The verdicts are per rung; m is not read
(≈ 1–2 days per fit at the s cost scaling, outside the Phase A gate; owner's rule of 2026-11-03).
