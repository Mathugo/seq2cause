# Findings — CausalNET with the topology (`causalnet/topology`), xs and s / latent, five seeds

The hypotheses of `plans/causalnet.md` §7 are scored here per rung on the five-seed mean of the paired
per-seed difference (same corpus and seed; `report`'s paired t-test, five pairs, low power — the
effect size is the primary readout). Tables: `tables/<rung>-latent-prior-arms/` (`report` over
`results/` with the fourteen pairs of `plans/{thp,lmt,causalnet}.md` §6). Source records:
`results/<rung>/latent/seed={0..4}/causalnet/topology/none/frozen/<grain>/` (annotate's `score.json`,
`score-ranking.json`, `annotate.json`), `test-read/causalnet-topology-<grain>/`, `causalnet-val-sweep/`,
`causalnet-freeze/`, under `freezes/2026-11-05-<rung>-latent-s<k>-external-causalnet.json` (`freeze@3`,
`family: external`, `method: causalnet`, `method_commit aaccce6d`, model `none`). Method: CausalNET at
`aaccce6d` with the lab's two patches (D-CN-1 the lossless decay-tensor collapse, 0.0 difference; D-CN-2 the
drivable entry), fixed arguments `epochs=200` (xs) / `epochs=50` (s) `device=auto seed=42 max_hop=1
readout=shipped`, `batch_size` 512; node = op, the topology is the soft prior (`mask_mode: count`); the
window (ms) is swept in the `N` slot. Reads: EC2 GPU replicas (A10G; `RUN.md` registry 2026-11-04 /
2026-11-05), validation on the whole xs split and the 10k / 5k s sample, test on the whole xs split and the
20k / 10k s sample (head), as every arm. The validation runs landed on their third apply (two env-build
failures, `RUN.md`).

## Caveats carried beside every number

- **Coarse scores at s.** The shipped read-out yields only 7 distinct F1 values over the 20 τ of the floor
  grid at s (20 at xs); the frozen τ 0.15 sits on a step of that staircase.
- **Session cut unstable across seeds at s.** At the frozen τ 0.15 the session test read returned 2 974 and
  3 686 edges on seeds 0 and 3 (F1 0.714 / 0.715) but 1 308–1 530 on seeds 1, 2 and 4 (F1 0.377–0.445): the
  5k-sequence validation fit and the 10k-sequence test fit put the score step on either side of τ. The
  five-seed std of the session F1 is 0.16 against 0.02 at request; every session pair at s inherits it.
- **Window choice.** Validation froze 60 s on 4 / 5 s corpora at both grains (10 s on seed 0) and a mixed
  set at xs (request 10 s × 4, 60 s × 1; session 1 s × 3, 10 s, 60 s); no cell is at a grid edge.
- **Soft mask.** The read counts the pairs the method returns outside the topology (below). Within-op
  pairs are dropped at projection as for every arm.
- **Sample-size note (s).** Validation and test samples differ (10k / 5k vs 20k / 10k); the registered note.
- **m.** The five m validation boxes failed at the first backward pass with CUDA out of memory on the
  24 GB class at the registered batch size (`RUN.md`, 2026-11-05); m has no CausalNET cell.

## Headline (directed F1 at the frozen cut, default floor 0.05, five-seed mean ± std; test reads)

| rung | grain | `causalnet/topology` | `floor/topology` (sample-free) | `lmt/topology` | `thp/topology` | `trace/core/shipped/frozen` |
|---|---|---|---|---|---|---|
| xs | request | **0.500 ± 0.030** (P 0.53, R 0.48; AUROC 0.71) | 0.820 ± 0.072 | 0.647 ± 0.043 | 0.356 ± 0.025 | 0.321 ± 0.043 |
| xs | session | **0.393 ± 0.017** (P 0.59, R 0.30; AUROC 0.63) | 0.643 ± 0.072 | 0.497 ± 0.046 | 0.233 ± 0.025 | 0.357 ± 0.021 |
| s | request | **0.765 ± 0.018** (P 0.65, R 0.94; AUROC 0.93) | 0.839 ± 0.014 | 0.643 ± 0.021 | 0.344 ± 0.020 | 0.289 ± 0.013 |
| s | session | **0.538 ± 0.163** (P 0.70, R 0.48; AUROC 0.85) | 0.740 ± 0.015 | 0.492 ± 0.017 | 0.380 ± 0.013 | 0.342 ± 0.018 |

Validation F1 at the frozen cells: xs 0.469–0.561 / 0.371–0.511, s 0.704–0.777 / 0.676–0.728 (request /
session). Reachable-only F1 0.57 / 0.43 at xs, 0.79 / 0.55 at s; `reachable.predict_all_f1` 0.20 / 0.21 and
0.12 / 0.08 (the cut is far from every scored pair).

## Pairs (Δ directed F1 = `causalnet/topology` − other, five-seed mean ± std of the difference, paired t-test)

| rung | pair | request | session |
|---|---|---|---|
| xs | − `floor/topology` | −0.319 ± 0.069, p 0.0005 | −0.250 ± 0.065, p 0.0010 |
| xs | − `floor/bigram` | +0.161 ± 0.047, p 0.0015 | +0.036 ± 0.027, p 0.041 |
| xs | − `trace/core/shipped/frozen` | +0.179 ± 0.041, p 0.0006 | +0.037 ± 0.025, p 0.028 |
| xs | − `baseline/granger/shipped/frozen` | +0.180 ± 0.036, p 0.0004 | +0.040 ± 0.028, p 0.036 |
| xs | − `lmt/topology` | −0.146 ± 0.041, p 0.0013 | −0.104 ± 0.040, p 0.0046 |
| xs | − `thp/topology` | +0.144 ± 0.050, p 0.0029 | +0.160 ± 0.034, p 0.0005 |
| s | − `floor/topology` | −0.075 ± 0.016, p 0.0004 | −0.202 ± 0.159, p 0.047 |
| s | − `floor/bigram` | +0.459 ± 0.022, p < 0.0001 | +0.197 ± 0.156, p 0.048 |
| s | − `trace/core/shipped/frozen` | +0.476 ± 0.014, p < 0.0001 | +0.196 ± 0.155, p 0.048 |
| s | − `baseline/granger/shipped/frozen` | +0.482 ± 0.021, p < 0.0001 | +0.190 ± 0.157, p 0.053 |
| s | − `lmt/topology` | +0.121 ± 0.012, p < 0.0001 | +0.046 ± 0.157, p 0.55 |
| s | − `thp/topology` | +0.421 ± 0.026, p < 0.0001 | +0.159 ± 0.159, p 0.089 |

## Hypotheses (`plans/causalnet.md` §7)

- **H-causalnet-topology-request — FAIL on xs and s.** −0.32 at xs and −0.075 at s against `floor/topology`
  (required ≥ +0.02): at s the arm comes within 0.06–0.09 of the sample-free prior on every seed (recall
  0.94 against the prior's 1.0, precision 0.65 against 0.72), the closest any method has come.
- **H-causalnet-topology-session — FAIL on xs and s.** −0.25 at xs; −0.20 at s, where seeds 0 and 3 are within
  0.04 of the floor and seeds 1, 2 and 4 are 0.30–0.34 below (the unstable cut above).
- **H-causalnet-vs-core — PASS on xs, PASS on s at request, NOT on s at session.** xs: +0.18 (p 0.0006) and
  +0.037 (p 0.028), 5 / 5 seeds above at request, 4 / 5 at session (seed 4: +0.004). s: request +0.48
  (p < 0.0001, 5 / 5); session +0.20 but p 0.048 with 5 / 5 seeds above — passes the registered test at the
  margin, carried with the instability caveat (std of the difference 0.155).
- **H-causalnet-outside-mask (reported).** The landed test records hold the fit-level count only: of the
  pairs the method scores, `n_edges_outside_mask` lies outside the topology's pair set (`test-read/
  causalnet-topology-<grain>/run/results.json`, per fit); the number of CUT edges outside the mask and their
  precision against the session truth are not in the landed records (the prediction files stay in object
  storage) — a follow-up from the synced `scores-*.npz` and the score-tier graphs, not scored here.

## Reading

CausalNET is the strongest of the three prior methods at s and the second at xs. At the request grain of s it
recovers 94 % of the truth edges at precision 0.65 and sits 0.07 under the prior it was given — the only
arm on any rung within 0.1 of the topology floor. Its weakness is the cut, not the ranking: the AUROC
(0.93 / 0.85 at s) is the highest landed on the rung, but the shipped read-out's scores are coarse, so the
frozen τ selects a step that the test fit of three seeds lands on the wrong side of at session. A finer
score or a re-selected cut is a method-side change the family does not make. The verdicts are per rung; m
has no cell (the A10G's memory, `RUN.md`).
