# Findings — LMT with the topology (`lmt/topology`), xs and s / latent, five seeds

The hypotheses of `plans/lmt.md` §7 are scored here per rung on the five-seed mean of the paired
per-seed difference (same corpus and seed; `report`'s paired t-test, five pairs, low power — the
effect size is the primary readout). Tables: `tables/<rung>-latent-prior-arms/` (`report` over
`results/` with the pairs of `plans/lmt.md` §6 and `plans/thp.md` §6; the arms' and floors' cells
are the landed ones). Source records: `results/<rung>/latent/seed={0..4}/lmt/topology/none/frozen/<grain>/`
(annotate's `score.json`, `score-ranking.json`, `annotate.json`), `test-read/lmt-topology-<grain>/`,
`lmt-val-sweep/`, `lmt-freeze/`, under `freezes/2026-11-04-<rung>-latent-s<k>-external-lmt.json`
(`freeze@3`, `family: external`, `method: lmt`, `method_commit 6f4143dc`, model `none`). Method:
LMT at `6f4143dc` with the lab's vectorised objective (D-LM-1, equal to the shipped loop to 3e-14) and
the re-registered fixed arguments (D-LM-5, selected on the xs seed-0 VALIDATION split and disclosed:
`epochs=200 lr=0.05 beta_dag=0.1 lambda_e=0.1 init_std=0.35 a_init=-2 seed=0 device=cpu q_in=0.5
q_out=0.01 dtype=float64 param_dtype=float64 chunk_events=0`); node = event type, the topology enters
as the prior's `q_in` / `q_out` split with the prior weight swept in the `N` slot. Reads: EC2
replicas (`RUN.md` registry, 2026-11-04), validation on the whole xs split and the 10k / 5k s
sample, test on the whole xs split and the 20k / 10k s sample (head), as every arm.

## Caveats carried beside every number

- **Grid edge in the setting slot.** The frozen prior weight is 100 (`N` = 100000, the TOP of the
  registered grid {1, 10, 100}) on 10 / 10 corpora at both grains; τ = 0.01 is interior on every
  cell. `freeze` flags τ edges only, so these cells are not flagged. Whether a heavier prior would
  score higher is unknown; extending the grid is a dated-addendum decision (`plans/lmt.md` §3).
- **Sample-size note (s).** Validation and test samples differ in size (10k / 5k vs 20k / 10k); a
  test number is not a replay of the validation number (the registered note, `plans/caps.md`).
- **Validation-selected arguments.** The fixed arguments were chosen on the xs seed-0 validation
  split (D-LM-5); every other corpus and every test read is out of that selection. The paper must
  say so (4× the shipped epochs; `a_init` and `init_std` changed).
- **Soft mask.** `mask_mode: count` — the method may return pairs outside the topology; the read
  counts them (below) and drops within-op pairs at projection as every arm.

## Headline (directed F1 at the frozen cut, default floor 0.05, five-seed mean ± std; test reads)

| rung | grain | `lmt/topology` | `floor/topology` (sample-free) | `floor/bigram` | `trace/core/shipped/frozen` | `baseline/granger/shipped/frozen` |
|---|---|---|---|---|---|---|
| xs | request | **0.647 ± 0.043** (P 0.65, R 0.65; AUROC 0.79) | 0.820 ± 0.072 | 0.340 ± 0.027 | 0.321 ± 0.043 | 0.321 ± 0.041 |
| xs | session | **0.497 ± 0.046** (P 0.66, R 0.40; AUROC 0.73) | 0.643 ± 0.072 | 0.357 ± 0.022 | 0.357 ± 0.021 | 0.354 ± 0.028 |
| s | request | **0.643 ± 0.021** (P 0.56, R 0.76; AUROC 0.91) | 0.839 ± 0.014 | 0.306 ± 0.025 | 0.289 ± 0.013 | 0.283 ± 0.014 |
| s | session | **0.492 ± 0.017** (P 0.39, R 0.66; AUROC 0.86) | 0.740 ± 0.015 | 0.342 ± 0.014 | 0.342 ± 0.018 | 0.348 ± 0.015 |

Validation F1 at the frozen cells: xs 0.571–0.710 / 0.403–0.560, s 0.612–0.702 / 0.517–0.574
(request / session); the test numbers sit within 0.03 of them at request and 0.04–0.07 under them
at session (the larger test sample adds pairs below τ faster than truth).

Reachable-only columns (`plans/per-sequence-rules.md`, addendum 2026-10-04): reachable F1 0.73 / 0.54
at xs, 0.66 / 0.50 at s; `reachable.predict_all_f1` 0.20 / 0.21 at xs, 0.12 / 0.08 at s — the arm's
cut is far from predicting every scored pair (LMT scores 4.7k–5.2k pairs at xs and 93k–103k at s and
keeps 312–438 / 3.4k–6.5k).

## Pairs (Δ directed F1 = `lmt/topology` − other, five-seed mean ± std of the difference, paired t-test)

| rung | pair | request | session |
|---|---|---|---|
| xs | − `floor/topology` | −0.173 ± 0.032, p 0.0003 | −0.146 ± 0.026, p 0.0002 |
| xs | − `floor/bigram` | +0.307 ± 0.041, p 0.0001 | +0.140 ± 0.036, p 0.0010 |
| xs | − `trace/core/shipped/frozen` | +0.325 ± 0.019, p < 0.0001 | +0.140 ± 0.034, p 0.0008 |
| xs | − `baseline/granger/shipped/frozen` | +0.326 ± 0.021, p < 0.0001 | +0.143 ± 0.036, p 0.0009 |
| s | − `floor/topology` | −0.196 ± 0.010, p < 0.0001 | −0.248 ± 0.003, p < 0.0001 |
| s | − `floor/bigram` | +0.337 ± 0.018, p < 0.0001 | +0.150 ± 0.015, p < 0.0001 |
| s | − `trace/core/shipped/frozen` | +0.354 ± 0.012, p < 0.0001 | +0.150 ± 0.012, p < 0.0001 |
| s | − `baseline/granger/shipped/frozen` | +0.360 ± 0.019, p < 0.0001 | +0.144 ± 0.011, p < 0.0001 |

The gain over every data-driven arm holds on 5 / 5 seeds at both rungs and both grains (per-seed
differences all positive; smallest +0.088 at xs session against the bigram floor). The deficit
against the topology floor holds on 5 / 5 seeds as well: LMT under the topology prior never reaches
the prior's own F1 — at request the floor's recall is 1.0 by construction and LMT's cut keeps
65–76 % of the truth at a precision 0.56–0.65 below the floor's 0.70–0.72.

## Hypotheses (`plans/lmt.md` §7)

- **H-lmt-topology-request — FAIL on xs and s.** The gain over `floor/topology` is required to be
  ≥ +0.02 (p < 0.05); it is −0.17 at xs and −0.20 at s, significant in the wrong direction.
- **H-lmt-topology-session — FAIL on xs and s.** −0.15 at xs, −0.25 at s.
- **H-lmt-vs-core — PASS on xs and s, both grains.** `lmt/topology` is above `trace/core/shipped/frozen`
  by +0.33 / +0.14 at xs and +0.35 / +0.15 at s (request / session), p ≤ 0.001 everywhere, 5 / 5 seeds.
- **H-lmt-outside-mask (reported).** The landed records carry the fit-level count only: of the
  pairs LMT scores non-zero, 4 500–4 914 of 4 746–5 172 at xs and 91 377–99 952 of 92 856–102 162 at s
  lie OUTSIDE the topology's pair set (`n_edges_outside_mask` in `test-read/lmt-topology-<grain>/run/
  results.json`); the number of CUT edges outside the mask and their precision against the session
  truth are not in the landed records (the prediction files stay in object storage) — a follow-up
  from the synced `scores-*.npz` and the score-tier graphs, not scored here.

## Reading

The topology prior does the work. `lmt/topology` sits between the data-driven arms (≈ 0.29–0.36 at
both grains) and the sample-free prior (0.82 / 0.84 at request); it beats every learned arm by a
wide, stable margin and loses to the prior it was given by a comparable one. At s the AUROC
(0.91 / 0.86) is the highest of any arm landed on the rung, so the ranking is informative and the
F1 deficit sits in the cut: the frozen τ keeps too few of the prior's pairs at request (recall 0.76
against the floor's 1.0) and admits too many outside pairs at session (precision 0.39 against 0.72).
Whether a prior weight above 100 (the grid top on 10 / 10 corpora) closes the gap is the open
question the registered grid cannot answer. The verdicts are per rung; m is not read ("where it
lands" by the Phase A gate, owner's rule of 2026-11-03).
