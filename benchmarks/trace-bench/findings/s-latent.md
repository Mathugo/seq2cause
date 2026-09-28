# Findings — s / latent, five seeds (test reads of 2026-09-27 and 2026-09-28)

Every pre-registered hypothesis (`plans/reference-arms.md` §6, `plans/baselines.md` §3,
`plans/per-sequence-rules.md` §6) is scored here on the five-seed mean of the paired per-seed
difference, as written; the tables the numbers come from are `tables/s-latent/` (`report`, 38
cells × 5 seeds, 12 pre-registered pairs, per-sequence document). Source records:
`results/s/latent/seed={0..4}/` — test reads `s-latent-s0-test-26-09-27` and
`s-latent-s{1..4}-test-26-09-28` under `freezes/2026-09-2{7,8}-s-latent-s{k}.json`; validation
sweeps `s-latent-s0-val-26-09-26` and `s-latent-s{1..4}-val-26-09-27`. Backbones:
argmin-validation checkpoints at steps 6750 / 4000 / 7500 / 5500 / 8000 (corpus-dependent,
unlike xs's uniform step 500), ε̂ = −0.104 / −0.112 / −0.088 / −0.091 / −0.060, all in regime.
This file supersedes `findings/s-latent-s0.md`.

Every arm on every seed: 0 corrupted cells (saturated or collapsed) on every path; `shipped`
≡ `fixed-kl` predictions on every cell. Predict-all directed F1 at the default floor
(type-level, `2p/(n+p)` over the universe): 0.096 / 0.123 / 0.119 / 0.125 / 0.115 (request),
0.062 / 0.076 / 0.078 / 0.083 / 0.080 (session). Coverage ceiling on the test read: 0.46–0.51
(request), 0.68–0.73 (session).

## Registered note — the validation sample and the test read differ in size at `s`

Carried from `findings/s-latent-s0.md` and `plans/caps.md` (addendum 2026-09-27; owner decision):
from rung `s` the validation sweep probes a 10k / 5k head sample and the test read 20k / 10k, so
the two jobs see different coverage and the test F1 of a frozen cell lands **above** its frozen
validation value as a coverage effect, not a generalisation check. On the five seeds the
test-minus-frozen-validation delta is **−0.015 … +0.047** across all 130 frozen cells (the few
negatives are saliency cells). The Shapley baseline's own 2k / 500 sample bounds its ceiling at
0.22–0.26 / 0.26–0.30 against 0.46–0.73 for every other arm; its rows below are sample-bounded.

## Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≡ fixed-kl) | 0.289 ± 0.013 | 0.342 ± 0.018 |
| `trace/core` fixed | 0.289 ± 0.013 | 0.342 ± 0.018 |
| `trace/cli` (shipped ≡ fixed-kl) | 0.292 ± 0.014 | 0.349 ± 0.015 |
| `trace/cli` fixed | 0.292 ± 0.014 | 0.351 ± 0.019 |
| `trace/cli-atomic` (shipped ≡ fixed-kl) | 0.287 ± 0.016 | 0.324 ± 0.015 |
| `trace/cli-atomic` fixed | 0.287 ± 0.018 | 0.321 ± 0.015 |
| `baseline/granger` shipped | 0.283 ± 0.014 | 0.348 ± 0.015 |
| `baseline/granger` fixed | 0.282 ± 0.015 | 0.347 ± 0.015 |
| `baseline/saliency` | 0.268 ± 0.025 | 0.277 ± 0.013 |
| `baseline/shapley` (2k / 500 sample) | 0.158 ± 0.014 | 0.151 ± 0.013 |
| `trace/cli/*/shipped` (the tool's cut) | 0.050–0.051 ± 0.023 | 0.065–0.067 ± 0.022 |
| `trace/cli-atomic/*/shipped` | 0.092–0.093 ± 0.016 | 0.141–0.149 ± 0.031 |

Directed AUROC 0.65–0.67 (request) and 0.73–0.74 (session) on the non-Shapley arms. Orientation
accuracy 0.83–0.87 (request; acyclic truths) and 0.71–0.76 (session; cyclic truths on every
seed, SID/AID null with the scorer's reason — the causal-validity columns of `tables/` carry
the ADMG reason on every cell). Against xs the trace arms are flat (request 0.29 vs 0.31–0.32;
session 0.34–0.35 vs 0.33–0.36) while predict-all halved, so the lift over the trivial
predictor roughly doubles with the vocabulary.

## Hypotheses (five-seed paired means; p from the paired t-test, n = 5)

| hypothesis | verdict | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | core − cli = −0.003 ± 0.003 (p = 0.060, request), −0.006 ± 0.004 (p = 0.034, session) — inside the band, and the sign now favours cli |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail at lag 2; the advantage is lag-graded** | lag-2 ratio 1.26× (request 0.191 vs 0.152), 1.15× (session 0.268 vs 0.233); lag-3 1.98× / 1.74×; lag-4 2.9× / 3.7× — the ≥ 2× ratio the hypothesis expected at lag 2 arrives at lag 3–4 (xs showed no gradient at all) |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | n/a at s | 0 saturated, 0 collapsed on every read of every seed |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a at s | no corrupted cells; no empty prediction; shipped-vs-frozen ΔF1 = −0.241 ± 0.028 / −0.282 ± 0.033 (cli, request / session), −0.194 ± 0.017 / −0.174 ± 0.041 (cli-atomic), all p ≤ 0.001; the cut's precision is higher (+0.30 / +0.14 where measurable) and its recall lower by 0.27–0.35 |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | identical predictions on every cell of every seed (no corrupted cell to fix); fixed vs fixed-kl = +0.000 ± 0.006 (request), −0.000 ± 0.001 (session) |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** | lag-1 recall = 0.53–0.60× the ceiling (core / cli / granger, both grains); lag-3 = 0.13–0.14× for core / cli but 0.24–0.27× for cli-atomic and saliency |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** | N ≤ 8 on 25 of 90 reference cells (28 %; per seed 4 / 4 / 2 / 8 / 7); the session grain froze N = 32 on 42 of 45 |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = −0.060 … −0.112 on every seed (order-2 plug-in floor); last / min validation-loss ratio 1.010–1.016 |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | **fail** | Granger: core − granger = +0.006 ± 0.008 (p = 0.180, request; wins 3 / 5) but **−0.006 ± 0.004 (p = 0.045, session; Granger wins 5 / 5)** — a baseline beats the reference arm at the session grain; saliency +0.021 ± 0.015 (p = 0.04) / +0.065 ± 0.009 (p < 0.001); Shapley +0.131 ± 0.016 / +0.191 ± 0.025 (both p < 0.001, sample-bounded per the registered note) |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.84–0.91 (request), 0.80–0.89 (session) across the five seeds' validation matrices |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 5.5–8.4 s vs 27–31 ms per sequence (204–271×) |
| H-perseq (pooled per-sequence F1 above the pooled predict-all value on every scoreable cell) | **fail (request)** / pass on the reference arms (session) | request: only `trace/cli-atomic` clears predict-all (15 / 15 frozen cells); core 0 / 15, cli 1 / 15, granger 1 / 10, saliency and Shapley 0 / 5; session: core, cli, cli-atomic 15 / 15 each, granger 10 / 10, Shapley 4 / 5, saliency 1 / 5 |

## Other findings

- **A baseline now beats the reference arm.** At the session grain Granger's directed F1 exceeds
  `trace/core`'s on all five seeds (paired +0.006, p = 0.045) with higher precision (p = 0.005),
  and matches `trace/cli` (0.348 vs 0.349). The xs verdict "pass by sign only" does not carry to
  `s`: on the type-level axis the TRACE particle machinery adds nothing over the Granger read-out
  of the same forward at this scale, and the two rank near-identical top deciles (Jaccard ≥ 0.80).
- **The estimator ordering inverted, mildly:** cli ≥ core on both grains (session p = 0.034) and
  cli > cli-atomic at session (+0.025, p = 0.007); at xs core ≥ cli ≥ cli-atomic. All gaps ≤ 0.03
  against a between-seed sd of 0.013–0.019 — the corpus effect still dominates the arm effect.
- **The lag-graded cli-atomic advantage generalises across seeds** (seed-0 finding confirmed):
  its per-lag recall overtakes cli's from lag 3 and reaches 2.9–3.7× by lag 4, at the cost of
  lag-1 recall (0.23 vs 0.27 request). The construction helps exactly where the paper claims,
  one lag later than the registered threshold.
- **The shipped cut remains the largest effect** (−0.17 … −0.28 F1, all p ≤ 0.001), never empty
  at `s`; its induced-empty clause stays untested until rung `m`.
- **The ancestor-link column separates from the parent-link column on seed 2 only** (all 38
  cells, ≤ 0.0017 pooled F1; seeds 0, 1, 3, 4 coincide) — the same per-corpus transitive-edge
  fact as xs seed 4 (`plans/per-sequence-rules.md`).
- **Run facts:** Job T chains 3 h 19 min – 4 h 26 min on one A10G (17–18 discover reads, 38
  annotate + 38 seqscore per seed; annotate minutes under the pinned scorer); peak 2.9 GB;
  manifests verified, `--args-diff` clean on all 95–96 records per seed, freezes bound at
  `5cddbb7` (seeds 1–4) / `0460443` (seed 0), package `9e194cd` / `4a2bae6`; ledger 205 entries.
