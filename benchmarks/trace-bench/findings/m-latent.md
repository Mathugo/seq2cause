# Findings — m / latent, five seeds (test reads of 2026-09-29 and 2026-09-30)

Every pre-registered hypothesis (`plans/reference-arms.md` §6, `plans/baselines.md` §3,
`plans/per-sequence-rules.md` §6) is scored here on the five-seed mean of the paired per-seed
difference, as written; the tables the numbers come from are `tables/m-latent/` (`report`, 38
cells × 5 seeds, 12 pre-registered pairs, per-sequence document). Source records:
`results/m/latent/seed={0..4}/` — test reads `m-latent-s0-test-26-09-29` and
`m-latent-s{1..4}-test-26-09-30` under `freezes/2026-09-{29,30}-m-latent-s{k}.json`; validation
sweeps `m-latent-s0-val-26-09-28` and `m-latent-s{1..4}-val-26-09-29`. Backbones:
argmin-validation checkpoints at steps 7500 / 4000 / 4250 / 6500 / 7750 (corpus-dependent),
ε̂ = −0.021 / −0.017 / −0.022 / −0.004 / −0.027, all in regime.
This file supersedes `findings/m-latent-s0.md`.

Every arm on every seed: 0 corrupted cells (saturated or collapsed) on every path of every read
— validation and test — so **H-hazard's fail clause settles at `m` for the latent corpora** (see
the table). `shipped` ≡ `fixed-kl` predictions on every cell. Predict-all directed F1 at the
default floor (type-level, `2p/(n+p)` over the universe): 0.029 / 0.025 / 0.027 / 0.023 / 0.028
(request), 0.014 / 0.013 / 0.014 / 0.012 / 0.015 (session). Coverage ceiling on the test read:
0.185–0.211 (request), 0.304–0.347 (session).

## Registered note — the validation sample and the test read differ in size at `m`

Carried from `findings/s-latent.md` and `plans/caps.md` (addendum 2026-09-27; owner decision):
the validation sweep probes a 10k / 5k head sample and the test read 20k / 10k, so the two jobs
see different coverage and the test F1 of a frozen cell lands **above** its frozen validation
value as a coverage effect, not a generalisation check. On the five seeds the
test-minus-frozen-validation delta is **−0.001 … +0.080** across all 130 frozen cells (the few
negatives are Shapley cells, which probe their own capped sample both times); the ceiling rises
0.126–0.143 → 0.185–0.211 (request) and 0.198–0.224 → 0.304–0.347 (session) between the two
jobs. The gap keeps growing with the rung because the universe grows (1.6–1.9 M request /
3.5–3.9 M session ordered pairs) while the samples stay fixed. The Shapley baseline's own
2k / 500 sample bounds its ceiling at 0.054–0.065 / 0.062–0.087 — under a fifth of the other
arms' — so its rows below are sample-bounded.

## Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≡ fixed-kl) | 0.204 ± 0.009 | 0.270 ± 0.010 |
| `trace/core` fixed | 0.205 ± 0.009 | 0.269 ± 0.010 |
| `trace/cli` (shipped ≡ fixed-kl) | 0.205 ± 0.009 | 0.274 ± 0.010 |
| `trace/cli` fixed | 0.205 ± 0.009 | 0.275 ± 0.010 |
| `trace/cli-atomic` (shipped ≡ fixed-kl) | 0.196 ± 0.005 | 0.247 ± 0.009 |
| `trace/cli-atomic` fixed | 0.195 ± 0.005 | 0.247 ± 0.009 |
| `baseline/granger` shipped | 0.206 ± 0.007 | 0.271 ± 0.011 |
| `baseline/granger` fixed | 0.206 ± 0.007 | 0.271 ± 0.010 |
| `baseline/saliency` | 0.192 ± 0.004 | 0.222 ± 0.025 |
| `baseline/shapley` (2k / 500 sample) | 0.070 ± 0.004 | 0.070 ± 0.006 |
| `trace/cli/*/shipped` (the tool's cut) | 0.024 ± 0.005 | 0.021 ± 0.003 |
| `trace/cli-atomic/*/shipped` | 0.028–0.030 ± 0.006 | 0.044–0.045 ± 0.005 |

Directed AUROC 0.58–0.59 (request) and 0.64–0.65 (session) on the non-Shapley arms, down from
0.65–0.67 / 0.73–0.74 at `s`. Orientation accuracy 0.92–0.96 (request; acyclic truths) and
0.73–0.90 (session; cyclic truths on every seed, SID/AID null with the scorer's ADMG reason on
every cell). Against `s` every type-level F1 fell (request 0.29 → 0.20, session 0.34–0.35 →
0.27) while predict-all fell faster (0.10–0.13 → 0.02–0.03, 0.06–0.08 → 0.01), so the lift over
the trivial predictor keeps growing with the vocabulary.

## Hypotheses (five-seed paired means; p from the paired t-test, n = 5)

| hypothesis | verdict | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | core − cli = −0.000 ± 0.003 (p = 0.74, request), **−0.004 ± 0.002 (p = 0.007, session)** — inside the band, but cli is now significantly above core at session (at `s`: p = 0.034) |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail at every lag on most seeds; the gradient flattened vs `s`** | lag-2 ratio 1.05–1.20×; lag-3 1.21–1.60×; lag-4 1.29–1.97×; lag-5 1.26–2.14× — the ≥ 2× ratio that arrived at lag 3–4 at `s` (2.9× / 3.7× at lag 4) is reached only once at `m` (seed 0, session, lag 5) |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | **FAIL at `m` (latent)** | 0 saturated, 0 collapsed on every path of every read — validation sweeps and test reads — of all five latent corpora (per-read bands 321 k / 1.6 M cells per path); the hypothesis's fail clause ("a rung ≥ m with zero corrupted cells on all ten corpora") settles for the latent half; the observable variant is not run |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a (induced-empty clause vacuous: no corrupted cells anywhere) | no empty prediction on any seed; shipped-vs-frozen paired ΔF1 = −0.181 ± 0.012 (request) / −0.254 ± 0.008 (session) for cli and −0.165 ± 0.009 / −0.202 ± 0.010 for cli-atomic, all p < 0.001, 0/5 wins |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | shipped ≡ fixed-kl edge sets and thresholded metrics on every cell of every seed (on seed 0 the ranking scores differ in the last float bits, AUROC/AP Δ ≤ 4e-9 — the first non-bit-identical rung); fixed vs fixed-kl null (+0.000, p = 0.29 request; −0.001, p = 0.40 session) |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** (first clause on every corpus; second clause once) | lag-1 recall = 0.65–0.70× the ceiling (request core/cli), 0.59–0.62× (session); lag-3 = 0.12–0.21× — the second clause holds except core-request on seed 2 (0.21×) |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** | N ≤ 8 on 18 of 90 reference cells (20 %; per seed 1 / 5 / 3 / 6 / 3) — between `s` (28 %) and the 80 % the hypothesis expected |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = −0.004 … −0.027 at the frozen checkpoints, all in regime; last / min 1.005–1.026 |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | **fail** by sign at both grains (Granger) | core − granger = −0.002 ± 0.003 (p = 0.26, request; granger wins 4/5) and −0.001 ± 0.001 (**p = 0.050, session**; granger wins 4/5) — at `s` the session gap was −0.006 (p = 0.045); saliency (−0.012 request, −0.048 session) and Shapley (sample-bounded) stay below |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.92–0.94 (request), 0.76–0.78 (session) on every seed's frozen configs; validation matrices, `max` aggregate |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 199–477× (session; 5.8–8.2 s vs 16–35 ms per sequence); 65–74× (request) |
| H-perseq (pooled per-sequence F1 above the pooled predict-all value on every scoreable cell) | pass on 25 of 26 frozen cells on every seed | the one failure is saliency at request on all five seeds (0.38–0.40 vs predict-all 0.39–0.42); every other arm clears at both grains; scoreable fraction 0.99 throughout — the request-side failures of `xs` and `s` are gone |

## Other findings

- **The three TRACE readings and Granger remain one method at `m`**: core, cli and Granger within
  0.005 at both grains on the five-seed means, with Granger's nose ahead at both (the H-baselines
  line). The stable separations are *within* the method family: cli > cli-atomic (+0.009,
  p = 0.019 request; +0.028, p < 0.001 session, 5/5 seeds) and cli > core at session (p = 0.007).
  `cli-atomic` is now the worst trace arm at both grains — a full reversal of its `s`-rung
  request lead — yet it remains the best per-sequence arm (pooled 0.48–0.50 both grains).
- **The shipped cut remains the largest effect anywhere**: −0.17 … −0.25 F1 against the frozen τ
  (p < 0.001, 0/5 wins), with 312–1,256 edges; never empty, so the induced-empty clause of H-cut
  has still never been exercised — and with H-hazard settling at zero corrupted cells, it may
  never be.
- **The ancestor-link column separates from the parent-link column on every scored cell of every
  seed** (max |ΔF1| ≈ 0.001): transitive target edges survive the per-sequence filter at `m`
  (at `s` only one seed separated, at `xs` one).
- **Run facts:** chains 13 h 46 min (seed 0, cap 18) and 15.8–16.8 h (seeds 1–4, cap 18 — the
  slowest margin is 1.07×); discover 205–244 min; **annotate 593–745 min, the m-rung cost driver
  on every seed** (the scorer's floor sweep over the ≈ 40× universe); seqscore 19–22 min; peak
  2.86–2.89 GB; `--args-diff` clean on all 94 / 95 / 94 / 94 records; freezes bound at `8063c5f`
  (seed 0) / `39ca636` (seeds 1–4); package commits `cedbbd1` / `a1dcfdb`; 0 corrupted cells;
  manifests verified (the `*.npz` left in object storage).
