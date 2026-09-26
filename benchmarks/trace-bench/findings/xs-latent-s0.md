# Findings — xs / latent / seed 0 (test read of 2026-09-25)

**Provisional: one seed.** Every pre-registered hypothesis (`plans/reference-arms.md` §6,
`plans/baselines.md` §3, `plans/per-sequence-rules.md`) is stated over the five-seed mean of the
paired per-seed difference; this file scores the single available seed so the direction is on
record before seeds 1–4 land, and is superseded by the rung's `tables/` once `report` can run
(it refuses below five seeds). Source records: `results/xs/latent/seed=0/` (test read
`xs-latent-s0-test-26-09-25` under `freezes/2026-09-25-xs-latent-s0.json`, backbone `ba7725c4…`
at step 500, ε̂ = −0.008; validation sweep `xs-latent-s0-val-26-09-25`). Predict-all directed
F1 at the default floor: 0.211 (request), 0.207 (session). Coverage ceiling: 0.44–0.49 (request),
0.61–0.62 (session).

## Type-level directed F1 at the frozen cells (test / frozen validation)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≡ fixed-kl) | 0.316 / 0.320 | 0.357 / 0.367 |
| `trace/core` fixed | 0.310 / 0.317 | 0.358 / 0.362 |
| `trace/cli` (shipped ≡ fixed-kl) | 0.308 / 0.327 | 0.348 / 0.358 |
| `trace/cli` fixed | 0.302 / 0.320 | 0.345 / 0.359 |
| `trace/cli-atomic` (shipped ≡ fixed-kl) | 0.291 / 0.316 | 0.325 / 0.343 |
| `trace/cli-atomic` fixed | 0.295 / 0.318 | 0.328 / 0.341 |
| `baseline/granger` shipped | 0.306 / 0.334 | 0.341 / 0.341 |
| `baseline/granger` fixed | 0.312 / 0.324 | 0.320 / 0.344 |
| `baseline/saliency` | 0.262 / 0.261 | 0.336 / 0.338 |
| `baseline/shapley` | 0.259 / 0.265 | 0.292 / 0.301 |
| `trace/cli/*/shipped` (the tool's cut) | 0.038–0.039 | 0.043–0.046 |
| `trace/cli-atomic/*/shipped` | 0.043–0.047 | 0.099–0.106 |

Every frozen cell lands within −0.03 of its validation value. Directed AUROC 0.60–0.68
throughout. Orientation accuracy 0.84–0.91 (request; acyclic truth) and 0.41–0.65 (session;
cyclic truth, SID/AID null with the scorer's reason).

## Hypotheses

| hypothesis | this seed | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | Δ = +0.008 (request), +0.009 (session); the `c = 1` request row from the validation table: \|Δ\| ≤ 0.021 (N = 2), 0.023 (N = 8), 0.041 (N = 32) over every τ, < 0.01 at the best τ |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail** | request: cli lag-2 0.14 / lag-3 0.09, cli-atomic 0.16 / 0.10 (≈ 1.1×); session: 0.15 / 0.11 vs 0.15 / 0.12 (≈ 1.0×) |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | n/a at xs | 0 saturated, 0 collapsed on 209,325 cells per path |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a at xs | no corrupted cells; shipped-vs-frozen ΔF1 −0.25 … −0.31 reported; the shipped cut emits 9–44 edges at precision 0.37–0.75 and recall 0.02–0.06 |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | identical predictions on every cell (no corrupted cell to fix); fixed vs fixed-kl within ±0.01 |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** | lag-1 recall 0.25–0.28 (request) and 0.28–0.29 (session) = 0.47–0.6× the ceiling; lag-3 recall 0.09–0.12 ≈ 0.2× |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N=32) − F1(N=8) < 0.01) | **fail** | 8 of 18 reference cells at N ≤ 8 (request 7/9, session 1/9); the session gain of N = 32 over N = 8 is < 0.01 on 5 of 6 arms |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = −0.008 at the frozen checkpoint (step 500; the order-2 floor is a plug-in estimate, not a bound); 0.63 at the last checkpoint |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | pass | request: 0.316 vs granger 0.312 (+0.004), saliency 0.262, shapley 0.259; session: 0.357 vs 0.341 / 0.336 / 0.292 |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.80 (request c1 N2), 0.78 (c2 N2), 0.80 (session c2 N2), 0.92 (c2 N32), validation matrices |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 4.0 s vs 0.016 s per sequence (≈ 250×) |
| H-perseq (pooled per-sequence F1 above predict-all on every scoreable cell) | **fail (request)** / pass (session) | request: core 0.421 < 0.471, cli 0.448 < 0.471, granger 0.435 < 0.438, saliency 0.512 < 0.517, shapley 0.386 < 0.438, cli-atomic 0.542 > 0.517; session: 0.40–0.43 > 0.29–0.31 on every arm; scoreable fraction 74–99 % |

## Other findings

- The ancestor-link column of the per-sequence axis equals the parent-link column in all 38
  cells (`plans/per-sequence-rules.md`, addendum 2026-09-25): the target-edge filter leaves only
  direct call pairs at xs.
- The shipped cut's `tau_by_lag` (pooled 95th percentile with exponential decay 0.3) sits at
  4.3 (lag 1) → 0.5 (lag 13) on request reads; the frozen τ of the same arms is 1e-4 … 3e-2, so
  the tool's rule discards all but the top few dozen pairs.
- Run facts: 20 discover reads, 38 annotate, 38 seqscore; 61 min on one A10G (discover 52 min,
  Shapley 47 of them); peak 3.07 GB; `--args-diff` clean on all 98 records.
