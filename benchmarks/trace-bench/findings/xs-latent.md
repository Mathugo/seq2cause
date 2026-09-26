# Findings — xs / latent, five seeds (test reads of 2026-09-25 and 2026-09-26)

Every pre-registered hypothesis (`plans/reference-arms.md` §6, `plans/baselines.md` §3,
`plans/per-sequence-rules.md` §6) is scored here on the five-seed mean of the paired per-seed
difference, as written; the tables the numbers come from are `tables/xs-latent/` (`report`, 38
cells × 5 seeds, 12 pre-registered pairs, per-sequence document). Source records:
`results/xs/latent/seed={0..4}/` — test reads `xs-latent-s0-test-26-09-25` and
`xs-latent-s{1..4}-test-26-09-26` under `freezes/2026-09-2{5,6}-xs-latent-s{k}.json`; validation
sweeps `xs-latent-s{k}-val-26-09-25`. Backbones: argmin-validation checkpoint at step 500 on every
seed, ε̂ = −0.008 / −0.005 / +0.015 / −0.003 / +0.092 (seed 4 borderline, in regime). This file
supersedes `findings/xs-latent-s0.md`.

Every arm on every seed: 0 corrupted cells (saturated or collapsed) on every path; `shipped`
≡ `fixed-kl` predictions on every cell; test directed F1 within −0.037 … +0.025 of the frozen
validation value on every frozen cell. Predict-all directed F1 at the default floor (type-level,
`2p/(n+p)`): 0.211 / 0.214 / 0.192 / 0.129 / 0.142 (request), 0.207 / 0.215 / 0.194 / 0.142 /
0.216 (session). Coverage ceiling 0.40–0.56 (request), 0.55–0.66 (session).

## Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≡ fixed-kl) | 0.321 ± 0.043 | 0.357 ± 0.021 |
| `trace/core` fixed | 0.317 ± 0.050 | 0.360 ± 0.030 |
| `trace/cli` (shipped ≡ fixed-kl) | 0.317 ± 0.045 | 0.353 ± 0.021 |
| `trace/cli` fixed | 0.314 ± 0.043 | 0.355 ± 0.023 |
| `trace/cli-atomic` (shipped ≡ fixed-kl) | 0.307 ± 0.043 | 0.332 ± 0.011 |
| `trace/cli-atomic` fixed | 0.312 ± 0.045 | 0.330 ± 0.018 |
| `baseline/granger` shipped | 0.321 ± 0.041 | 0.354 ± 0.028 |
| `baseline/granger` fixed | 0.318 ± 0.047 | 0.348 ± 0.035 |
| `baseline/saliency` | 0.279 ± 0.040 | 0.324 ± 0.016 |
| `baseline/shapley` | 0.255 ± 0.050 | 0.286 ± 0.013 |
| `trace/cli/*/shipped` (the tool's cut) | 0.033 ± 0.016 | 0.029 ± 0.012 |
| `trace/cli-atomic/*/shipped` | 0.048 ± 0.036 | 0.073 ± 0.028 |

The corpus effect dominates the arm effect: the between-seed sd of any arm is 0.01–0.05 while the
paired between-arm differences are ≤ 0.02 (seed 3 is the sparse corpus — predict-all 0.13–0.14,
ceiling 0.40 / 0.55 — and pulls every arm to 0.18–0.33). Directed AUROC 0.57–0.66 (request) and
0.61–0.67 (session) at the frozen cells. Orientation accuracy 0.72–0.96 (request; acyclic truth)
and 0.34–0.75 (session; cyclic truth, SID/AID null with the scorer's reason).

## Hypotheses (five-seed paired means; p from the paired t-test, n = 5)

| hypothesis | verdict | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | Δ = +0.004 ± 0.009 (p = 0.35, request), +0.004 ± 0.005 (p = 0.19, session); per-seed −0.011 … +0.010 |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail** | lag-2 ratio 0.77–1.13× (request), 0.96–1.37× (session) on every seed; cli's lag-2 recall 0.10–0.26, never reaching 2× |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | n/a at xs | 0 saturated, 0 collapsed on 2.6 M cells per path over the five seeds |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a at xs | no corrupted cells; no empty prediction (the cut emits 4–51 edges at request, 10–114 at session); shipped-vs-frozen ΔF1 −0.284 ± 0.037 (cli, request), −0.324 ± 0.030 (cli, session), −0.259 ± 0.038 / ± 0.036 (cli-atomic), all p < 0.001; the cut's precision is higher by 0.07–0.24 and its recall lower by 0.31–0.36 |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | identical predictions on every cell (no corrupted cell to fix); fixed vs fixed-kl: −0.004 ± 0.010 (p = 0.40, core request), +0.003 ± 0.012 (p = 0.57, core session), within ±0.005 on the cli arms' frozen cut; −0.011 (p < 0.01) on `cli-atomic/fixed/shipped/session` — a shipped-cut cell, ~1–5 edges |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** | lag-1 recall = 0.42–0.63× the ceiling on every seed and grain (core, cli); lag-3 = 0.07–0.42× (above 0.2× on seeds 1, 2 at request) |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** | N ≤ 8 on 55 of 90 reference cells (61 %; request 34/45, session 21/45); validation gain of N = 32 over N = 8 ≥ 0.01 on 12 cells (0.011–0.026, mostly session cli/core) |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ ≤ 0.092 on every seed (order-2 plug-in floor; the 250-step grid is coarse around a sharp minimum, rule as registered) |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | pass by sign only | granger shipped: core − granger = +0.001 ± 0.008 (p = 0.84, request), +0.003 ± 0.010 (p = 0.54, session) — a tie within noise, Granger wins on 2 of 5 seeds at each grain; granger fixed +0.004 / +0.009 (n.s.); saliency +0.042 ± 0.008 (p < 0.001) / +0.032 ± 0.012 (p = 0.004); Shapley +0.066 ± 0.020 (p = 0.002) / +0.071 ± 0.011 (p < 0.001) |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.79–0.92 on every seed and grain (validation matrices at the frozen Granger cell) |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 3.0–5.3 s vs 0.016–0.028 s per sequence (134–282×) |
| H-perseq (pooled per-sequence F1 above predict-all on every scoreable cell) | **fail (request)** / pass on the reference arms (session) | request: core 0.423 vs 0.456 (1 of 5 seeds above), cli 0.431 vs 0.456 (0/5), granger 0.426 vs 0.449 (1/5), saliency 0.352 vs 0.482 (0/5), shapley 0.339 vs 0.466 (0/5), cli-atomic 0.505 vs 0.500 (2/5); session: core 0.433, cli 0.414, cli-atomic 0.443, granger 0.419 vs 0.30–0.31 (5/5 each); saliency 0.296 vs 0.303 (2/5), shapley 0.289 vs 0.306 (0/5); scoreable fraction 0.60–1.00 (request) |

## Other findings

- **The three TRACE readings and Granger are one method at xs.** Core, cli and Granger on the same
  forward differ by ≤ 0.004 F1 (all n.s.); cli-atomic trails cli by 0.010 (p = 0.17, request) and
  0.021 (p = 0.048, session); Granger ranks the same top decile (Jaccard ≥ 0.79). The
  pre-registered ordering trace > Granger cannot be claimed on five seeds.
- **The shipped cut is the largest effect in the study**, costing 0.26–0.32 F1 against the frozen
  τ on every seed with a 4–114-edge output; it is never empty at xs (no corrupted cells to induce
  one), so H-cut's induced-empty clause stays untested until rung `m`.
- **The ancestor-link column separates from the parent-link column on seed 4 only** (≤ 0.0013
  pooled F1; `plans/per-sequence-rules.md`, addendum 2026-09-26); on seeds 0–3 the two coincide.
- **Run facts:** Job T chains 51–77 min on one A10G (discover 44–68 min, Shapley 65–90 % of it),
  peak 3.07 GB; 18–20 discover reads, 38 annotate + 38 seqscore per seed; `--args-diff` clean on
  all 96–98 records per seed; package commit `38bd891` on every box; records verified against the
  four manifests (the 36–40 `*.npz` per read left in object storage).
- **Harness fix landed with this file:** `report` looked for `seqscore.json` at the cell root while
  the job writes it under `<cell>/seqscore/`, so the per-sequence document would have been silently
  empty on every real run (the pipeline test copied the file up); `load_cells` now accepts both.
