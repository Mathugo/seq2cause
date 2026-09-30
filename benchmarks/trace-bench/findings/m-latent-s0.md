# Findings — m / latent / seed 0 (test read of 2026-09-29)

**Provisional: one seed.** Every pre-registered hypothesis (`plans/reference-arms.md` §6,
`plans/baselines.md` §3, `plans/per-sequence-rules.md`) is stated over the five-seed mean of the
paired per-seed difference; this file scores the single available `m` seed so the direction is on
record before seeds 1–4 land, and is superseded by the rung's `tables/` once `report` can run
(it refuses below five seeds). Source records: `results/m/latent/seed=0/` (test read
`m-latent-s0-test-26-09-29` under `freezes/2026-09-29-m-latent-s0.json`, backbone `1d6a5485…`
at step 7500, ε̂ = −0.021; validation sweep `m-latent-s0-val-26-09-28`). Predict-all directed
F1 at the default floor (type-level, `2p/(n+p)` over the universe): 0.029 (request), 0.014
(session). Coverage ceiling on the test read: 0.185 (request), 0.304 (session); on the
validation sample 0.126 / 0.198 (see the registered note below).

## Registered note — the validation sample and the test read differ in size at `m`

The `s` mechanism (`findings/s-latent-s0.md`, owner decision 2026-09-27: sizes stay as
registered) repeats at `m` with a larger gap. The validation sweep probes a 10,000 / 5,000 head
sample, the test read 20,000 / 10,000; on this corpus the reachable-recall ceiling rises from
0.126 to 0.185 (request) and from 0.198 to 0.304 (session) between the two jobs, and **test
directed F1 sits above the frozen validation value on every one of the 26 frozen cells, by
+0.027 … +0.068** (request +0.027 … +0.047, session +0.027 … +0.068; Shapley, probing its own
capped sample both times, moves −0.005 … −0.000). The frozen τ transfers — no cell lost F1 to
the larger read — but the "test within validation" comparison is a coverage effect at `m` and
is not read as a generalisation check. The gap grows with the rung because the universe grows
(1.7 M ordered pairs at request, 3.7 M at session, vs 0.53 M / 1.1 M at `s`) while the sample
sizes stay fixed.

The same mechanism bounds the Shapley baseline separately: its reads take 2,000 / 500 sequences
(D-SB-14), so its coverage ceiling is 0.057 / 0.067 against 0.185 / 0.304 for every other arm —
at `m` the Shapley sample covers less than a fifth of what the full-sample arms reach, and its
type-level F1 (0.063 / 0.062) is a sample-bounded number, not a probe verdict.

## Type-level directed F1 at the frozen cells (test / frozen validation)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≡ fixed-kl) | 0.194 / 0.148 | 0.252 / 0.190 |
| `trace/core` fixed | 0.195 / 0.148 | 0.252 / 0.190 |
| `trace/cli` (shipped ≡ fixed-kl) | 0.194 / 0.150 | 0.258 / 0.191 |
| `trace/cli` fixed | 0.194 / 0.149 | 0.258 / 0.190 |
| `trace/cli-atomic` (shipped ≡ fixed-kl) | 0.188 / 0.149 | 0.232 / 0.184 |
| `trace/cli-atomic` fixed | 0.188 / 0.150 | 0.232 / 0.185 |
| `baseline/granger` shipped | 0.195 / 0.150 | 0.253 / 0.190 |
| `baseline/granger` fixed | 0.196 / 0.149 | 0.254 / 0.190 |
| `baseline/saliency` | 0.188 / 0.145 | 0.199 / 0.173 |
| `baseline/shapley` (2k / 500 sample) | 0.063 / 0.063 | 0.062 / 0.067 |
| `trace/cli/*/shipped` (the tool's cut) | 0.021 (333–343 edges) | 0.016 (312–316 edges) |
| `trace/cli-atomic/*/shipped` | 0.025–0.029 (654–722 edges) | 0.042–0.043 (1,239–1,256 edges) |

Directed AUROC 0.57–0.58 (request) and 0.60–0.62 (session) on every non-Shapley arm (Shapley
0.52 / 0.52), down from 0.64–0.65 / 0.69–0.72 at `s`. Orientation accuracy 0.96 (request;
acyclic truth) and 0.89 (session; cyclic truth, SID/AID null with the scorer's reason).
Against the `s` seed-0 read every type-level F1 fell (request 0.29–0.31 → 0.19–0.20, session
0.34–0.36 → 0.23–0.26) while predict-all fell faster (0.096 → 0.029, 0.062 → 0.014): the
absolute level drops with the ten-fold universe, the lift over the trivial predictor grows.

## Hypotheses

| hypothesis | this seed | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | Δ = 0.000 (request), −0.005 (session) |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail** at lag 2–4; the 2× ratio arrives at session lag 5 | request: ratios cli-atomic/cli 1.06 (lag 2), 1.30 (3), 1.43 (4), 1.66 (5); session 1.08, 1.45, 1.82, 2.14 (lag 5), 2.6 (lag 8) — the `s` pattern, shifted one lag deeper |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | **fail on this corpus** (first live rung) | 0 collapsed, 0 saturated on 321,185 (request) / 1,639,456 (session) cells per path on every read; the validation sweeps of seeds 0–4 also count 0 — the rung-level fail clause ("zero on all corpora") is one test read of seeds 1–4 from settling |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a (induced-empty clause vacuous: no corrupted cells) | no empty prediction; shipped-vs-frozen ΔF1 −0.17 … −0.24 (cli), −0.16 … −0.19 (cli-atomic); the cut's precision is 0.50–0.83 and its recall 0.008–0.022 |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | edge sets and thresholded metrics identical on every cell; the ranking scores now differ in the last float bits (AUROC/AP Δ ≤ 4e-9) — the first rung where the two paths are not bit-identical; fixed vs fixed-kl within ±0.001 everywhere |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** (first clause) | lag-1 recall = 0.67–0.70× the ceiling (request core/cli), 0.61–0.62× (session); lag-3 = 0.15–0.18× (request), 0.15–0.16× (session) — the second clause holds for core and cli |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** | N ≤ 8 on 1 of 18 reference cells (cli-atomic/fixed/request); 17 of 18 froze at N = 32 — worse than `s` (4/18) |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = −0.021 at the frozen checkpoint (step 7500; order-2 plug-in floor 2.161 vs validation loss 2.046); last checkpoint 1.024× the minimum |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | **fail** by sign (Granger, both grains) | request: core 0.194 vs granger 0.195 (−0.002); session: core 0.252 vs granger 0.253 (−0.000 shipped, −0.001 fixed); saliency 0.188 / 0.199 and shapley 0.063 / 0.062 below — at `s` the sign flipped only at request |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.92 (request c1 N32, shipped), 0.77 (session c2 N32, shipped), 0.93 / 0.77 (fixed); validation matrices, `max` aggregate |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 5.79 s vs 0.030 s per sequence (≈ 194×); request 0.85 s vs 0.014 s (≈ 60×) |
| H-perseq (pooled per-sequence F1 above the pooled predict-all value on every scoreable cell) | pass on every arm but saliency at request | request (predict-all 0.421): cli-atomic 0.495, granger 0.466, core 0.461, cli 0.460, shapley 0.453 (vs 0.434) above; saliency 0.383 below; session (predict-all 0.159): every arm above (0.206–0.480); scoreable fraction 0.99 both grains — a flip from `s`, where most arms failed at request |

## Other findings

- **The three TRACE readings and Granger stay one method at `m`** at the type level: core, cli
  and Granger differ by ≤ 0.005 at both grains, and Granger's sign advantage over core now
  appears at both grains. `cli-atomic` is now the *worst* trace arm at both grains (−0.006
  request, −0.026 session against cli) — at `s` it led at request — yet it remains the best
  per-sequence arm at both grains (0.495 / 0.480 pooled).
- **The shipped cut remains the largest effect**: −0.16 … −0.24 F1 against the frozen τ, with
  312–1,256 edges; never empty (no corrupted cells to induce one), so H-cut's induced-empty
  clause stays untested at `m` seed 0.
- **The ancestor-link column separates from the parent-link column on all 38 cells at `m`**
  (max |ΔF1| = 0.0011): transitive target edges survive the per-sequence filter for the first
  time (at `s` the two columns were equal on every cell; at `xs` only seed 4 separated).
- **Short-sequence skips:** the session reads probed 9,975 of 10,000 (25 skipped short); the
  request reads probed all 20,000; 205 request / 38 session sequences were unscoreable on the
  per-sequence axis (scoreable fraction 0.99).
- **Run facts:** 16 discover reads, 38 annotate, 38 seqscore; 13 h 46 min on one A10G (discover
  205 min — saliency 30 + 46 min, Shapley 28 + 48 min, particle reads 2.5–5.0 min each;
  **annotate 593 min — the m-rung cost driver, vs 6.8 min at `s`** — the scorer's floor sweep
  over the ~40× universe; seqscore 19 min); peak 2.86 GB; `--args-diff` clean on all 94
  records; package commit `cedbbd1` (tree dirty on the box: the untracked `input/` artifact
  sync, as at every test read); records verified against the manifest (the `*.npz` left in
  object storage).
