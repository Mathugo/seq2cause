# Findings — xl / latent, five seeds (test reads of 2026-10-06, scored 2026-10-06/07)

Every pre-registered hypothesis (`plans/reference-arms.md` §6, `plans/baselines.md` §3,
`plans/per-sequence-rules.md` §6) is scored here on the five-seed mean of the paired per-seed
difference, as written; the tables the numbers come from are `tables/xl-latent/` (`report`, 38
cells × 5 seeds, 12 pre-registered pairs, per-sequence document, reachable-only columns). Source
records: `results/xl/latent/seed={0..4}/` — discover reads `xl-latent-s{0..4}-test-26-10-05`
(the GPU half of the test read) under `freezes/2026-10-05-xl-latent-s{k}.json`, scored by
`xl-latent-s{0..4}-annotate-26-10-05` (the CPU half: annotate + seqscore); validation sweeps
`xl-latent-s{k}-val-26-10-04` scored twice (`-val-score`, `-val-score2`; the freezes are taken
from the second, on the extended τ grids of the 2026-10-05 addenda). Backbones:
argmin-validation checkpoints at steps 1500 / 750 / 4000 / 2500 / 750, ε̂ = +0.042 / +0.017 /
+0.036 / +0.005 / +0.007, all in regime. `xl` is the last rung of the ladder.

Every arm on every seed: 0 corrupted cells (saturated or collapsed) on every path of every read
— validation and test. Predict-all directed F1 at the default floor (type-level, `2p/(n+p)` over
the universe): 0.0071 / 0.0066 / 0.0060 / 0.0058 / 0.0065 (request), 0.0031 / 0.0029 / 0.0026 /
0.0026 / 0.0027 (session). Coverage ceiling on the test read: 0.060–0.064 (request), 0.087–0.093
(session); universe 55.5–66.3 M (request) / 130.9–152.8 M (session) ordered pairs, truth
193–198 k / 196–202 k directed edges.

## Registered note — the validation sample and the test read differ in size at `xl`

Carried from `findings/s-latent.md` … `findings/l-latent.md` and `plans/caps.md` (addendum
2026-09-27; owner decision): the validation sweep probes a 10k / 5k head sample and the test
read 20k / 10k, so the test F1 of a frozen cell lands **above** its frozen validation value as a
coverage effect, not a generalisation check. On the five seeds the test-minus-frozen-validation
delta is **−0.001 … +0.038** across all 130 frozen cells (every negative is a Shapley cell, which
probes its own capped sample both times); the ceiling rises 0.034–0.038 → 0.060–0.064 (request)
and 0.048–0.054 → 0.087–0.093 (session) between the two jobs. The validation sample reaches
3–5 % of the truth edges and the test read 6–9 %: the most ceiling-bound rung of the ladder.

## Registered note — at request the frozen cells are the every-observed-pair cut

Carried from the 2026-10-05 addenda (`plans/reference-arms.md` §3, `plans/baselines.md` §2) and
the `freeze@3` reference line. On the extended grids every request-grain reference cell, Granger
and saliency froze at τ = 0 or at the bottom of its grid on four of five seeds (seed 2 is the
exception for the trace arms), so the frozen request cut is "every pair the arm scored
positively"; the freeze records the F1 of predicting *every* scored pair as a reference line
(0.056–0.061 on the validation sample). On the test read that line is **0.084–0.089** per seed
(from the read's coverage: `tp = ceiling × truth`, predictions = the co-observed pairs), and the
frozen cells sit on it: `trace/cli-atomic` (τ = 0 or 1e-9 on four seeds) and saliency (`p0` on
four seeds) predict 85–96 % of the co-observed pairs and score 0.000–0.002 **above** the line,
`trace/core` and `trace/cli` (76–78 % of the pairs) 0.001–0.002 above it, Granger (62–66 %)
0.004–0.007 below it. The request-grain ordering at `xl` is therefore the count of positively
scored pairs, not a ranking quality; the hypotheses below are scored as registered, with this
note beside them. At session every arm is interior to its grid (no grid-edge cell on any seed)
and the sweep beats the reference line by 0.02–0.04.

## Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≈ fixed-kl) | 0.088 ± 0.002 | **0.097 ± 0.009** |
| `trace/core` fixed | 0.088 ± 0.002 | 0.097 ± 0.008 |
| `trace/cli` (shipped ≈ fixed-kl) | 0.088 ± 0.002 | 0.093 ± 0.009 |
| `trace/cli` fixed | 0.088 ± 0.002 | 0.093 ± 0.010 |
| `trace/cli-atomic` (shipped ≈ fixed-kl) | 0.089 ± 0.002 | 0.088 ± 0.010 |
| `trace/cli-atomic` fixed | 0.089 ± 0.002 | 0.088 ± 0.010 |
| `baseline/granger` shipped | 0.083 ± 0.003 | 0.095 ± 0.009 |
| `baseline/granger` fixed | 0.083 ± 0.004 | 0.095 ± 0.009 |
| `baseline/saliency` | **0.089 ± 0.002** | 0.095 ± 0.013 |
| `baseline/shapley` (2k / 500 sample) | 0.022 ± 0.001 | 0.023 ± 0.002 |
| `trace/cli/*/shipped` (the tool's cut) | 0.001 ± 0.001 | 0.001 ± 0.001 |
| `trace/cli-atomic/*/shipped` | 0.002 ± 0.001 | 0.001 ± 0.001 |
| *every observed pair (reference line, test)* | *0.084–0.089* | *0.061–0.064* |

Directed AUROC 0.523–0.531 (request) and 0.533–0.545 (session) on the non-Shapley arms, down
from 0.55–0.56 / 0.57–0.58 at `l`. Orientation accuracy 0.94–0.97 (request; acyclic truths) and
0.71–0.83 (session; cyclic truths on every seed, SID/AID null with the scorer's ADMG reason on
every cell). Against `l` every type-level F1 fell (request 0.135 → 0.088, session 0.16 → 0.097)
while predict-all fell faster (0.011 → 0.006, 0.005 → 0.003), so the ratio over the trivial
predictor keeps growing with the vocabulary (12× → 14× at request); the absolute lift does not.

## Hypotheses (five-seed paired means; p from the paired t-test, n = 5)

| hypothesis | verdict | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | core − cli = +0.0002 ± 0.0001 (p = 0.032, request; 4/5) and **+0.0044 ± 0.0014 (p = 0.002, session; core wins 5/5)** — inside the band; the session sign of `l` (core above cli) holds and widens |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail at every lag on every seed; at session the ratio is ≤ 1 on every seed** | request: lag-2 … lag-5 ratio 1.10–1.20×; session: 0.81–1.02× — the gradient of `s` is gone for good |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | **FAIL at `xl` (latent)** | 0 saturated, 0 collapsed on every path of every read — validation sweeps and test reads — of all five latent corpora (per-read bands 254–383 k request / 1.26–1.61 M session cells per path); with `m` and `l`, the fail clause holds on three rungs for the latent half; the observable variant is not run |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a (induced-empty clause vacuous: no corrupted cells anywhere) | no empty prediction on any seed; shipped-vs-frozen paired ΔF1 = −0.087 ± 0.002 (request) / −0.092 ± 0.009 (session) for cli and −0.087 ± 0.002 / −0.087 ± 0.010 for cli-atomic, all p < 0.001, 0/5 wins |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | at equal frozen configurations shipped and fixed-kl agree in F1 to four decimals on every cell; the thresholded counts are identical on only 3–5 of the 10 equal-configuration cell pairs per seed and the ranking AUROC differs by up to 1.1e-4 … 7.5e-4 (the float-accumulation gap of `m` and `l`, no cell counted as corrupted); fixed vs fixed-kl +0.0001 (p = 0.12 request; p = 0.67 session) |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** (first clause on every corpus; second clause on most request cells) | lag-1 recall = 0.67–0.73× the ceiling (request, trace arms), 0.54–0.59× (session); lag-3 = 0.19–0.24× (request) and 0.18–0.21× (session) |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** (on the `xl` grid N ∈ {2, 8, 16}: the memory estimate refused N = 32) | N ≤ 8 on 36 of 90 reference cells (40 %; per seed 9 / 7 / 5 / 8 / 7) — `l` 30 %, `m` 20 %; session freezes N = 16, the top of the grid, on 42 of 45 cells; F1 is flat in N to ≤ 0.001 on the validation tables, so the top-of-grid freezes are tie-breaks to the larger N, not gains |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = +0.005 … +0.042 at the frozen checkpoints, all in regime; last / min 1.19–1.35 (the uniform 12k budget overfits every `xl` corpus; the argmin rule holds) |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | **fail at request — by saliency (p = 0.040), on the every-observed-pair cut**; pass by sign at session, with saliency not separated | core − saliency = **−0.0009 ± 0.0007 (p = 0.040, request; saliency wins 4/5)** and +0.0024 ± 0.0057 (p = 0.40, session; core wins 2/5); core − granger = +0.0057 ± 0.0017 (p = 0.002, request; 5/5) and +0.0019 ± 0.0014 (p = 0.039, session; 4/5); Shapley (sample-bounded) 0.07 below at both grains. The request failure is the reference-line note above: saliency's `p0` cut predicts 85–94 % of the observed pairs and `trace/cli-atomic` at τ = 0 — not a baseline, but above core by the same mechanism (−0.0007, p = 0.035, 0/5) — 92–96 % |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.77–0.87 (request), 0.66–0.74 (session) on every seed's frozen core read of the test split (the Granger and core columns of the same forward; token-pair `max` aggregate over the 20k / 10k sample) |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 264–294× (session; 5.9–7.4 s vs 22–26 ms per sequence); 67–69× (request) |
| H-perseq (pooled per-sequence F1 above the pooled predict-all value on every scoreable cell) | pass on 23–26 of 26 frozen cells per seed (26/26 on seed 2) | failures: saliency at request on seeds 0 and 1 (0.352–0.380 vs predict-all 0.352–0.380), `trace/cli-atomic` fixed / fixed-kl at request on seeds 1, 3 (fixed only) and 4 (0.353–0.379 vs 0.352–0.377) — the arms whose request cut is every candidate pair sit on the predict-all value, above it on some seeds by rounding; every other arm clears at both grains; scoreable fraction 0.86–0.96 (request), 0.85–0.86 (session) |

## Other findings

- **The request grain no longer orders the arms.** Every comparable arm is within 0.003 of the
  every-observed-pair reference line on every seed (0.084–0.089 on test), and the two arms that
  *are* that cut (saliency at `p0`, `trace/cli-atomic` at τ = 0) are the two on top. H-baselines
  fails at request by saliency for the second rung running (p = 0.010 at `l`, 0.040 here), and
  the `l` reading — best on the type-level graph, worst within the sequence — holds: saliency and
  cli-atomic are the arms that fail H-perseq at request. The per-sequence reading
  (`findings/rca-reading-latent.md`) now carries the `xl` column.
- **Session is where the arms separate, and `trace/core` leads.** core > cli (+0.004, p = 0.002,
  5/5), cli > cli-atomic (+0.005, p = 0.016, 5/5), core > cli-atomic (+0.009, p = 0.003, 5/5),
  core > Granger (+0.002, p = 0.039, 4/5); saliency is the unstable arm (0.079–0.107 across
  seeds, sd 0.013, above core on three seeds and 0.006–0.010 below it on the other two). The
  `l` separations (core > cli, cli > cli-atomic at session) hold with the same signs.
- **The shipped cut is nearly empty and is the largest effect anywhere**: −0.087 … −0.092 F1
  against the frozen τ (p < 0.001, 0/5 wins); 112–565 edges at request (precision 0.17–0.66,
  recall ≤ 0.0016) and 53–3,482 at session (precision 0.06–0.51, recall ≤ 0.0011). Never empty, so
  the induced-empty clause of H-cut has still never been exercised on any rung.
- **The ancestor-link column separates from the parent-link column** on all 38 scored cells of
  every seed (max |ΔF1| 0.0008), as at `m` and `l`.
- **Bidirected truth at `xl`:** 181–209 k (request) and 660–757 k (session) pairs at the default
  floor; a frozen core prediction places 11.9–12.9 k (request) and 18.7–31.2 k (session) directed
  edges on such pairs (the confounded-pair diagnostic; scored by the benchmark's rules only).
- **Per-sequence axis at session falls for the first time**: pooled F1 0.23–0.29 for the trace
  arms and Granger against 0.36–0.39 at `l`, through precision (0.13–0.19); predict-all is 0.13,
  so every arm still clears it by 0.10–0.15. Saliency (0.287) and Granger (0.276) are above core
  (0.250; core − Granger −0.027, p = 0.039, 1/5) within the trace at session.
- **Run facts:** the five discover halves ran 3 h 15 min – 3 h 36 min on a g5.2xlarge (discover
  wall clock 1.3–1.6 h request + 1.7–2.0 h session; the saliency and Shapley reads are 29–62 min
  each, the particle reads 3–5 min; peak device 7.7–8.0 GiB, resident ≤ 3.7 GiB). The five
  scoring halves ran 14 h 56 min – 20 h 02 min on an r7i.8xlarge (256 GiB): annotate 11–18.5 min per
  request cell at 30.3–35.8 GiB peak resident and 13–20 min per session cell at 69.5–81.3 GiB,
  seqscore 5.7–8.7 / 14.8–20.3 min at 26.1–30.3 / 69.6–80.1 GiB — the footprint is the
  universe-and-truth side of the scorer, identical on every arm and on the shipped cuts
  (`plans/caps.md`, addendum 2026-10-07). `--args-diff` clean on all 18 / 19 / 18 / 16 / 19
  discover-side and all 78 scoring-side records per seed; every read bound to its freeze at
  `1595ea4`; package `45ccf30` on every box; 0 corrupted cells; manifests verified (the `*.npz`,
  `prediction-*` and `ranking-*` files left in object storage).
