# Findings — l / latent, five seeds (test reads of 2026-10-03)

Every pre-registered hypothesis (`plans/reference-arms.md` §6, `plans/baselines.md` §3,
`plans/per-sequence-rules.md` §6) is scored here on the five-seed mean of the paired per-seed
difference, as written; the tables the numbers come from are `tables/l-latent/` (`report`, 38
cells × 5 seeds, 12 pre-registered pairs, per-sequence document). Source records:
`results/l/latent/seed={0..4}/` — discover reads `l-latent-s0-test-26-10-02` and
`l-latent-s{1..4}-test-26-10-03` under `freezes/2026-10-0{2,3}-l-latent-s{k}.json`, scored by
`l-latent-s{0..4}-annotate-26-10-03` (the scoring half, run again after a harness fix — see
*Run facts*); validation sweeps `l-latent-s0-val-26-09-30` and `l-latent-s{1..4}-val-26-10-02`.
Backbones: argmin-validation checkpoints at steps 1250 / 1750 / 750 / 1500 / 3000,
ε̂ = +0.036 / +0.045 / +0.063 / +0.042 / +0.055, all in regime.

Every arm on every seed: 0 corrupted cells (saturated or collapsed) on every path of every read
— validation and test. Predict-all directed F1 at the default floor (type-level, `2p/(n+p)` over
the universe): 0.013 / 0.011 / 0.011 / 0.010 / 0.011 (request), 0.006 / 0.005 / 0.005 / 0.005 /
0.005 (session). Coverage ceiling on the test read: 0.107–0.125 (request), 0.163–0.189
(session).

## Registered note — the validation sample and the test read differ in size at `l`

Carried from `findings/s-latent.md`, `findings/m-latent.md` and `plans/caps.md` (addendum
2026-09-27; owner decision): the validation sweep probes a 10k / 5k head sample and the test
read 20k / 10k, so the two jobs see different coverage and the test F1 of a frozen cell lands
**above** its frozen validation value as a coverage effect, not a generalisation check. On the
five seeds the test-minus-frozen-validation delta is **−0.003 … +0.048** across all 130 frozen
cells (every negative is a Shapley cell, which probes its own capped sample both times); the
ceiling rises 0.066–0.076 → 0.107–0.125 (request) and 0.102–0.120 → 0.163–0.189 (session)
between the two jobs. At `l` the note applies with more force than at `m`: the universe is
11.3–14.0 M (request) / 25.6–30.6 M (session) ordered pairs, the validation sample reaches
7–12 % of the truth edges and the test read 11–19 %, so both jobs are ceiling-bound and the
frozen τ sits at or near the bottom of its grid (recorded grid-edge cells: two on seed 0, five
on seed 2). The Shapley baseline's own 2k / 500 sample bounds its ceiling at ≈ 0.03 at both
grains — a quarter of the other arms' — so its rows below are sample-bounded.

## Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≈ fixed-kl) | 0.135 ± 0.006 | 0.161 ± 0.011 |
| `trace/core` fixed | 0.135 ± 0.006 | 0.161 ± 0.011 |
| `trace/cli` (shipped ≈ fixed-kl) | 0.135 ± 0.007 | 0.155 ± 0.013 |
| `trace/cli` fixed | 0.135 ± 0.006 | 0.154 ± 0.012 |
| `trace/cli-atomic` (shipped ≈ fixed-kl) | 0.133 ± 0.006 | 0.148 ± 0.014 |
| `trace/cli-atomic` fixed | 0.133 ± 0.005 | 0.148 ± 0.015 |
| `baseline/granger` shipped | 0.134 ± 0.006 | 0.161 ± 0.007 |
| `baseline/granger` fixed | 0.134 ± 0.006 | 0.161 ± 0.007 |
| `baseline/saliency` | **0.139 ± 0.007** | 0.153 ± 0.005 |
| `baseline/shapley` (2k / 500 sample) | 0.035 ± 0.003 | 0.038 ± 0.004 |
| `trace/cli/*/shipped` (the tool's cut) | 0.006 ± 0.003 | 0.003 ± 0.002 |
| `trace/cli-atomic/*/shipped` | 0.006 ± 0.004 | 0.006 ± 0.002 |

Directed AUROC 0.55–0.56 (request) and 0.57–0.58 (session) on the non-Shapley arms, down from
0.58–0.59 / 0.64–0.65 at `m`. Orientation accuracy 0.93–0.97 (request; acyclic truths) and
0.72–0.88 (session; cyclic truths on every seed, SID/AID null with the scorer's ADMG reason on
every cell). Against `m` every type-level F1 fell (request 0.20 → 0.135, session 0.27 → 0.16)
while predict-all fell faster (0.025 → 0.011, 0.014 → 0.005), so the lift over the trivial
predictor keeps growing with the vocabulary.

## Hypotheses (five-seed paired means; p from the paired t-test, n = 5)

| hypothesis | verdict | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | core − cli = +0.001 ± 0.002 (p = 0.36, request), **+0.006 ± 0.004 (p = 0.022, session; core wins 5/5)** — inside the band, and the sign has turned: at `m` and `s` cli was significantly above core at session |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail at every lag on every seed; at session the ratio is now below 1 on three seeds** | request: lag-2 … lag-5 ratio 0.99–1.22×; session: 0.76–1.02× on seeds 0–3 and 1.07–1.21× on seed 4 — the lag gradient of `s` (2.9× / 3.7× at lag 4) and the flattened one of `m` (≤ 2.1×) are gone |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | **FAIL at `l` (latent)** | 0 saturated, 0 collapsed on every path of every read — validation sweeps and test reads — of all five latent corpora (per-read bands 225 k–433 k request / 1.3–2.2 M session cells per path); with `m`, the fail clause holds on two rungs for the latent half; the observable variant is not run |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a (induced-empty clause vacuous: no corrupted cells anywhere) | no empty prediction on any seed; shipped-vs-frozen paired ΔF1 = −0.129 ± 0.005 (request) / −0.152 ± 0.014 (session) for cli and −0.126 ± 0.005 / −0.141 ± 0.014 for cli-atomic, all p < 0.001, 0/5 wins |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | at equal frozen configurations shipped and fixed-kl agree in F1 to four decimals on every cell; the thresholded counts are no longer always identical (they differ on 11 of the 27 equal-configuration cell pairs) and the ranking AUROC differs by ≤ 3e-8 on core and by up to 5e-4 on the cli paths — larger than `m`'s 4e-9, with no cell counted as corrupted; three request cells froze different configurations on the two paths (seed 0 cli-atomic; seed 2 cli and cli-atomic), \|ΔF1\| ≤ 0.0014; fixed vs fixed-kl null (−0.000, p = 0.32 request; +0.000, p = 0.53 session) |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** (first clause on every corpus; second clause on most request cells) | lag-1 recall = 0.69–0.76× the ceiling (request, trace arms), 0.49–0.63× (session); lag-3 = 0.18–0.25× (request) and 0.14–0.20× (session) |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** | N ≤ 8 on 27 of 90 reference cells (30 %; per seed 7 / 4 / 5 / 6 / 5) — `m` 20 %, `s` 28 %; session freezes N = 32 on 39 of 45 cells |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = +0.036 … +0.063 at the frozen checkpoints, all in regime but positive for the first time since `xs`; last / min 1.15–1.23 (the uniform 12k budget overfits every `l` corpus; the argmin rule holds) |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | **fail at request — by saliency, significantly**; split at session | core − saliency = **−0.003 ± 0.002 (p = 0.010, request; saliency wins 5/5)** and +0.008 ± 0.007 (p = 0.050, session; core wins 4/5); core − granger = +0.002 ± 0.001 (p = 0.042, request; core wins 5/5 — the first significant core-over-Granger gap at request) and −0.000 ± 0.004 (p = 1.00, session; 3/5); Shapley (sample-bounded) stays 0.10–0.12 below |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.84–0.90 (request), 0.68–0.79 (session) on every seed's frozen configs; validation matrices, `max` aggregate |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 188–217× (session; 5.5–7.2 s vs 29–38 ms per sequence); 60–72× (request) |
| H-perseq (pooled per-sequence F1 above the pooled predict-all value on every scoreable cell) | pass on 25 of 26 frozen cells on four seeds, 26 of 26 on seed 3 | the one failure is saliency at request on seeds 0, 1, 2, 4 (0.35–0.43 vs predict-all 0.38–0.43); every other arm clears at both grains; scoreable fraction 0.94–0.98 (request), 0.89–0.91 (session) |

## Other findings

- **Saliency is the best type-level arm at request on every seed.** The gap is small (+0.003)
  but paired and one-signed (p = 0.010), and it is the first significant H-baselines failure
  that does not involve Granger. At session saliency falls back below core and Granger. On the
  per-sequence axis the same arm is the only one that fails H-perseq at request — best on the
  type-level graph, worst within the sequence.
- **Core, cli, cli-atomic and Granger are within 0.003 at request and 0.013 at session.** The
  stable separations at `l` are core > cli-atomic (+0.003, p = 0.039 request; +0.013, p = 0.007
  session), cli > cli-atomic at session (+0.007, p = 0.002, 5/5) and core > cli at session
  (+0.006, p = 0.022, 5/5). `cli-atomic` is the worst trace arm at both grains, as at `m`, and
  again the best trace arm per sequence on four of five seeds at request and three of five at
  session.
- **The shipped cut remains the largest effect anywhere and is now close to empty in recall**:
  −0.13 … −0.15 F1 against the frozen τ (p < 0.001, 0/5 wins), recall 0.0005–0.005. Its size is
  unstable at session: 101–485 edges on seeds 0 and 4 (precision 0.37–0.78) against
  1,802–5,412 on seeds 1–3 (precision 0.04–0.08). Never empty, so the induced-empty clause of
  H-cut has still never been exercised.
- **The ancestor-link column separates from the parent-link column** on all 38 scored cells of
  seeds 0, 1, 2, 4 and on 36 of 38 on seed 3 (max |ΔF1| 0.003), as at `m`.
- **Bidirected truth at `l`:** 50–61 k (request) and 179–219 k (session) pairs at the default
  floor; a frozen core prediction places 6.7–8.5 k (request) and 7.3–13.5 k (session) directed
  edges on such pairs (the confounded-pair diagnostic; scored by the benchmark's rules only).
- **Run facts:** the five test reads finished discover in 3.1–3.7 h of probe wall clock on a
  g5.2xlarge (peak device memory 6.4–6.7 GB) and were stopped in annotate (chains exit 143,
  synced): the confounded-pair count rebuilt the truth pair set per predicted edge — 147–246 min
  per frozen request cell (`plans/caps.md`, addendum 2026-10-03). After the fix the scoring half
  ran from the synced reads: a first attempt on a 16 GB host was killed for memory on the first
  session cell (exit 137; all 19 request cells done); the second, on a 32 GB CPU host, completed
  in 2 h 38 min – 3 h 15 min (annotate 82–105 min, seqscore 72–91 min, peak resident memory
  16.0–18.9 GB). `--args-diff` clean on all 17 / 17 / 19 / 18 / 19 discover-side records and on
  all 78 scoring-side records per seed; freezes bound at `9d35928` (seed 0) / `318bdb5` (seeds
  1–4); package commits `36d868d` / `cd99c3a` (discover) and `575ba6b` (scoring); 0 corrupted
  cells; manifests verified (the `*.npz` left in object storage).
