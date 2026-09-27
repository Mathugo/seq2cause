# Findings — s / latent / seed 0 (test read of 2026-09-27)

**Provisional: one seed.** Every pre-registered hypothesis (`plans/reference-arms.md` §6,
`plans/baselines.md` §3, `plans/per-sequence-rules.md`) is stated over the five-seed mean of the
paired per-seed difference; this file scores the single available `s` seed so the direction is on
record before seeds 1–4 land, and is superseded by the rung's `tables/` once `report` can run
(it refuses below five seeds). Source records: `results/s/latent/seed=0/` (test read
`s-latent-s0-test-26-09-27` under `freezes/2026-09-27-s-latent-s0.json`, backbone `e14f224e…`
at step 6750, ε̂ = −0.104; validation sweep `s-latent-s0-val-26-09-26`). Predict-all directed
F1 at the default floor (type-level, `2p/(n+p)` over the universe): 0.096 (request), 0.062
(session). Coverage ceiling on the test read: 0.47 (request), 0.69 (session); on the
validation sample 0.39 / 0.56 (see the registered note below).

## Registered note — the validation sample and the test read differ in size at `s`

At `xs` both jobs probed the whole split, so the test value of a frozen cell could be checked
against its validation value like for like (`findings/xs-latent.md`: −0.037 … +0.025). From `s`
upward the caps table (`plans/caps.md`, "Caps per rung") gives the validation sweep a 10,000 /
5,000 head sample and the test read 20,000 / 10,000 — the sweep re-probes its sample on 60 cells
per grain, the read once per frozen cell, so the read can afford twice the sequences inside its
cap. The consequence on this corpus: the reachable-recall ceiling rises from 0.388 to 0.474
(request) and from 0.560 to 0.690 (session) between the two jobs, and **test directed F1 sits
above the frozen validation value on every one of the 26 frozen cells, by +0.005 … +0.047**
(request +0.005 … +0.036, session −0.009 … +0.047; the one negative is saliency at session).
The frozen τ transfers — no cell lost F1 to the larger read — but the "test within validation"
comparison is a coverage effect at `s` and is not read as a generalisation check. Owner decision
2026-09-27: the sizes stay as registered (a 20k / 10k validation sweep would double the 8 h
sweeps of Job V); the asymmetry is reported here and beside every later `s` … `xl` headline,
and the ceilings of both jobs are carried in the freeze `diagnostics` and in `annotate`.

The same mechanism bounds the Shapley baseline separately: its reads take 2,000 / 500 sequences
(D-SB-14), so its coverage ceiling is 0.245 / 0.298 against 0.474 / 0.690 for every other arm.
Its F1 below is a sample-bounded number; the H-baselines line for Shapley compares a
cost-capped sample with full-sample arms, as the caps table registers.

## Type-level directed F1 at the frozen cells (test / frozen validation)

| arm | request | session |
|---|---|---|
| `trace/core` (shipped ≡ fixed-kl) | 0.290 / 0.264 | 0.352 / 0.317 |
| `trace/core` fixed | 0.299 / 0.263 | 0.353 / 0.315 |
| `trace/cli` (shipped ≡ fixed-kl) | 0.294 / 0.266 | 0.357 / 0.316 |
| `trace/cli` fixed | 0.299 / 0.266 | 0.360 / 0.313 |
| `trace/cli-atomic` (shipped ≡ fixed-kl) | 0.306 / 0.279 | 0.335 / 0.320 |
| `trace/cli-atomic` fixed | 0.309 / 0.278 | 0.335 / 0.320 |
| `baseline/granger` shipped | 0.292 / 0.266 | 0.357 / 0.314 |
| `baseline/granger` fixed | 0.293 / 0.265 | 0.354 / 0.312 |
| `baseline/saliency` | 0.269 / 0.254 | 0.277 / 0.286 |
| `baseline/shapley` (2k / 500 sample) | 0.174 / 0.169 | 0.160 / 0.154 |
| `trace/cli/*/shipped` (the tool's cut) | 0.034–0.038 (54–59 edges) | 0.048–0.056 (93–103 edges) |
| `trace/cli-atomic/*/shipped` | 0.097–0.099 (205–218 edges) | 0.125–0.128 (467–512 edges) |

Directed AUROC 0.64–0.65 (request) and 0.69–0.72 (session) on every non-Shapley arm (Shapley
0.56 / 0.58). Orientation accuracy 0.84–0.90 (request; acyclic truth) and 0.60–0.76 (session;
cyclic truth, SID/AID null with the scorer's reason). Against the five-seed xs means the trace
arms are flat with the vocabulary (0.29–0.31 vs 0.31–0.32 request; 0.34–0.36 vs 0.33–0.36
session) while predict-all fell from 0.21 to 0.10 / 0.06, so the lift over the trivial
predictor grew.

## Hypotheses

| hypothesis | this seed | numbers |
|---|---|---|
| H-estimator (core vs cli at the frozen cut, \|ΔF1\| ≤ 0.05) | pass | Δ = −0.004 (request), −0.005 (session) |
| H-construction (cli-atomic lag ≥ 2 recall ≥ 2× cli's where cli's < 0.2) | **fail** at lag 2; the advantage appears from lag 3 | request: cli lag-2 0.210 / lag-3 0.098 / lag-4 0.048, cli-atomic 0.210 / 0.134 / 0.080 (1.0× / 1.4× / 1.7×); session: cli 0.248 / 0.101 / 0.025, cli-atomic 0.289 / 0.188 / 0.109 (1.2× / 1.9× / 4.4×) — at xs the ratio was 0.8–1.4× at every lag |
| H-hazard (corrupted cells on the shipped path, rung ≥ m) | n/a at s | 0 saturated, 0 collapsed on 226,951 (request) / 1.6 M (session) cells per path on every read |
| H-cut (an empty shipped edge set where corrupted cells exist) | n/a at s | no corrupted cells; no empty prediction; shipped-vs-frozen ΔF1 −0.26 … −0.31 (cli), −0.21 (cli-atomic); the cut's precision is 0.41–0.77 and its recall 0.02–0.08 |
| H-fix (fixed-kl ≥ shipped − 0.01) | pass | bit-identical edge sets on every cell (no corrupted cell to fix); fixed vs fixed-kl within +0.009 (core request), +0.005 (cli request), ≤ 0.003 elsewhere |
| H-lag (lag-1 recall ≥ 0.8× the ceiling; lag ≥ 3 < 0.2×) | **fail** | lag-1 recall = 0.48–0.62× the ceiling (request), 0.49–0.56× (session); lag-3 = 0.14–0.28× (core and cli below 0.2×, cli-atomic and saliency above) |
| H-sat (frozen N ≤ 8 on ≥ 80 % of reference cells; F1(N = 32) − F1(N = 8) < 0.01 there) | **fail** | N ≤ 8 on 4 of 18 reference cells (request 4/9: core 2, cli-atomic 8; session 0/9 — every session cell froze at N = 32) |
| H-regime (ε̂ < 0.1, reported only) | pass | ε̂ = −0.104 at the frozen checkpoint (step 6750; order-2 plug-in floor 1.675 vs validation loss 1.247); last checkpoint 1.015× the minimum |
| H-baselines (every baseline below `trace/core/shipped/frozen`) | **fail** by sign (Granger) | request: core 0.290 vs granger 0.292 (−0.002), saliency 0.269, shapley 0.174; session: core 0.352 vs granger 0.357 (−0.005), saliency 0.277, shapley 0.160 — the xs five-seed tie (+0.001 / +0.003, n.s.) leans the other way on this seed |
| H-granger-agree (top-10 % Jaccard ≥ 0.5 on the same forward) | pass | 0.91 (request c1 N32, the frozen Granger cell), 0.84 (c1 N2, the frozen core cell), 0.89 (session c2 N32), 0.80 (c2 N2); validation matrices, `max` aggregate |
| H-shapley-cost (≥ 5× the core probe per sequence, session) | pass | 5.9 s vs 0.027 s per sequence (≈ 218×); request 0.87 s vs 0.013 s |
| H-perseq (pooled per-sequence F1 above the pooled predict-all value on every scoreable cell) | **fail (request)** / pass on the reference arms (session) | request (predict-all 0.449): cli-atomic 0.531 > 0.449; core 0.365, cli 0.395, granger 0.388, saliency 0.213, shapley 0.345 (vs 0.446) all below; session (predict-all 0.187): core 0.364, cli 0.369, cli-atomic 0.484, granger 0.349, shapley 0.212 (vs 0.202) above; saliency 0.156 below; scoreable fraction 0.91 (request) / 0.98–0.99 (session) |

## Other findings

- **The three TRACE readings and Granger stay one method at `s`** at the type level: core, cli
  and Granger differ by ≤ 0.005 at both grains. `cli-atomic` is the odd one out in opposite
  directions — best at request (+0.012 over cli) and worst at session (−0.022) — and is the only
  arm whose per-sequence pooled F1 clears predict-all at request.
- **The shipped cut remains the largest effect**: −0.21 … −0.31 F1 against the frozen τ, with
  54–512 edges; never empty (no corrupted cells to induce one), so H-cut's induced-empty clause
  stays untested until rung `m`.
- **The ancestor-link column equals the parent-link column on all 38 cells**
  (`plans/per-sequence-rules.md`): no transitive target edge survives the filter on this corpus.
- **Short-sequence skips at the deeper contexts:** the session Shapley read probed 496 of its
  500-sequence sample and `cli-atomic` at context 4 probed 9,865 of 10,000; the frozen cells at
  context 2 probed every sequence.
- **Run facts:** 17 discover reads, 38 annotate, 38 seqscore; 3 h 47 min on one A10G (discover
  204 min — saliency 30 + 44 min, Shapley 29 + 49 min, particle reads 2.6–4.7 min each; annotate
  6.8 min in total under the pinned scorer fix; seqscore 9.3 min); peak 2.89 GB; `--args-diff`
  clean on all 95 records; package commit `4a2bae6`; records verified against the manifest (the
  34 `*.npz` left in object storage).
