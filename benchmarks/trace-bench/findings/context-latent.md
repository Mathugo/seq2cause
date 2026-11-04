# Findings — the context family (`plans/reference-arms.md`, addendum 2026-10-31), latent, five seeds

The hypotheses of the addendum's §8 are scored here per rung on the five-seed mean of the paired
per-seed difference, as written, by `contexttable` (`tables/context-latent/<rung>/`: the family's
cells beside the landed `trace/core/fixed-kl/frozen`, `trace/cli-atomic/fixed-kl/frozen` and
`trace/cli-atomic/fixed-kl/shipped` cells of `results/`; the registered pairs through
`report.paired_table`; the τ drift from a test-side `scoresweep` over every test read's score file)
and by `report` on the family's own tree (`tables/<rung>-latent-context/`: 16 cells × 5 seeds, 14
pairs, the per-sequence document; the landed reference and floor cells are "not in this family",
`tables/reasons-context-family.json`). Source records: `results-context/<rung>/latent/seed={0..4}/`
— the cells, `test-read/` and `val-sweep/` under `freezes/<date>-<rung>-latent-s<k>-context.json`
(`freeze@3`, `family: context`; the family's two validation tables per grain merged under D-SB-24).
Code: seq2cause v0.3.1 (`280ed948`, D-SB-19) through the harness's engine adapter (D-SB-20 … 23);
scorer trace-bench `a3eb104` (`v0.3.0-score3`); corpora `chadyuk/trace-bench@v0.3.0`. The paper's
results are frozen (2026-10-29); this family runs in the harness only and the paper decides later.

Verdicts are mechanical (`contexttable`'s `verdicts`): a hypothesis fails only by its registered
fail clause; one whose inputs are absent is "not computed", never passed. H-hazard is not scored
for the family (§6 a: the CMI columns carry no corrupted-cell statistics of their own; the
information-gain paths of every pass had 0 corrupted cells on every read). The root-cause axes
(H-ctx-keep-context-share and the `plans/rca-arms.md` hypotheses of 2026-10-31) are scored in
`benchmarks/trace-bench-rca/findings/` when those rows land.

## xs (test reads `xs-latent-s{0..4}-ctx-test-26-10-31` under `freezes/2026-11-01-xs-latent-s{k}-context.json`)

Validation sweeps `xs-latent-s{k}-ctx-val-26-10-31` (N = 8 on the landed seed-k backbones; whole
validation split, 1,154–1,229 request / 433–458 session sequences), test reads on the whole test
split (1,223–1,304 request / 443–487 session sequences; 8 / 9 / 8 / 8 / 9 discover reads per seed,
the information-gain and CMI cells of one layout sharing a read where they froze at one c). Every
read: 0 corrupted cells on the information-gain paths; `context_mode` and `window_offset` as frozen;
the freeze bound at its commit (`0e0e118e`, #77). Identity checks (i)–(iii) of §7 passed on the
validation sweeps before the freeze (RUN.md rows of 2026-10-31).

### Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| cell (`fixed-kl` path unless stated) | request | session |
|---|---|---|
| `trace/core-ig-overwrite` (c 2 / 1 / 1 / 2 / 2 at request; 2 / 2 / 2 / 4 / 4 at session) | 0.317 ± 0.045 | 0.354 ± 0.028 |
| `trace/core-cmi-overwrite` (c 2 / 1 / 1 / 2 / 2; 2 / 4 / 2 / 4 / 2) | 0.313 ± 0.050 | 0.338 ± 0.028 |
| `trace/core-ig-keep` (c = 1, `none`) | 0.311 ± 0.043 | 0.361 ± 0.024 |
| `trace/core-cmi-keep` (c = 1, `none`) | 0.314 ± 0.049 | 0.345 ± 0.029 |
| `trace/cli-atomic-cmi-keep` fixed-kl | 0.304 ± 0.037 | 0.315 ± 0.032 |
| `trace/cli-atomic-cmi-keep` fixed | 0.304 ± 0.038 | 0.315 ± 0.032 |
| `trace/cli-atomic-cmi-keep` fixed-kl, shipped cut | 0.061 ± 0.013 | 0.079 ± 0.029 |
| landed `trace/core` fixed-kl (c 2 / 1 / 1 / 2 / 1, N 2 / 8 / 2 / 2 / 2 at request) | 0.321 ± 0.043 | 0.357 ± 0.021 |
| landed `trace/cli-atomic` fixed-kl | 0.307 ± 0.043 | 0.332 ± 0.011 |
| landed `trace/cli-atomic` fixed-kl, shipped cut | 0.048 ± 0.036 | 0.073 ± 0.028 |

Per seed (request / session), information gain overwrite: 0.308 / 0.348, 0.336 / 0.355,
0.328 / 0.354, 0.245 / 0.315, 0.367 / 0.395; CMI overwrite: 0.290 / 0.333, 0.351 / 0.348,
0.319 / 0.342, 0.239 / 0.295, 0.364 / 0.372; information gain keep: 0.306 / 0.358, 0.336 / 0.360,
0.328 / 0.363, 0.240 / 0.329, 0.347 / 0.397; CMI keep: 0.290 / 0.338, 0.351 / 0.344, 0.319 / 0.340,
0.243 / 0.310, 0.363 / 0.391; atomic CMI (fixed-kl): 0.312 / 0.319, 0.329 / 0.319, 0.312 / 0.324,
0.239 / 0.263, 0.328 / 0.350. At request the keep and overwrite cells coincide where the overwrite
cell froze at c = 1 (seeds 1 and 2: identity check (iii) on the test read, the same numbers).

### Hypotheses (paired per-seed differences, five-seed mean, paired t-test)

| hypothesis | request | session | verdict |
|---|---|---|---|
| **H-ctx-cmi-f1** (\|CMI − IG\| ≤ 0.01 at matched context) | overwrite −0.004 (p 0.51); keep +0.002 (p 0.73) | overwrite **−0.016 (p 0.006)**; keep **−0.017 (p 0.005)** | request **pass**; session **fail** |
| **H-ctx-cmi-tau** (CMI's mean \|τ drift\| ≤ IG's, within one grid step) | overwrite 0.50 vs 0.60; keep 0.59 vs 0.81 | overwrite 0.30 vs 0.38; keep 0.19 vs 0.11 (+0.08, under the half-decade) | **pass** on both grains, both layouts |
| **H-ctx-keep-perseq** (keep's per-sequence pooled F1 ≥ overwrite's at session) | not applicable | IG −0.003 (p 0.82); CMI −0.014 (p 0.22) — neither below by > 0.01 with p < 0.05 | **pass** |
| **H-atomic-cmi-f1** (atomic CMI ≥ landed `trace/cli-atomic` fixed-kl) | −0.003 (p 0.76) | −0.017 (p 0.22) | **pass** (no significant deficit) |
| **H-atomic-cmi-tau** (atomic CMI's τ drift ≤ the landed atomic arm's F1 drift) | 0.70 vs F1 drift −0.009 | 0.10 vs −0.013 | **pass** |
| **H-atomic-cmi-shipped** (shipped cut non-empty, F1 ≥ the landed shipped cut) | +0.013 (p 0.50), non-empty on 5 / 5 | +0.005 (p 0.44), non-empty on 5 / 5 | **pass** |
| H-ctx-keep-context-share (per-trace root-cause axis) | — | — | not computed (root-cause rows pending) |

**H-ctx-cmi-f1 fails at the session grain on xs:** the CMI cell sits 0.016–0.017 under the
information-gain cell at matched context on every seed (overwrite: −0.015 / −0.007 / −0.012 / −0.020
/ −0.023; keep: −0.020 / −0.016 / −0.023 / −0.019 / −0.006), beyond the registered 0.01 with
p < 0.01. At request the two scores are within 0.004 (the validation sweeps already showed the
session gap: 0.02–0.04 on every seed at the frozen cells). The CMI's bounded scale is not the
cause of the gap (its τ drift is the smaller of the two at session); the gap sits in precision
and orientation (session orientation accuracy 0.53–0.57 for the CMI cells against 0.62–0.64 for
the information-gain cells; skeleton F1 within 0.01).

### Descriptive (not hypotheses)

- **Keep vs overwrite at matched score.** Session: information gain +0.008 (p 0.021; keep above
  overwrite on 5 / 5 seeds), CMI +0.007 (p 0.22); request: −0.006 (p 0.21) and +0.001. The
  session `none` layout tests every observed token after BOS (c = 1) and loses nothing against the
  c = 2 / 4 overwrite cells — the context effect of the family is a small session-grain gain.
- **The atomic CMI arm** reads 0.304 / 0.315 (request / session), 0.003 / 0.017 under the landed
  atomic information-gain arm (n.s.) and 0.010 / 0.030 under the core CMI keep cell (request p
  0.37; session p 0.008 on the type-level table's own pair); its `fixed` path (the specials
  excluded from the model draw) equals its `fixed-kl` path within 0.0003 on every seed and grain.
  Its shipped cut (the tool's pooled-percentile rule on the CMI scores) keeps 29–146 edges per
  seed: F1 0.061 / 0.079, above the landed atomic shipped cut by +0.013 / +0.005.
- **Grid edges.** The atomic CMI arm froze at the bottom of the τ grid at request on seeds 0 and
  2 (τ 1e-7); every other cell is interior. The test-side argmax lies within one grid step of the
  frozen τ on 48 / 60 frozen cells (mean |drift| 0.10–0.81 decades); at request the overwrite cells
  drift most (0.50–0.60 decades) and their F1 moves by ≤ 0.006.
- **Against the landed `trace/core` cell** (frozen at its own (c, N) over the landed N grid): the
  family's information-gain overwrite cell at N = 8 reads −0.005 / −0.003 (p 0.09 / 0.50) — the N
  and τ of the two freezes differ; the validation rows at N = 8 were identical (identity (i)).
- **Per-sequence axis** (`tables/xs-latent-context/xs-latent-perseq.md`): pooled directed F1
  0.41–0.44 at request and 0.39–0.42 at session for the core cells, 0.44 / 0.39 for the atomic
  arm (the highest request value: it tests cause 0, which the core staircase never does, and its
  pooled recall is 0.88); scoreable fraction 0.94 / 1.00 for the keep cells against 0.82 / 0.97
  for the overwrite cells (every observed token is a candidate under c = 1).

## s (test reads `s-latent-s{0..4}-ctx-test-26-11-01` under `freezes/2026-11-01-s-latent-s{k}-context.json`)

Validation sweeps `s-latent-s{k}-ctx-val-26-11-01` (N = 8 on the landed seed-k backbones; the 10k / 5k
validation sample of `plans/caps.md`, 8,894–10,000 request / 4,466–5,000 session sequences probed — the
c ≥ 2 overwrite cells skip the sequences shorter than their window), test reads on the 20k / 10k test
sample (19,735–20,000 request / 9,735–10,000 session sequences probed; 9 / 8 / 9 / 9 / 9 discover reads
per seed). Every read: 0 corrupted cells on the information-gain paths; `context_mode` and
`window_offset` as frozen; the freeze bound at its commit (`c9374ed9`, #79); each read staged on the xs
context test read of its seed, grain and probe. Identity checks (i)–(iii) of §7 passed on the
validation sweeps before the freeze (RUN.md rows of 2026-11-01). No cell froze at a grid edge. At
request every information-gain overwrite cell froze at c = 1, so the information-gain keep and
overwrite cells are the same cell read twice (identity (iii) holds on the test read: the same numbers
on every seed); the CMI overwrite cell froze at c = 2 on seeds 2 and 4, within 0.002 of its keep cell.
The context effect of the family is a session-grain effect at s as at xs. The registered sample-size note applies: the
test sample is twice the validation sample and every frozen cell reads 0.01–0.05 above its validation
F1.

### Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| cell (`fixed-kl` path unless stated) | request | session |
|---|---|---|
| `trace/core-ig-overwrite` (c = 1 on every seed at request; 2 / 2 / 2 / 2 / 4 at session) | 0.290 ± 0.012 | 0.334 ± 0.017 |
| `trace/core-cmi-overwrite` (c 1 / 1 / 2 / 1 / 2; 4 / 2 / 2 / 4 / 4) | 0.263 ± 0.012 | 0.303 ± 0.017 |
| `trace/core-ig-keep` (c = 1, `none`) | 0.290 ± 0.012 | 0.341 ± 0.012 |
| `trace/core-cmi-keep` (c = 1, `none`) | 0.264 ± 0.011 | 0.305 ± 0.013 |
| `trace/cli-atomic-cmi-keep` fixed-kl | 0.263 ± 0.013 | 0.289 ± 0.015 |
| `trace/cli-atomic-cmi-keep` fixed | 0.263 ± 0.013 | 0.289 ± 0.015 |
| `trace/cli-atomic-cmi-keep` fixed-kl, shipped cut | 0.080 ± 0.013 | 0.128 ± 0.017 |
| landed `trace/core` fixed-kl (c = 1, N 2 / 8 / 32 / 8 / 2 at request; c 2 / 2 / 4 / 4 / 2, N 32 at session) | 0.289 ± 0.013 | 0.342 ± 0.018 |
| landed `trace/cli-atomic` fixed-kl | 0.287 ± 0.016 | 0.324 ± 0.016 |
| landed `trace/cli-atomic` fixed-kl, shipped cut | 0.093 ± 0.015 | 0.149 ± 0.030 |

Per seed (request / session), information gain overwrite: 0.295 / 0.341, 0.307 / 0.353, 0.284 / 0.337,
0.291 / 0.333, 0.274 / 0.306; CMI overwrite: 0.274 / 0.319, 0.276 / 0.319, 0.255 / 0.305, 0.263 / 0.282,
0.249 / 0.291; information gain keep: 0.295 / 0.336, 0.307 / 0.346, 0.284 / 0.338, 0.291 / 0.357,
0.274 / 0.326; CMI keep: 0.274 / 0.310, 0.276 / 0.321, 0.257 / 0.303, 0.263 / 0.304, 0.249 / 0.287;
atomic CMI (fixed-kl): 0.271 / 0.306, 0.278 / 0.296, 0.266 / 0.292, 0.251 / 0.264, 0.248 / 0.287;
landed `trace/cli-atomic`: 0.306 / 0.335, 0.301 / 0.343, 0.275 / 0.306, 0.278 / 0.322, 0.273 / 0.312.

### Hypotheses (paired per-seed differences, five-seed mean, paired t-test)

| hypothesis | request | session | verdict |
|---|---|---|---|
| **H-ctx-cmi-f1** (\|CMI − IG\| ≤ 0.01 at matched context) | overwrite **−0.027 (p < 0.001)**; keep **−0.026 (p < 0.001)** | overwrite **−0.031 (p 0.006)**; keep **−0.035 (p 0.002)** | **fail** on both grains |
| **H-ctx-cmi-tau** (CMI's mean \|τ drift\| ≤ IG's, within one grid step) | overwrite 0.50 vs 0.31 (+0.19 decade); keep 0.40 vs 0.31 | overwrite 0.20 vs 0.38; keep 0.29 vs 0.19 (+0.10) | **pass** (every excess under the half-decade) |
| **H-ctx-keep-perseq** (keep's per-sequence pooled F1 ≥ overwrite's at session) | not applicable | IG −0.004 (p 0.08); CMI −0.003 (p 0.21) — neither below by > 0.01 | **pass** |
| **H-atomic-cmi-f1** (atomic CMI ≥ landed `trace/cli-atomic` fixed-kl) | **−0.024 (p 0.005)** | **−0.035 (p 0.011)** | **fail** on both grains |
| **H-atomic-cmi-tau** (atomic CMI's drift ≤ the landed atomic arm's F1 drift) | F1 drift 0.014 vs 0.022 (τ drift 0.40 decade) | 0.014 vs 0.015 (0.30) | **pass** |
| **H-atomic-cmi-shipped** (shipped cut non-empty, F1 ≥ the landed shipped cut) | **−0.013 (p < 0.001)**, non-empty 5 / 5 | −0.022 (p 0.19), non-empty 5 / 5 | request **fail**; session pass |
| H-ctx-keep-context-share (per-trace root-cause axis) | — | — | not computed (root-cause rows pending) |

**H-ctx-cmi-f1 fails on both grains at s** (it failed at session only on xs): the CMI cell sits under
the information-gain cell at matched context on every seed and grain (request: −0.021 / −0.031 /
−0.029 / −0.028 / −0.025 overwrite, the same within 0.002 for keep; session: −0.022 / −0.034 / −0.032 /
−0.051 / −0.015 overwrite, −0.026 / −0.025 / −0.035 / −0.053 / −0.039 keep), two to three times the
registered 0.01. The validation sweeps showed the same gap at the frozen cells (0.01 at request,
0.03–0.04 at session). The gap sits in precision: at request the CMI cells predict more edges (recall
0.314–0.323 against 0.306) at precision 0.225–0.229 against 0.277; at session precision 0.266–0.272
against 0.283–0.308 with skeleton F1 0.344–0.345 against 0.362–0.368 and orientation accuracy
0.71–0.72 against 0.73–0.75. The CMI's τ drift is the smaller of the two at session (H-ctx-cmi-tau
passes), so the bounded scale is not what costs it.

**H-atomic-cmi-f1 fails at s** (it passed on xs): the atomic CMI arm reads 0.263 / 0.289 against the
landed atomic information-gain arm's 0.287 / 0.324, below on every seed (request −0.035 / −0.023 /
−0.009 / −0.027 / −0.025; session −0.029 / −0.047 / −0.014 / −0.058 / −0.025). The landed arm freezes
its own (c, N) over the landed grid (N = 32 at session on every seed, c 2–4; at request c 1–2, N 2–32);
the family's arm reads c = 1, N = 8 by registration, so the pair confounds the score with the particle count — the landed H-sat band (F1 flat
in N from N = 8) argues against N as the cause, and the atomic CMI arm's own τ drift is the smaller
(H-atomic-cmi-tau passes). Its shipped cut (the pooled-percentile rule on the CMI scores; 235–382 edges
at request, 566–6,203 at session) reads 0.080 / 0.128 against the landed shipped cut's 0.093 / 0.149:
below by 0.013 with p < 0.001 at request (**H-atomic-cmi-shipped fails there**), by 0.022 with p 0.19 at
session (pass).

### Descriptive (not hypotheses)

- **Keep vs overwrite at matched score.** Request: identical cells on the information gain (c = 1
  everywhere); CMI +0.0004 (p 0.21). Session: information gain +0.007 (p 0.37; keep above overwrite on
  3 / 5 seeds, +0.024 / +0.020 on seeds 3 and 4, where overwrite froze at c = 2 / c = 4), CMI +0.002
  (p 0.72). The session `none` layout (c = 1,
  every observed token tested) loses nothing against the overwrite cells; the keep cells are the less
  dispersed (sd 0.012 / 0.013 against 0.017).
- **Against the landed `trace/core` cell:** the family's information-gain overwrite cell at N = 8 reads
  +0.001 (p 0.54) at request and −0.008 (p 0.048) at session, where the landed cell froze at N = 32
  — a particle-count effect at the edge of significance, inside the landed H-sat band.
- **The atomic arm's two paths** (`fixed` = the specials excluded from the model draw) agree within
  0.0002 on every seed and grain, as on xs.
- **Grid edges and drift.** No cell froze at a grid edge. The test-side argmax τ lies within one grid
  step of the frozen τ on 58 / 60 frozen cells (two drift a full decade: the CMI overwrite and keep
  cells of seed 2 at request); mean |drift| 0.19–0.50 decades. The five-seed mean test F1 exceeds the
  validation F1 by 0.013–0.037 on every frozen cell (positive on 58 / 60 per-seed cells; the double
  test sample), most on the information-gain cells.
- **Per-sequence axis** (`tables/s-latent-context/s-latent-perseq.md`): pooled directed F1 0.35–0.37
  at request and 0.31–0.34 at session for the core cells; the atomic CMI arm 0.445 / 0.384 (pooled
  recall 0.657 / 0.472 — it tests cause 0, which the core staircase never does); scoreable fraction
  0.965 / 0.995 for the keep cells against 0.934 / 0.988 for the overwrite cells. Per-sequence AUROC
  0.63 / 0.71 (core), 0.66 / 0.77 (atomic).

## m (test reads `m-latent-s{0..4}-ctx-test-26-11-01` under `freezes/2026-11-01-m-latent-s{k}-context.json`)

Validation sweeps `m-latent-s{k}-ctx-val-26-11-01` (N = 8 on the landed seed-k backbones; the 10k / 5k
validation sample, 8,816–10,000 request / 4,241–4,988 session sequences probed), test reads on the
20k / 10k test sample (8 discover reads per seed: the information-gain and CMI cells of each layout
froze at one c on every seed). Every read: 0 corrupted cells on the information-gain paths;
`context_mode` and `window_offset` as frozen; the freeze bound at its commit (`c19d08f6`, #80); each
read staged on the s context test read of its seed, grain and probe. Identity checks (i)–(iii) of §7
passed on the validation sweeps before the freeze (all 2,160 landed N = 8 rows of the m tables
reproduced; RUN.md rows of 2026-11-01). No cell froze at a grid edge. Every request cell froze at
c = 1 and every session core overwrite cell at c = 2 (g = 2), so the keep and overwrite cells of one
score are the same read at request and differ only by the context layout at session. The
sample-size note applies: every frozen cell reads 0.03–0.07 above its validation F1.

### Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| cell (`fixed-kl` path unless stated) | request | session |
|---|---|---|
| `trace/core-ig-overwrite` (c = 1 at request; c = 2 at session, every seed) | 0.204 ± 0.009 | 0.268 ± 0.010 |
| `trace/core-cmi-overwrite` (c = 1; c = 2) | 0.184 ± 0.009 | 0.224 ± 0.013 |
| `trace/core-ig-keep` (c = 1, `none`) | 0.204 ± 0.009 | 0.268 ± 0.009 |
| `trace/core-cmi-keep` (c = 1, `none`) | 0.184 ± 0.009 | 0.223 ± 0.014 |
| `trace/cli-atomic-cmi-keep` fixed-kl | 0.179 ± 0.004 | 0.206 ± 0.006 |
| `trace/cli-atomic-cmi-keep` fixed | 0.179 ± 0.004 | 0.206 ± 0.006 |
| `trace/cli-atomic-cmi-keep` fixed-kl, shipped cut | 0.028 ± 0.002 | 0.042 ± 0.006 |
| landed `trace/core` fixed-kl (c = 1, N = 32 at request; c = 2, N = 32 at session) | 0.204 ± 0.009 | 0.270 ± 0.010 |
| landed `trace/cli-atomic` fixed-kl (c 1 / 2; N 32 / 2 / 32 / 2 / 8 at request, 32 / 8 / 32 / 8 / 32 at session) | 0.196 ± 0.005 | 0.247 ± 0.009 |
| landed `trace/cli-atomic` fixed-kl, shipped cut | 0.031 ± 0.006 | 0.045 ± 0.005 |

Per seed (request / session), information gain overwrite: 0.194 / 0.251, 0.205 / 0.272, 0.198 / 0.269,
0.209 / 0.271, 0.215 / 0.276; CMI overwrite: 0.171 / 0.201, 0.181 / 0.230, 0.185 / 0.226, 0.185 / 0.231,
0.195 / 0.233; information gain keep at session 0.253 / 0.272 / 0.269 / 0.272 / 0.276; CMI keep at
session 0.199 / 0.228 / 0.224 / 0.232 / 0.233; atomic CMI (fixed-kl): 0.173 / 0.195, 0.178 / 0.207,
0.181 / 0.205, 0.180 / 0.211, 0.182 / 0.210; landed `trace/cli-atomic`: 0.188 / 0.232, 0.194 / 0.251,
0.199 / 0.250, 0.196 / 0.247, 0.201 / 0.254.

### Hypotheses (paired per-seed differences, five-seed mean, paired t-test)

| hypothesis | request | session | verdict |
|---|---|---|---|
| **H-ctx-cmi-f1** (\|CMI − IG\| ≤ 0.01 at matched context) | overwrite **−0.021 (p 0.001)**; keep **−0.021 (p 0.001)** | overwrite **−0.044 (p < 0.001)**; keep **−0.045 (p < 0.001)** | **fail** on both grains |
| **H-ctx-cmi-tau** (CMI's mean \|τ drift\| ≤ IG's, within one grid step) | overwrite 0.70 vs 0.39 (+0.31 decade); keep 0.70 vs 0.39 | overwrite 0.58 vs 0.42 (+0.16); keep 0.58 vs 0.52 | **pass** (every excess under the half-decade) |
| **H-ctx-keep-perseq** (keep's per-sequence pooled F1 ≥ overwrite's at session) | not applicable | IG −0.003 (p 0.13); CMI −0.005 (p 0.018) — neither below by > 0.01 | **pass** |
| **H-atomic-cmi-f1** (atomic CMI ≥ landed `trace/cli-atomic` fixed-kl) | **−0.017 (p < 0.001)** | **−0.041 (p < 0.001)** | **fail** on both grains |
| **H-atomic-cmi-tau** (atomic CMI's drift ≤ the landed atomic arm's F1 drift) | F1 drift 0.031 vs 0.038 (τ drift 0.61 decade) | 0.037 vs 0.053 (0.29) | **pass** |
| **H-atomic-cmi-shipped** (shipped cut non-empty, F1 ≥ the landed shipped cut) | −0.002 (p 0.44), non-empty 5 / 5 | −0.003 (p 0.29), non-empty 5 / 5 | **pass** |
| H-ctx-keep-context-share (per-trace root-cause axis) | — | — | not computed (per-trace rows pending) |

**H-ctx-cmi-f1 fails on both grains at m, as at s, and the session gap widens with the rung** (xs
−0.016 / −0.017, s −0.031 / −0.036, m −0.044 / −0.045): the CMI cell sits under the information-gain
cell on every seed (request −0.023 / −0.024 / −0.013 / −0.024 / −0.020; session −0.050 / −0.042 /
−0.043 / −0.040 / −0.043 overwrite). The gap is precision again — at request 0.219 against 0.283 at
recall 0.159 against 0.160; at session 0.249 against 0.318 at recall 0.206 against 0.231 — with
orientation accuracy 0.82 against 0.89 at session and skeleton F1 within 0.006. The CMI's τ drift is
the larger of the two on both grains at m (0.58–0.70 against 0.39–0.52 decades), still under the
registered half-decade excess, so H-ctx-cmi-tau passes; the author's claim that the bounded score
drifts less does not show at m.

**H-atomic-cmi-f1 fails at m** (as at s): the atomic CMI arm reads 0.179 / 0.206 against the landed
atomic information-gain arm's 0.196 / 0.247, below on every seed (request −0.015 / −0.016 / −0.018 /
−0.016 / −0.019; session −0.037 / −0.044 / −0.045 / −0.036 / −0.044). The landed arm's N differs by
seed (2–32); the family's arm reads N = 8 by registration. Its shipped cut (791–865 edges at request,
1,418–7,676 at session) reads 0.028 / 0.042 against the landed shipped cut's 0.031 / 0.045 (n.s.;
H-atomic-cmi-shipped passes).

### Descriptive (not hypotheses)

- **Keep vs overwrite at matched score.** Request: identical cells (c = 1 everywhere, both scores).
  Session: information gain +0.0001 (p 0.78), CMI −0.0008 (p 0.14) — the `none` layout at c = 1 and
  the overwrite layout at c = 2 read the same type-level F1 within 0.001 on every seed; per sequence
  the keep cells are 0.003–0.005 under the overwrite cells (pooled F1 0.402 vs 0.405 and 0.375 vs
  0.380; scoreable 0.992 vs 0.989).
- **Against the landed `trace/core` cell** (N = 32 on every seed and grain): the family's
  information-gain overwrite cell at N = 8 reads −0.0001 (p 0.60) at request and −0.002 (p 0.007) at
  session — the landed H-sat band (F1 flat in N) holds to within 0.002 at m.
- **The atomic arm's two paths** agree within 0.0001 on every seed and grain.
- **Grid edges and drift.** No cell froze at a grid edge. The test-side argmax τ lies within one grid
  step of the frozen τ on 52 / 60 frozen cells (the CMI cells drift a full decade on eight: seeds 0 and
  3 at request on both core layouts, seed 0 at session on both, seed 4 of the atomic arm at request on
  both paths);
  mean |drift| 0.29–0.70 decades. The five-seed mean test F1 exceeds the validation F1 by 0.030–0.068
  on every frozen cell (positive on 60 / 60 per-seed cells), most on the information-gain session
  cells (+0.067).
- **Per-sequence axis** (`tables/m-latent-context/m-latent-perseq.md`): pooled directed F1 0.43–0.44
  at request and 0.38–0.41 at session for the core cells; the atomic CMI arm 0.460 / 0.423 (pooled
  recall 0.822 / 0.624 — it tests cause 0, which the core staircase never does); scoreable fraction
  0.975 / 0.992 for the keep cells against 0.975 / 0.989 for the overwrite cells. Per-sequence AUROC
  0.67–0.68 / 0.78–0.80 (core), 0.69 / 0.83 (atomic).

## l (test reads `l-latent-s{0..4}-ctx-test-26-11-02` under `freezes/2026-11-02-l-latent-s{k}-context.json`)

Validation sweeps `l-latent-s{k}-ctx-val-26-11-01` (N = 8 on the landed seed-k l backbones, the 10k / 5k sample;
43–47 min each on a g5.2xlarge): every registered check passes on 5 / 5 seeds — identity checks (ii)
and (iii) bit-equal on every seed; identity (i) (the re-scored `trace/core` and Granger rows against
the landed l tables) exact on every seed (2,160 / 2,160 landed N = 8 rows). 0 corrupted cells. Frozen in
`freezes/2026-11-02-l-latent-s{k}-context.json`. **Every CMI request cell froze at the bottom of the
τ grid (1e-7) on every seed** (`at_grid_edge`: the three CMI arms at request, 20 cells of 80) — the
caveat of §6 (b) realised at l, where the request grain is ceiling-bound (reference line 0.093–0.100;
the landed request cells froze at the grid bottom from l up, note of 2026-10-05); the xl grid of
2026-10-05 reaches 0. Validation F1 at the frozen cells (request / session), information gain
overwrite 0.092 / 0.113, 0.100 / 0.120, 0.100 / 0.116, 0.102 / 0.124, 0.099 / 0.121; CMI overwrite
0.083 / 0.094, 0.096 / 0.104, 0.087 / 0.088, 0.097 / 0.107, 0.096 / 0.109; atomic CMI 0.086 / 0.094,
0.097 / 0.100, 0.089 / 0.093, 0.099 / 0.105, 0.098 / 0.108. Every request cell froze at c = 1 and
every session core overwrite cell at c = 2. The l scoresweeps over nine cells per grain peaked at
9.0–10.1 GiB resident, the same as the landed 60-cell l scoresweeps: the scorer's peak scales with
the universe, not the cell count, which sizes the xl scoring half (two instances, as the landed xl
Job V).

Test reads on the 20k / 10k test sample (8 discover reads per seed; 112–129 min of stage wall clock per
seed on a g5.2xlarge, of which annotate + seqscore 82–97 min at a 15.2–18.0 GiB peak): every registered
check passes on 5 / 5 seeds — every stage `ok` on box commit `09579ed3`, freeze bound at `e1b21b11` (#81),
staging on the m context test reads, 438-file manifests, 16 / 16 frozen cells read exactly once at
their (c, N, g, τ) and `context_mode`, 20,000 request / 9,354–10,000 session sequences probed, 0
corrupted cells on the information-gain paths. The sample-size note applies: every frozen cell reads
above its validation F1 (80 / 80 per-seed cells; five-seed means +0.021 … +0.041).

### Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| cell (`fixed-kl` path unless stated) | request | session |
|---|---|---|
| `trace/core-ig-overwrite` (c = 1 at request; c = 2 at session, every seed) | 0.135 ± 0.006 | 0.158 ± 0.011 |
| `trace/core-cmi-overwrite` (c = 1; c = 2) | 0.124 ± 0.007 | 0.126 ± 0.017 |
| `trace/core-ig-keep` (c = 1, `none`) | 0.135 ± 0.006 | 0.160 ± 0.012 |
| `trace/core-cmi-keep` (c = 1, `none`) | 0.124 ± 0.007 | 0.126 ± 0.018 |
| `trace/cli-atomic-cmi-keep` fixed-kl | 0.125 ± 0.008 | 0.121 ± 0.011 |
| `trace/cli-atomic-cmi-keep` fixed | 0.125 ± 0.008 | 0.121 ± 0.011 |
| `trace/cli-atomic-cmi-keep` fixed-kl, shipped cut | 0.006 ± 0.004 | 0.007 ± 0.003 |
| landed `trace/core` fixed-kl (c = 1, N 32 / 32 / 32 / 32 / 8 at request; c = 2, N = 32 at session) | 0.136 ± 0.006 | 0.161 ± 0.011 |
| landed `trace/cli-atomic` fixed-kl (c = 1, N 8 / 2 / 8 / 8 / 8 at request; c = 2, N = 32 at session) | 0.133 ± 0.006 | 0.148 ± 0.014 |
| landed `trace/cli-atomic` fixed-kl, shipped cut | 0.006 ± 0.004 | 0.006 ± 0.002 |

Per seed (request / session), information gain overwrite: 0.126 / 0.148, 0.139 / 0.162, 0.134 / 0.145,
0.140 / 0.167, 0.138 / 0.168; CMI overwrite: 0.115 / 0.114, 0.129 / 0.136, 0.116 / 0.103, 0.128 / 0.131,
0.130 / 0.146; information gain keep at session 0.149 / 0.169 / 0.145 / 0.168 / 0.169; CMI keep at
session 0.113 / 0.135 / 0.102 / 0.131 / 0.146; atomic CMI (fixed-kl): 0.116 / 0.114, 0.130 / 0.125,
0.117 / 0.106, 0.129 / 0.126, 0.132 / 0.133; landed `trace/cli-atomic`: 0.125 / 0.138, 0.134 / 0.150,
0.130 / 0.129, 0.138 / 0.159, 0.138 / 0.163.

### Hypotheses (paired per-seed differences, five-seed mean, paired t-test)

| hypothesis | request | session | verdict |
|---|---|---|---|
| **H-ctx-cmi-f1** (\|CMI − IG\| ≤ 0.01 at matched context) | overwrite **−0.012 (p 0.002)**; keep **−0.012 (p 0.002)** | overwrite **−0.032 (p 0.001)**; keep **−0.034 (p 0.001)** | **fail** on both grains |
| **H-ctx-cmi-tau** (registered F1-drift basis at l: CMI's mean \|F1 drift\| not above IG's by > 0.01) | overwrite 0.032 vs 0.037; keep 0.032 vs 0.037 | overwrite 0.026 vs 0.039; keep 0.025 vs 0.040 | **pass** |
| **H-ctx-keep-perseq** (keep's per-sequence pooled F1 ≥ overwrite's at session) | not applicable | IG +0.002 (p 0.87); CMI −0.007 (p < 0.001) — neither below by > 0.01 | **pass** |
| **H-atomic-cmi-f1** (atomic CMI ≥ landed `trace/cli-atomic` fixed-kl) | −0.008 (p 0.004), inside the band | **−0.027 (p < 0.001)** | **pass at request, fail at session** |
| **H-atomic-cmi-tau** (F1-drift basis) | 0.031 vs 0.034 | 0.021 vs 0.035 | **pass** |
| **H-atomic-cmi-shipped** (shipped cut non-empty, F1 ≥ the landed shipped cut) | −0.0005 (p 0.49), non-empty 5 / 5 | +0.0002 (p 0.55), non-empty 5 / 5 | **pass** |
| H-ctx-keep-context-share (per-trace root-cause axis) | — | — | not computed (per-trace rows pending) |

**H-ctx-cmi-f1 fails on both grains at l, as at s and m, but the absolute session gap no longer widens:**
xs −0.016 / −0.017, s −0.031 / −0.036, m −0.044 / −0.045, l −0.032 / −0.034. The levels fall faster than
the gap does; relative to the information-gain cell the overwrite gap keeps growing (descriptive, not a
registered reading): −4.4 %, −9.3 %, −16.4 %, −20.2 % at session; −1.3 %, −9.2 %, −10.1 %, −8.8 % at
request. The CMI cell sits under the information-gain cell on every seed (request −0.011 / −0.010 /
−0.018 / −0.012 / −0.008; session −0.034 / −0.026 / −0.041 / −0.036 / −0.022 overwrite). At session the
gap is precision (0.152 against 0.210, recall 0.110 against 0.128) and orientation (accuracy 0.72
against 0.82, lower on every seed); skeleton F1 does not separate them (0.105 against 0.103).

**The request verdict is read under the grid-floor caveat of §6 (b).** All five CMI request cells froze
at τ = 1e-7, the bottom of the l grid, and their validation F1 (0.083–0.097) lies under the
every-scored-pair reference line of their seed (0.093–0.100) by 0.002–0.012; the information-gain cells
froze inside the grid (1e-5 / 1e-4; validation F1 0.091–0.102, that is, within ±0.002 of the line). So the
CMI cell at request is limited by where the grid stops, and part of the −0.012 may belong to the grid
rather than the score. The verdict stands as registered: the l grid was fixed before the read and is not
re-opened here. The xl grid reaches τ = 0 and shows what the extension does on validation (xl section).
The session cells are interior on every seed and carry no such caveat.

**Post hoc (2026-11-03, not a registered run; the owner's decision to record it).** The landed l test
request read of the core overwrite pass was re-swept locally with the xl grid (τ down to 0; the benchmark's
scorer, the same score file; the registered cells reproduce on 5 / 5). With the CMI cell at τ = 0 the
difference to the frozen information-gain cell is −0.0016 / −0.0086 / −0.0040 / −0.0083 / −0.0073, mean
−0.0059 — inside the band — against the registered −0.0119 at 1e-7: about half of the l request gap is the
grid floor. At τ = 0 the CMI and the information-gain columns give the same F1 and the same edge count on
every seed (49,321 … 65,536 edges): on this read a pair has a positive CMI exactly when it has a positive
information gain, so a τ = 0 comparison carries no score content, and the xl request grain (below) is read
the same way. The registered l verdict stands; this paragraph is its context, not a re-selection.

**H-atomic-cmi-f1 passes at request and fails at session.** The request difference is inside the band
(−0.009 / −0.004 / −0.012 / −0.009 / −0.006) with the same floor caveat on the atomic CMI cells; the
session difference is −0.024 / −0.025 / −0.023 / −0.033 / −0.030. The landed arm's N differs by seed at
request (2–8) and is 32 at session, against the family's registered N = 8. The shipped cuts are small
(199–701 edges at request, 319–5,812 at session) and match the landed shipped cuts' F1 within noise.

**The drift hypotheses pass on a weak basis.** The registered F1-drift basis does not isolate threshold
stability: every frozen cell reads above validation on a test sample twice the size, and the arms with the
higher F1 show the larger drift. That the CMI arms drift less at l therefore says little about the
author's claim on threshold stability; the τ-drift readings at xs, s and m are the direct test.

### Descriptive (not hypotheses)

- **Keep vs overwrite at matched score.** Request: identical cells (c = 1 everywhere, both scores).
  Session: information gain +0.002 (p 0.22), CMI −0.0005 (p 0.27).
- **Against the landed `trace/core` cell:** the family's information-gain overwrite cell at N = 8 reads
  −0.0001 (p 0.78) at request and −0.003 (p 0.07) at session, where the landed cell reads N = 32.
- **The atomic arm's two paths** agree within 0.0001 on every seed and grain.
- **Grid edges:** 20 request cells (`core-cmi-overwrite`, `core-cmi-keep` and both paths of the atomic arm,
  five seeds); none at session.
- **Per-sequence axis** (`tables/l-latent-context/l-latent-perseq.md`): pooled directed F1 at request 0.429
  (information gain), 0.414 (CMI), 0.432 (atomic CMI, recall 0.918 — it tests cause 0, which the core
  staircase never does); at session 0.377–0.378 (information gain), 0.292–0.299 (CMI), 0.287 (atomic CMI).
  Scoreable fraction 0.952 at request; at session 0.902 for the overwrite cells (566 of 10,000 short
  sequences skipped on seed 0) and 0.897 for the keep cells. Pooled AUROC 0.654 / 0.833 (information gain),
  0.635 / 0.802 (CMI), 0.699 / 0.832 (atomic CMI).

## xl (test reads `xl-latent-s{0..4}-ctx-test-26-11-03` + `…-ctx-annotate-26-11-03` under `freezes/2026-11-03-xl-latent-s{k}-context.json`)

Validation sweeps `xl-latent-s{k}-ctx-val-26-11-02` + `xl-latent-s{k}-ctx-val-score-26-11-02` (N = 8 on the landed seed-k xl
backbones, the 10k / 5k sample; two instances per seed, as the landed xl Job V): the GPU half took 34–35
min on a g5.2xlarge (peak device 3.2–4.1 GiB, host ≤ 3.6 GiB), the CPU half 45–52 min on an r7i.4xlarge
(scoresweeps 341–438 s at request, 899–1,122 s at session; peak 41.2–47.5 GiB, the landed xl peaks, so the
peak is universe-bound here too). Every registered check passes on 5 / 5 seeds: every stage `ok` on box
commit `09579ed3`, manifests sha-exact, `--args-diff` clean, 0 corrupted cells, CMI within [0, log 2],
identities (ii) and (iii) bit-equal. Identity (i) runs as D-SB-26: 12 landed N = 8 score files per seed
bit-equal on their 90 columns, and 120 `trace/core/fixed-kl` rows per seed exact against the landed score2
tables. Frozen in `freezes/2026-11-03-xl-latent-s{k}-context.json`, on the xl grids of 2026-10-05 (20 τ
values, down to 0). Every request cell froze at c = 1 and every session core overwrite cell at c = 2.

Validation directed F1 at the frozen cells, seeds 0–4 (descriptive; the hypotheses are scored on the test read):

| cell (`fixed-kl`) | request | session |
|---|---|---|
| `trace/core-ig-overwrite` | 0.0578 / 0.0533 / 0.0582 / 0.0568 / 0.0581 | 0.0682 / 0.0593 / 0.0711 / 0.0692 / 0.0662 |
| `trace/core-cmi-overwrite` | 0.0578 / 0.0533 / 0.0582 / 0.0568 / 0.0581 | 0.0584 / 0.0513 / 0.0637 / 0.0612 / 0.0567 |
| `trace/core-ig-keep` | as overwrite (same read) | 0.0695 / 0.0596 / 0.0723 / 0.0706 / 0.0663 |
| `trace/core-cmi-keep` | as overwrite (same read) | 0.0586 / 0.0511 / 0.0645 / 0.0622 / 0.0563 |
| `trace/cli-atomic-cmi-keep` | 0.0600 / 0.0563 / 0.0608 / 0.0592 / 0.0612 | 0.0601 / 0.0528 / 0.0644 / 0.0620 / 0.0576 |
| every-scored-pair reference line, core overwrite read | 0.0600 / 0.0563 / 0.0608 / 0.0592 / 0.0612 | 0.0536 / 0.0511 / 0.0536 / 0.0532 / 0.0548 |

**The request grain does not separate the scores at xl.** With the grid reaching 0, all five CMI request
cells freeze at τ = 0 (grid edge) and the information-gain cells at τ = 1e-8 … 1e-6; the two read the same
validation F1 within 0.00005 on every seed, on nearly the same edge set (35,586–39,401 against 35,642–39,484
edges). Both keep almost every pair the staircase scored above zero, 0.002–0.003 under the reference line. The
atomic CMI cell at τ = 0 predicts every observed pair, so it equals the line on 5 / 5 seeds. **A pass of
H-ctx-cmi-f1 at request on the xl test read would therefore not show that the two scores agree.** It also
bears on the l request verdict (l section): given a grid that reaches 0, the request gap is absent on xl
validation. That is a different rung and split, and it does not change the l verdict.

**The session grain still separates them.** The information-gain overwrite cells freeze at τ = 1e-5 … 1e-4
and lead the CMI overwrite cells (τ = 1e-9 … 3e-6) by 0.007–0.010 on every seed. The CMI cells sit 0.0002–0.0101
above the reference line (0.0002 on seed 1), the information-gain cells 0.0083–0.0174. One further grid edge:
`trace/core-cmi-keep` at session on seed 1 freezes at τ = 0. So 21 of 80 cells sit at an edge.

Test reads as two instances per seed, as the landed xl Job T: the discover half (`…-ctx-test-26-11-03`,
g5.2xlarge, 8 reads on the 20k / 10k test sample staged on the l context test reads, 31–33 min of stage
wall clock, peak device 3.9–4.1 GiB) and the scoring half (`…-ctx-annotate-26-11-03`, r7i.8xlarge, 16
annotate + 16 seqscore, 6.9–8.5 h, peak 69.9–81.7 GiB on the session cells — the landed xl per-cell
figures). Every registered check passes on 5 / 5 seeds: every stage `ok` on box commit `48c13535`, freeze
bound at `b373d248` (#82), both manifests verified, `--args-diff` clean on all 43 records per seed, 16 / 16
frozen cells read exactly once at their (c, N, g, τ) and `context_mode`, 20,000 request / 8,603–10,000
session sequences probed, 0 corrupted cells on the information-gain paths. The sample-size note applies:
every frozen cell reads above its validation F1 (80 / 80 per-seed cells; five-seed means +0.018 … +0.031).

### Type-level directed F1 at the frozen cells (test; five-seed mean ± sd)

| cell (`fixed-kl` path unless stated) | request | session |
|---|---|---|
| `trace/core-ig-overwrite` (c = 1 at request; c = 2 at session, every seed) | 0.088 ± 0.002 | 0.096 ± 0.008 |
| `trace/core-cmi-overwrite` (c = 1, τ = 0; c = 2) | 0.088 ± 0.002 | 0.077 ± 0.010 |
| `trace/core-ig-keep` (c = 1, `none`) | 0.088 ± 0.002 | 0.097 ± 0.009 |
| `trace/core-cmi-keep` (c = 1, `none`; τ = 0 at request) | 0.088 ± 0.002 | 0.076 ± 0.012 |
| `trace/cli-atomic-cmi-keep` fixed-kl (τ = 0 at request) | 0.089 ± 0.002 | 0.079 ± 0.008 |
| `trace/cli-atomic-cmi-keep` fixed | 0.089 ± 0.002 | 0.079 ± 0.008 |
| `trace/cli-atomic-cmi-keep` fixed-kl, shipped cut | 0.002 ± 0.001 | 0.001 ± 0.001 |
| landed `trace/core` fixed-kl (c = 1, N 8 / 8 / 16 / 8 / 16 at request; c = 2, N = 16 at session) | 0.088 ± 0.002 | 0.097 ± 0.009 |
| landed `trace/cli-atomic` fixed-kl (c = 1, N 8 / 16 / 8 / 2 / 2; c = 2, N 16 / 16 / 16 / 8 / 16) | 0.089 ± 0.002 | 0.088 ± 0.010 |
| landed `trace/cli-atomic` fixed-kl, shipped cut | 0.002 ± 0.001 | 0.001 ± 0.001 |

Per seed (request / session), information gain overwrite: 0.090 / 0.099, 0.085 / 0.085, 0.089 / 0.104,
0.089 / 0.102, 0.089 / 0.090; CMI overwrite: 0.090 / 0.075, 0.085 / 0.066, 0.089 / 0.089, 0.089 / 0.086,
0.089 / 0.070; information gain keep at session 0.100 / 0.085 / 0.107 / 0.103 / 0.090; CMI keep at session
0.077 / 0.061 / 0.090 / 0.086 / 0.069; atomic CMI (fixed-kl): 0.090 / 0.080, 0.086 / 0.069, 0.089 / 0.088,
0.089 / 0.084, 0.090 / 0.072; landed `trace/cli-atomic`: 0.090 / 0.088, 0.086 / 0.076, 0.089 / 0.099,
0.090 / 0.098, 0.090 / 0.080.

### Hypotheses (paired per-seed differences, five-seed mean, paired t-test)

| hypothesis | request | session | verdict |
|---|---|---|---|
| **H-ctx-cmi-f1** (\|CMI − IG\| ≤ 0.01 at matched context) | overwrite −0.0001 (p 0.36); keep −0.0001 (p 0.36) — degenerate, see below | overwrite **−0.019 (p < 0.001)**; keep **−0.020 (p < 0.001)** | **pass at request (degenerate), fail at session** |
| **H-ctx-cmi-tau** (F1-drift basis at xl) | overwrite 0.031 vs 0.031; keep 0.031 vs 0.031 | overwrite 0.019 vs 0.029; keep 0.018 vs 0.029 | **pass** |
| **H-ctx-keep-perseq** (keep's per-sequence pooled F1 ≥ overwrite's at session) | not applicable | IG −0.0001 (p 0.99); CMI −0.007 (p 0.075) — neither below by > 0.01 | **pass** |
| **H-atomic-cmi-f1** (atomic CMI ≥ landed `trace/cli-atomic` fixed-kl) | −0.0001 (p 0.13) — degenerate | −0.009 (p 0.002), inside the band | **pass** (session by the band) |
| **H-atomic-cmi-tau** (F1-drift basis) | 0.029 vs 0.029 | 0.019 vs 0.025 | **pass** |
| **H-atomic-cmi-shipped** (shipped cut non-empty, F1 ≥ the landed shipped cut) | +0.00004 (p 0.58), non-empty 5 / 5 | −0.00001 (p 0.84), non-empty 5 / 5 | **pass** |
| H-ctx-keep-context-share (per-trace root-cause axis) | — | — | not computed (xl has no root-cause row) |

**The request grain is degenerate at xl, as the validation sweeps said it would be.** Every CMI request
cell froze at τ = 0 and every information-gain cell at 1e-8 … 1e-6; on the test read the two cuts keep
64,182–70,319 (information gain) against 65,336–70,579 (CMI) edges and read the same F1 within 0.0004 on
every seed, the same precision, recall and orientation accuracy (0.95). The passes of H-ctx-cmi-f1 and
H-atomic-cmi-f1 at request say that the τ = 0 cut of a bounded score and the near-floor cut of an
unbounded one keep nearly the same pairs — not that the scores agree (the l post hoc note: a pair has a
positive CMI exactly when it has a positive information gain). They are recorded as passes because the
registered clause passes; they carry no weight in the reading of the family.

**H-ctx-cmi-f1 fails at session at xl, the fifth rung of five.** The CMI cell sits under the
information-gain cell on every seed (overwrite −0.024 / −0.020 / −0.014 / −0.017 / −0.021; keep −0.023 /
−0.023 / −0.018 / −0.018 / −0.021). The gap is precision again — 0.086 against 0.129 (per seed 0.076 /
0.058 / 0.124 / 0.108 / 0.062 against 0.142 / 0.100 / 0.147 / 0.148 / 0.105) at recall 0.074 against
0.078 — and orientation (0.675 against 0.728, lower on every seed); the CMI cells keep 1.0–1.9× the
edges of the information-gain cells (110,773–257,062 against 104,088–148,705) and their skeleton F1 is
the higher one (0.084 against 0.069). The relative gap is −19.8 % (overwrite) / −21.1 % (keep), as at l.

**H-atomic-cmi-f1 passes at session by the band:** −0.0074 / −0.0068 / −0.0115 / −0.0137 / −0.0076, mean
−0.009, p 0.002, every seed negative; the landed arm reads N = 16 (seed 3: 8) at c = 2 against the
family's registered N = 8 at c = 1. The atomic CMI arm matches the core CMI cells at session (0.079
against 0.077) and leads them per sequence (pooled F1 0.182 against 0.181 / 0.173). The shipped cuts are
small (145–588 edges at request, 90–2,945 at session) and equal the landed shipped cuts within 0.0004.

### Descriptive (not hypotheses)

- **Keep vs overwrite at matched score.** Request: identical cells. Session: information gain +0.001
  (per seed +0.0006 / −0.0006 / +0.0034 / +0.0010 / −0.0007), CMI −0.0007; per sequence the keep cells
  read −0.0001 (information gain) and −0.007 (CMI, p 0.075) under the overwrite cells, with scoreable
  fractions 0.822 against 0.850 (the overwrite reads at c = 2 skip 1,308–1,397 short sessions; the keep
  reads skip none, but the per-sequence rule scores fewer of theirs).
- **Against the landed `trace/core` cell:** the family's information-gain overwrite cell reads within 0.0002
  at request and −0.001 at session (per seed −0.0004 / +0.0003 / −0.0029 / 0.0000 / −0.0018; the landed
  cell reads N = 16 at session).
- **The atomic arm's two paths** agree within 0.0001 on every seed and grain.
- **Per-sequence axis** (`tables/xl-latent-context/xl-latent-perseq.md`): pooled directed F1 at request
  0.380 (information gain), 0.379 (CMI), 0.367 (atomic CMI, recall 1.000); at session 0.241 (information
  gain, both layouts), 0.181 / 0.173 (CMI overwrite / keep), 0.182 (atomic CMI). Pooled AUROC 0.638 / 0.613
  / 0.650 at request, 0.820 / 0.774 / 0.803 at session. Scoreable fraction 0.911 at request.
- **Drift:** every per-seed frozen cell reads above its validation F1 (80 / 80); mean F1 drift 0.029–0.031
  at request on every arm, 0.018–0.019 (CMI arms) and 0.029–0.030 (information-gain arms) at session.

## Across the rungs (type-level, five seeds per rung; the family's test reads)

| | xs | s | m | l | xl |
|---|---|---|---|---|---|
| information gain overwrite, request / session | 0.317 / 0.354 | 0.290 / 0.334 | 0.204 / 0.268 | 0.135 / 0.158 | 0.088 / 0.096 |
| CMI − information gain, request (overwrite) | −0.004 (pass) | −0.027 (**fail**) | −0.021 (**fail**) | −0.012 (**fail**; grid floor) | −0.0001 (pass; degenerate) |
| CMI − information gain, session (overwrite / keep) | −0.016 / −0.017 (**fail**) | −0.031 / −0.036 (**fail**) | −0.044 / −0.045 (**fail**) | −0.032 / −0.034 (**fail**) | −0.019 / −0.020 (**fail**) |
| the session gap relative to the information-gain cell | −4.4 % | −9.3 % | −16.4 % | −20.2 % | −19.8 % |
| atomic CMI − landed `trace/cli-atomic`, request / session | −0.003 / −0.017 (pass / pass) | −0.024 / −0.035 (**fail** / **fail**) | −0.017 / −0.041 (**fail** / **fail**) | −0.008 / −0.027 (pass / **fail**) | −0.0001 / −0.009 (pass / pass) |
| H-ctx-cmi-tau, H-atomic-cmi-tau | pass (τ-drift basis) | pass (τ-drift) | pass (τ-drift) | pass (F1-drift basis) | pass (F1-drift) |
| H-atomic-cmi-shipped | pass | fail at request | pass | pass | pass |
| H-ctx-keep-perseq | pass | pass | pass | pass | pass |

**Reading.** (1) At the session grain the bounded CMI score is below the information gain on every seed
of every rung, by 0.016–0.045 absolute and by a relative gap that grows from 4 % at xs to about 20 % at l
and xl; the loss is precision and orientation, never recall or the skeleton. The author's claim that the
two scores agree within 0.01 does not hold on trace-bench. (2) At the request grain the comparison is
informative only where the τ grid is interior to both scores: s and m (fail by 0.02–0.03); at xs the
gap is inside the band; at l the CMI cells sit on the grid floor and about half the gap is the floor;
at xl, with the grid reaching 0, both scores keep nearly every scored pair and the pass is empty.
(3) The atomic CMI arm does not beat the landed atomic information-gain arm anywhere; it is below it
on s, m and l at session and within the band on xs and xl. (4) The keep layout (every observed token a
candidate, c = 1) is a free choice: it matches the overwrite layout within 0.002 at the type level and
within 0.01 per sequence on every rung. (5) The drift hypotheses pass everywhere, but on the F1-drift
basis of l and xl they say little (the lower-F1 arm drifts less by construction); the τ-drift readings
of xs, s and m are the test, and there the CMI's drift is the larger of the two at m (within the
registered half-decade). The root-cause axes of the family are in `benchmarks/trace-bench-rca/findings/`
(xs landed; s, m pending).


### Not landed here

The xs case-level root-cause rows of the family are landed (`benchmarks/trace-bench-rca/findings/rca-latent.md`,
section of 2026-11-02: H-rca-hop0 fails, H-rca-source and H-rca-presub pass, the family's hop0 rows
under `control/rule`); the s / m case-level rows and the per-trace axis (the GPU replica) follow. The
type-level and per-sequence axes are complete on xs … xl.
