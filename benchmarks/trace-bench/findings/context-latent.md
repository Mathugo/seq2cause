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

## l (validation frozen 2026-11-02; test reads pending)

`l-latent-s{k}-ctx-val-26-11-01` (N = 8 on the landed seed-k l backbones, the 10k / 5k sample;
43–47 min each on a g5.2xlarge): every registered check passes on 5 / 5 seeds — identity checks (ii)
and (iii) bit-equal on every seed; identity (i) (the re-scored `trace/core` and Granger rows against
the landed l tables) passes on the seeds whose l corpus is pulled locally (seed 0: 432 / 432 rows;
the remaining seeds are run as disk space allows — see RUN.md). 0 corrupted cells. Frozen in
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

### Not landed here

The xs case-level root-cause rows of the family are landed (`benchmarks/trace-bench-rca/findings/rca-latent.md`,
section of 2026-11-02: H-rca-hop0 fails, H-rca-source and H-rca-presub pass, the family's hop0 rows
under `control/rule`); the s / m case-level rows, the per-trace axis (the GPU replica), the l test
reads and the xl rung (type-level and per-sequence axes only) follow.
