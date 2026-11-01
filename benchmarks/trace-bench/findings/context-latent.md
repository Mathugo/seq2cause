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

### Not landed here

The root-cause rows of the family (`benchmarks/trace-bench-rca`: case level under the `hop0`,
`hop1` and `source` readings with the pre-fault subtraction, the per-trace axis, the earliest-error
control) and the s / m rungs follow; the l / xl rungs read the type-level and per-sequence axes
only.
