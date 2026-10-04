# Findings — the root-cause-analysis reading: the per-sequence axis first, then the type-level axis on the reachable truth (xs … l / latent, five seeds)

Score-side reading of 2026-10-04 over the landed test reads of the four rungs
(`results/{xs,s,m,l}/latent/seed={0..4}/`, tables `tables/{xs,s,m,l}-latent/`); no method run, no
new freeze, no pre-registered hypothesis. It applies the owner's reading order of 2026-10-04
(`plans/per-sequence-rules.md`, addendum of that date) and gives the reachable-only columns of
`report` across rungs.

## Why two readings

A root-cause analysis is handed one trace and asks which of its events caused which. That is the
**per-sequence axis**: every probed sequence is scored on its own position pairs against the
truth induced for that sequence. It leads here.

The **type-level axis** asks how much of the system's causal graph over event types a fixed
number of traces recovers. Its benchmark F1 counts every true edge, including those whose two
event types never occur together in the probed sample — edges no method can find. That share
shrinks as the vocabulary grows while the sample stays fixed (the *reachable share* row of the
last table of each grain: 0.48 → 0.12 at request, 0.61 → 0.18 at session), and it drags every
arm's benchmark F1 down with it. The **reachable-only** columns score the same predictions
against the truth the read could reach: true and false positives are the benchmark's, only the
unreachable false negatives leave the count.

Three limits apply to the reachable columns and are not repeated under each table:

- **τ was frozen on the benchmark F1, not on these columns**, and is not re-selected. Under a low
  reachable share that selection favours predicting more, so the reachable F1 here is a lower
  bound on what a selection on it would reach, and an ordering of arms on it is partly an
  ordering of where each arm's τ landed.
- **Shapley reads its own smaller sample from `s` on** (2,000 / 500 sequences), so its reachable
  truth is a smaller, more frequent subset; its reachable rows are not comparable with the other
  arms' and are marked.
- Every table is at the default floor 0.05 and at the frozen cell of each arm
  (`<arm>/shipped/frozen`, `…/none/frozen` for saliency and Shapley); the shipped-cut row is
  `trace/cli/shipped/shipped`.

## Tables

### request grain

Per-sequence axis — pooled directed F1 within the trace (five-seed mean ± sd):

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.423 ± 0.061 | 0.343 ± 0.060 | 0.444 ± 0.015 | 0.429 ± 0.018 |
| `trace/cli` | 0.431 ± 0.046 | 0.360 ± 0.068 | 0.448 ± 0.012 | 0.427 ± 0.024 |
| `trace/cli-atomic` | 0.505 ± 0.052 | 0.488 ± 0.058 | 0.491 ± 0.013 | 0.438 ± 0.030 |
| `baseline/granger` | 0.426 ± 0.052 | 0.376 ± 0.034 | 0.447 ± 0.019 | 0.427 ± 0.017 |
| `baseline/saliency` | 0.352 ± 0.121 | 0.278 ± 0.065 | 0.371 ± 0.011 | 0.389 ± 0.033 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.339 ± 0.047 | 0.337 ± 0.041 | 0.437 ± 0.013 | 0.465 ± 0.024 |
| `trace/cli` at the tool's shipped cut | 0.039 ± 0.032 | 0.082 ± 0.042 | 0.057 ± 0.012 | 0.050 ± 0.013 |
| *predict every candidate pair* | *0.456* | *0.397* | *0.409* | *0.401* |
| *scoreable fraction of the probed sequences* | *0.85* | *0.97* | *0.98* | *0.95* |

Per-sequence precision / recall (pooled):

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.38 / 0.52 | 0.32 / 0.37 | 0.37 / 0.55 | 0.31 / 0.72 |
| `trace/cli` | 0.37 / 0.54 | 0.34 / 0.38 | 0.37 / 0.57 | 0.31 / 0.72 |
| `trace/cli-atomic` | 0.37 / 0.85 | 0.36 / 0.74 | 0.35 / 0.83 | 0.28 / 0.97 |
| `baseline/granger` | 0.39 / 0.48 | 0.32 / 0.47 | 0.38 / 0.54 | 0.32 / 0.66 |
| `baseline/saliency` | 0.40 / 0.33 | 0.32 / 0.25 | 0.34 / 0.41 | 0.31 / 0.52 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.28 / 0.44 | 0.25 / 0.50 | 0.32 / 0.67 | 0.33 / 0.77 |

Type-level axis, reachable truth only — directed F1 (five-seed mean ± sd):

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.416 ± 0.056 | 0.382 ± 0.018 | 0.413 ± 0.025 | 0.312 ± 0.029 |
| `trace/cli` | 0.405 ± 0.051 | 0.391 ± 0.022 | 0.407 ± 0.030 | 0.309 ± 0.039 |
| `trace/cli-atomic` | 0.406 ± 0.059 | 0.379 ± 0.019 | 0.365 ± 0.014 | 0.287 ± 0.031 |
| `baseline/granger` | 0.419 ± 0.048 | 0.365 ± 0.029 | 0.427 ± 0.011 | 0.326 ± 0.026 |
| `baseline/saliency` | 0.378 ± 0.053 | 0.341 ± 0.031 | 0.342 ± 0.011 | 0.361 ± 0.021 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.333 ± 0.065 | 0.299 ± 0.019 | 0.358 ± 0.012 | 0.384 ± 0.026 |
| `trace/cli` at the tool's shipped cut | 0.066 ± 0.031 | 0.099 ± 0.042 | 0.112 ± 0.023 | 0.049 ± 0.024 |
| *predict every scored pair* | *0.358* | *0.253* | *0.224* | *0.241* |

Type-level axis, reachable truth only — precision / recall:

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.30 / 0.78 | 0.27 / 0.64 | 0.28 / 0.81 | 0.19 / 0.90 |
| `trace/cli` | 0.28 / 0.79 | 0.29 / 0.63 | 0.27 / 0.82 | 0.19 / 0.90 |
| `trace/cli-atomic` | 0.28 / 0.87 | 0.26 / 0.69 | 0.23 / 0.84 | 0.17 / 0.95 |
| `baseline/granger` | 0.30 / 0.71 | 0.25 / 0.71 | 0.29 / 0.79 | 0.20 / 0.85 |
| `baseline/saliency` | 0.28 / 0.61 | 0.22 / 0.77 | 0.21 / 0.88 | 0.23 / 0.85 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.24 / 0.56 | 0.20 / 0.58 | 0.24 / 0.67 | 0.26 / 0.70 |

Type-level axis, the benchmark's directed F1 over the whole truth (the headline of the per-rung findings), with the share of the truth the read can reach:

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.321 ± 0.043 | 0.289 ± 0.013 | 0.204 ± 0.009 | 0.135 ± 0.006 |
| `trace/cli` | 0.317 ± 0.045 | 0.292 ± 0.014 | 0.205 ± 0.009 | 0.135 ± 0.007 |
| `trace/cli-atomic` | 0.307 ± 0.043 | 0.287 ± 0.016 | 0.196 ± 0.005 | 0.133 ± 0.006 |
| `baseline/granger` | 0.321 ± 0.041 | 0.283 ± 0.014 | 0.206 ± 0.007 | 0.134 ± 0.006 |
| `baseline/saliency` | 0.279 ± 0.040 | 0.268 ± 0.025 | 0.192 ± 0.004 | 0.139 ± 0.007 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.255 ± 0.050 | 0.158 ± 0.014 | 0.070 ± 0.004 | 0.035 ± 0.003 |
| `trace/cli` at the tool's shipped cut | 0.033 ± 0.016 | 0.051 ± 0.023 | 0.024 ± 0.005 | 0.006 ± 0.003 |
| *reachable share of the truth (`trace/core` read)* | *0.48* | *0.48* | *0.20* | *0.12* |
| *reachable share, Shapley's read* | *0.49* | *0.22* | *0.06* | *0.03* |

### session grain

Per-sequence axis — pooled directed F1 within the trace (five-seed mean ± sd):

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.433 ± 0.049 | 0.346 ± 0.063 | 0.412 ± 0.023 | 0.383 ± 0.055 |
| `trace/cli` | 0.414 ± 0.046 | 0.350 ± 0.063 | 0.406 ± 0.020 | 0.359 ± 0.053 |
| `trace/cli-atomic` | 0.443 ± 0.047 | 0.423 ± 0.057 | 0.470 ± 0.012 | 0.385 ± 0.061 |
| `baseline/granger` | 0.419 ± 0.052 | 0.337 ± 0.062 | 0.407 ± 0.022 | 0.364 ± 0.027 |
| `baseline/saliency` | 0.296 ± 0.055 | 0.161 ± 0.018 | 0.313 ± 0.053 | 0.359 ± 0.030 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.289 ± 0.029 | 0.214 ± 0.022 | 0.194 ± 0.010 | 0.189 ± 0.015 |
| `trace/cli` at the tool's shipped cut | 0.042 ± 0.014 | 0.045 ± 0.026 | 0.016 ± 0.002 | 0.009 ± 0.005 |
| *predict every candidate pair* | *0.303* | *0.173* | *0.151* | *0.143* |
| *scoreable fraction of the probed sequences* | *0.97* | *0.99* | *0.99* | *0.90* |

Per-sequence precision / recall (pooled):

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.39 / 0.48 | 0.33 / 0.37 | 0.34 / 0.53 | 0.26 / 0.77 |
| `trace/cli` | 0.42 / 0.42 | 0.34 / 0.36 | 0.34 / 0.50 | 0.24 / 0.73 |
| `trace/cli-atomic` | 0.40 / 0.50 | 0.36 / 0.52 | 0.37 / 0.65 | 0.26 / 0.73 |
| `baseline/granger` | 0.39 / 0.46 | 0.32 / 0.36 | 0.33 / 0.53 | 0.24 / 0.71 |
| `baseline/saliency` | 0.29 / 0.31 | 0.37 / 0.10 | 0.37 / 0.29 | 0.27 / 0.54 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.22 / 0.44 | 0.13 / 0.53 | 0.11 / 0.72 | 0.11 / 0.79 |

Type-level axis, reachable truth only — directed F1 (five-seed mean ± sd):

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.443 ± 0.015 | 0.390 ± 0.023 | 0.442 ± 0.010 | 0.338 ± 0.052 |
| `trace/cli` | 0.438 ± 0.020 | 0.399 ± 0.019 | 0.461 ± 0.008 | 0.319 ± 0.054 |
| `trace/cli-atomic` | 0.413 ± 0.022 | 0.364 ± 0.020 | 0.381 ± 0.010 | 0.307 ± 0.050 |
| `baseline/granger` | 0.436 ± 0.027 | 0.399 ± 0.018 | 0.447 ± 0.009 | 0.336 ± 0.016 |
| `baseline/saliency` | 0.391 ± 0.017 | 0.301 ± 0.015 | 0.315 ± 0.055 | 0.275 ± 0.018 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.337 ± 0.015 | 0.231 ± 0.010 | 0.182 ± 0.006 | 0.177 ± 0.016 |
| `trace/cli` at the tool's shipped cut | 0.048 ± 0.020 | 0.082 ± 0.020 | 0.055 ± 0.006 | 0.016 ± 0.007 |
| *predict every scored pair* | *0.299* | *0.124* | *0.077* | *0.078* |

Type-level axis, reachable truth only — precision / recall:

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.35 / 0.59 | 0.29 / 0.58 | 0.32 / 0.70 | 0.22 / 0.73 |
| `trace/cli` | 0.35 / 0.60 | 0.31 / 0.58 | 0.35 / 0.68 | 0.21 / 0.72 |
| `trace/cli-atomic` | 0.32 / 0.59 | 0.26 / 0.61 | 0.26 / 0.70 | 0.20 / 0.68 |
| `baseline/granger` | 0.34 / 0.60 | 0.31 / 0.56 | 0.33 / 0.69 | 0.22 / 0.73 |
| `baseline/saliency` | 0.27 / 0.74 | 0.19 / 0.73 | 0.20 / 0.77 | 0.17 / 0.81 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.23 / 0.64 | 0.14 / 0.63 | 0.10 / 0.74 | 0.10 / 0.75 |

Type-level axis, the benchmark's directed F1 over the whole truth (the headline of the per-rung findings), with the share of the truth the read can reach:

| arm | xs | s | m | l |
|---|---|---|---|---|
| `trace/core` | 0.357 ± 0.021 | 0.342 ± 0.018 | 0.270 ± 0.010 | 0.161 ± 0.011 |
| `trace/cli` | 0.353 ± 0.021 | 0.349 ± 0.015 | 0.274 ± 0.010 | 0.155 ± 0.013 |
| `trace/cli-atomic` | 0.332 ± 0.011 | 0.324 ± 0.015 | 0.247 ± 0.009 | 0.148 ± 0.014 |
| `baseline/granger` | 0.354 ± 0.028 | 0.348 ± 0.015 | 0.271 ± 0.011 | 0.161 ± 0.007 |
| `baseline/saliency` | 0.324 ± 0.016 | 0.277 ± 0.013 | 0.222 ± 0.025 | 0.153 ± 0.005 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.286 ± 0.013 | 0.151 ± 0.013 | 0.070 ± 0.006 | 0.038 ± 0.004 |
| `trace/cli` at the tool's shipped cut | 0.029 ± 0.012 | 0.067 ± 0.022 | 0.021 ± 0.003 | 0.003 ± 0.002 |
| *reachable share of the truth (`trace/core` read)* | *0.61* | *0.70* | *0.33* | *0.18* |
| *reachable share, Shapley's read* | *0.60* | *0.26* | *0.07* | *0.03* |

## Reading

**Per-sequence axis (the root-cause-analysis reading).**

- Within a trace the trace arms and Granger score a pooled F1 of 0.34–0.51 at request and
  0.34–0.47 at session, with no downward trend from `xs` to `l`: the vocabulary grows 100-fold
  and the per-trace result holds.
- `trace/cli-atomic` is the best trace arm within the trace at every rung and both grains (by
  0.01–0.13 at request, 0.00–0.07 at session), through recall: it finds 0.74–0.97 of a request
  trace's true links against 0.37–0.72 for `trace/core`, at similar precision. On the type-level
  axis the same arm is the worst trace arm from `m` on.
- At request the lift over predicting every candidate pair is small or negative: the reference
  arms sit below it at `xs` and `s` and 0.03–0.04 above it at `m` and `l`; only
  `trace/cli-atomic` clears it at every rung. At session predict-all falls to 0.14–0.17 from
  `s` on and every trace arm clears it by 0.17 or more.
- Precision within the trace is 0.24–0.42 for the trace arms at every rung: of the links an arm
  draws inside a trace, fewer than half are true.
- The tool's shipped cut finds 0.5–4 % of a trace's true links (pooled F1 ≤ 0.08).

**Type-level axis on the reachable truth.**

- The reachable F1 of the trace arms and Granger is 0.29–0.43 at request and 0.31–0.46 at
  session across the four rungs. The benchmark F1's fall from 0.32 to 0.135 (request) and from
  0.36 to 0.16 (session) is mostly the reachable share falling, not the arms finding less of
  what they can see.
- **Precision is the limit, not recall.** The trace arms and Granger find 0.56–0.95 of the reachable
  truth; their precision is 0.17–0.36. At `l` request `trace/core` finds 90 % of the reachable edges at a
  precision of 0.19.
- **At request the lift over predicting every scored pair is modest**: 0.06–0.19 across rungs
  for `trace/core` (0.07 at `l`). At session the trivial predictor falls to 0.08 from `m` on
  and the lift is 0.26–0.37.
- **Orderings that change on this reading** (at the frozen τ, so read with the first limit
  above): Granger is above `trace/core` at `m` on both grains (+0.014 request, p = 0.24; +0.004
  session, p = 0.026); `trace/cli` is above `trace/cli-atomic` by more than on the benchmark F1
  (+0.042 at `m` request, p = 0.036; +0.079 at `m` session, p < 0.001); at `l` request saliency
  has the highest reachable F1 of the comparable arms (0.361 against 0.312 for `trace/core`),
  through precision (0.23 against 0.19).
- The shipped cut is high-precision and nearly empty on this reading too: precision 0.27–0.83,
  recall of the reachable truth 0.01–0.09.

**What does not change.** The benchmark's scorer output stays the type-level headline of every
per-rung findings file and of the paired hypotheses; no verdict there is rescored.
