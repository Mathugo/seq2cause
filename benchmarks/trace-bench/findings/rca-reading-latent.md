# Findings — the root-cause-analysis reading: the per-sequence axis first, then the type-level axis on the reachable truth (xs … xl / latent, five seeds)

Score-side reading of 2026-10-04 over the landed test reads of the five rungs
(`results/{xs,s,m,l,xl}/latent/seed={0..4}/`, tables `tables/{xs,s,m,l,xl}-latent/`; the `xl` column
added 2026-10-07 when that rung landed); no method run, no
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
last table of each grain: 0.48 → 0.06 at request, 0.61 → 0.09 at session), and it drags every
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

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.423 ± 0.061 | 0.343 ± 0.060 | 0.444 ± 0.015 | 0.429 ± 0.018 | 0.380 ± 0.013 |
| `trace/cli` | 0.431 ± 0.046 | 0.360 ± 0.068 | 0.448 ± 0.012 | 0.427 ± 0.024 | 0.380 ± 0.014 |
| `trace/cli-atomic` | 0.505 ± 0.052 | 0.488 ± 0.058 | 0.491 ± 0.013 | 0.438 ± 0.030 | 0.369 ± 0.014 |
| `baseline/granger` | 0.426 ± 0.052 | 0.376 ± 0.034 | 0.447 ± 0.019 | 0.427 ± 0.017 | 0.383 ± 0.011 |
| `baseline/saliency` | 0.352 ± 0.121 | 0.278 ± 0.065 | 0.371 ± 0.011 | 0.389 ± 0.033 | 0.369 ± 0.012 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.339 ± 0.047 | 0.337 ± 0.041 | 0.437 ± 0.013 | 0.465 ± 0.024 | 0.379 ± 0.018 |
| `trace/cli` at the tool's shipped cut | 0.039 ± 0.032 | 0.082 ± 0.042 | 0.057 ± 0.012 | 0.050 ± 0.013 | 0.021 ± 0.015 |
| *predict every candidate pair* | *0.456* | *0.397* | *0.409* | *0.401* | *0.367* |
| *scoreable fraction of the probed sequences* | *0.85* | *0.97* | *0.98* | *0.95* | *0.91* |

Per-sequence precision / recall (pooled):

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.38 / 0.52 | 0.32 / 0.37 | 0.37 / 0.55 | 0.31 / 0.72 | 0.25 / 0.76 |
| `trace/cli` | 0.37 / 0.54 | 0.34 / 0.38 | 0.37 / 0.57 | 0.31 / 0.72 | 0.25 / 0.76 |
| `trace/cli-atomic` | 0.37 / 0.85 | 0.36 / 0.74 | 0.35 / 0.83 | 0.28 / 0.97 | 0.23 / 1.00 |
| `baseline/granger` | 0.39 / 0.48 | 0.32 / 0.47 | 0.38 / 0.54 | 0.32 / 0.66 | 0.27 / 0.66 |
| `baseline/saliency` | 0.40 / 0.33 | 0.32 / 0.25 | 0.34 / 0.41 | 0.31 / 0.52 | 0.23 / 0.99 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.28 / 0.44 | 0.25 / 0.50 | 0.32 / 0.67 | 0.33 / 0.77 | 0.24 / 0.96 |

Type-level axis, reachable truth only — directed F1 (five-seed mean ± sd):

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.416 ± 0.056 | 0.382 ± 0.018 | 0.413 ± 0.025 | 0.312 ± 0.029 | 0.291 ± 0.007 |
| `trace/cli` | 0.405 ± 0.051 | 0.391 ± 0.022 | 0.407 ± 0.030 | 0.309 ± 0.039 | 0.291 ± 0.006 |
| `trace/cli-atomic` | 0.406 ± 0.059 | 0.379 ± 0.019 | 0.365 ± 0.014 | 0.287 ± 0.031 | 0.264 ± 0.005 |
| `baseline/granger` | 0.419 ± 0.048 | 0.365 ± 0.029 | 0.427 ± 0.011 | 0.326 ± 0.026 | 0.315 ± 0.007 |
| `baseline/saliency` | 0.378 ± 0.053 | 0.341 ± 0.031 | 0.342 ± 0.011 | 0.361 ± 0.021 | 0.265 ± 0.005 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.333 ± 0.065 | 0.299 ± 0.019 | 0.358 ± 0.012 | 0.384 ± 0.026 | 0.320 ± 0.015 |
| `trace/cli` at the tool's shipped cut | 0.066 ± 0.031 | 0.099 ± 0.042 | 0.112 ± 0.023 | 0.049 ± 0.024 | 0.023 ± 0.020 |
| *predict every scored pair* | *0.358* | *0.253* | *0.224* | *0.241* | *0.261* |

Type-level axis, reachable truth only — precision / recall:

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.30 / 0.78 | 0.27 / 0.64 | 0.28 / 0.81 | 0.19 / 0.90 | 0.17 / 0.94 |
| `trace/cli` | 0.28 / 0.79 | 0.29 / 0.63 | 0.27 / 0.82 | 0.19 / 0.90 | 0.17 / 0.94 |
| `trace/cli-atomic` | 0.28 / 0.87 | 0.26 / 0.69 | 0.23 / 0.84 | 0.17 / 0.95 | 0.15 / 1.00 |
| `baseline/granger` | 0.30 / 0.71 | 0.25 / 0.71 | 0.29 / 0.79 | 0.20 / 0.85 | 0.19 / 0.83 |
| `baseline/saliency` | 0.28 / 0.61 | 0.22 / 0.77 | 0.21 / 0.88 | 0.23 / 0.85 | 0.15 / 1.00 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.24 / 0.56 | 0.20 / 0.58 | 0.24 / 0.67 | 0.26 / 0.70 | 0.19 / 0.95 |

Type-level axis, the benchmark's directed F1 over the whole truth (the headline of the per-rung findings), with the share of the truth the read can reach:

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.321 ± 0.043 | 0.289 ± 0.013 | 0.204 ± 0.009 | 0.135 ± 0.006 | 0.088 ± 0.002 |
| `trace/cli` | 0.317 ± 0.045 | 0.292 ± 0.014 | 0.205 ± 0.009 | 0.135 ± 0.007 | 0.088 ± 0.002 |
| `trace/cli-atomic` | 0.307 ± 0.043 | 0.287 ± 0.016 | 0.196 ± 0.005 | 0.133 ± 0.006 | 0.089 ± 0.002 |
| `baseline/granger` | 0.321 ± 0.041 | 0.283 ± 0.014 | 0.206 ± 0.007 | 0.134 ± 0.006 | 0.083 ± 0.003 |
| `baseline/saliency` | 0.279 ± 0.040 | 0.268 ± 0.025 | 0.192 ± 0.004 | 0.139 ± 0.007 | 0.089 ± 0.002 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.255 ± 0.050 | 0.158 ± 0.014 | 0.070 ± 0.004 | 0.035 ± 0.003 | 0.022 ± 0.001 |
| `trace/cli` at the tool's shipped cut | 0.033 ± 0.016 | 0.051 ± 0.023 | 0.024 ± 0.005 | 0.006 ± 0.003 | 0.001 ± 0.001 |
| *reachable share of the truth (`trace/core` read)* | *0.48* | *0.48* | *0.20* | *0.12* | *0.06* |
| *reachable share, Shapley's read* | *0.49* | *0.22* | *0.06* | *0.03* | *0.01* |

### session grain

Per-sequence axis — pooled directed F1 within the trace (five-seed mean ± sd):

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.433 ± 0.049 | 0.346 ± 0.063 | 0.412 ± 0.023 | 0.383 ± 0.055 | 0.250 ± 0.042 |
| `trace/cli` | 0.414 ± 0.046 | 0.350 ± 0.063 | 0.406 ± 0.020 | 0.359 ± 0.053 | 0.232 ± 0.033 |
| `trace/cli-atomic` | 0.443 ± 0.047 | 0.423 ± 0.057 | 0.470 ± 0.012 | 0.385 ± 0.061 | 0.230 ± 0.060 |
| `baseline/granger` | 0.419 ± 0.052 | 0.337 ± 0.062 | 0.407 ± 0.022 | 0.364 ± 0.027 | 0.276 ± 0.030 |
| `baseline/saliency` | 0.296 ± 0.055 | 0.161 ± 0.018 | 0.313 ± 0.053 | 0.359 ± 0.030 | 0.287 ± 0.071 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.289 ± 0.029 | 0.214 ± 0.022 | 0.194 ± 0.010 | 0.189 ± 0.015 | 0.142 ± 0.007 |
| `trace/cli` at the tool's shipped cut | 0.042 ± 0.014 | 0.045 ± 0.026 | 0.016 ± 0.002 | 0.009 ± 0.005 | 0.005 ± 0.003 |
| *predict every candidate pair* | *0.303* | *0.173* | *0.151* | *0.143* | *0.132* |
| *scoreable fraction of the probed sequences* | *0.97* | *0.99* | *0.99* | *0.90* | *0.85* |

Per-sequence precision / recall (pooled):

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.39 / 0.48 | 0.33 / 0.37 | 0.34 / 0.53 | 0.26 / 0.77 | 0.15 / 0.88 |
| `trace/cli` | 0.42 / 0.42 | 0.34 / 0.36 | 0.34 / 0.50 | 0.24 / 0.73 | 0.13 / 0.86 |
| `trace/cli-atomic` | 0.40 / 0.50 | 0.36 / 0.52 | 0.37 / 0.65 | 0.26 / 0.73 | 0.13 / 0.89 |
| `baseline/granger` | 0.39 / 0.46 | 0.32 / 0.36 | 0.33 / 0.53 | 0.24 / 0.71 | 0.17 / 0.74 |
| `baseline/saliency` | 0.29 / 0.31 | 0.37 / 0.10 | 0.37 / 0.29 | 0.27 / 0.54 | 0.19 / 0.69 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.22 / 0.44 | 0.13 / 0.53 | 0.11 / 0.72 | 0.11 / 0.79 | 0.08 / 0.95 |

Type-level axis, reachable truth only — directed F1 (five-seed mean ± sd):

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.443 ± 0.015 | 0.390 ± 0.023 | 0.442 ± 0.010 | 0.338 ± 0.052 | 0.234 ± 0.043 |
| `trace/cli` | 0.438 ± 0.020 | 0.399 ± 0.019 | 0.461 ± 0.008 | 0.319 ± 0.054 | 0.212 ± 0.038 |
| `trace/cli-atomic` | 0.413 ± 0.022 | 0.364 ± 0.020 | 0.381 ± 0.010 | 0.307 ± 0.050 | 0.200 ± 0.055 |
| `baseline/granger` | 0.436 ± 0.027 | 0.399 ± 0.018 | 0.447 ± 0.009 | 0.336 ± 0.016 | 0.268 ± 0.038 |
| `baseline/saliency` | 0.391 ± 0.017 | 0.301 ± 0.015 | 0.315 ± 0.055 | 0.275 ± 0.018 | 0.242 ± 0.082 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.337 ± 0.015 | 0.231 ± 0.010 | 0.182 ± 0.006 | 0.177 ± 0.016 | 0.135 ± 0.006 |
| `trace/cli` at the tool's shipped cut | 0.048 ± 0.020 | 0.082 ± 0.020 | 0.055 ± 0.006 | 0.016 ± 0.007 | 0.006 ± 0.008 |
| *predict every scored pair* | *0.299* | *0.124* | *0.077* | *0.078* | *0.092* |

Type-level axis, reachable truth only — precision / recall:

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.35 / 0.59 | 0.29 / 0.58 | 0.32 / 0.70 | 0.22 / 0.73 | 0.14 / 0.83 |
| `trace/cli` | 0.35 / 0.60 | 0.31 / 0.58 | 0.35 / 0.68 | 0.21 / 0.72 | 0.12 / 0.83 |
| `trace/cli-atomic` | 0.32 / 0.59 | 0.26 / 0.61 | 0.26 / 0.70 | 0.20 / 0.68 | 0.12 / 0.81 |
| `baseline/granger` | 0.34 / 0.60 | 0.31 / 0.56 | 0.33 / 0.69 | 0.22 / 0.73 | 0.16 / 0.73 |
| `baseline/saliency` | 0.27 / 0.74 | 0.19 / 0.73 | 0.20 / 0.77 | 0.17 / 0.81 | 0.15 / 0.82 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.23 / 0.64 | 0.14 / 0.63 | 0.10 / 0.74 | 0.10 / 0.75 | 0.07 / 0.96 |

Type-level axis, the benchmark's directed F1 over the whole truth (the headline of the per-rung findings), with the share of the truth the read can reach:

| arm | xs | s | m | l | xl |
|---|---|---|---|---|---|
| `trace/core` | 0.357 ± 0.021 | 0.342 ± 0.018 | 0.270 ± 0.010 | 0.161 ± 0.011 | 0.097 ± 0.009 |
| `trace/cli` | 0.353 ± 0.021 | 0.349 ± 0.015 | 0.274 ± 0.010 | 0.155 ± 0.013 | 0.093 ± 0.009 |
| `trace/cli-atomic` | 0.332 ± 0.011 | 0.324 ± 0.015 | 0.247 ± 0.009 | 0.148 ± 0.014 | 0.088 ± 0.010 |
| `baseline/granger` | 0.354 ± 0.028 | 0.348 ± 0.015 | 0.271 ± 0.011 | 0.161 ± 0.007 | 0.095 ± 0.009 |
| `baseline/saliency` | 0.324 ± 0.016 | 0.277 ± 0.013 | 0.222 ± 0.025 | 0.153 ± 0.005 | 0.095 ± 0.013 |
| `baseline/shapley` (own 2k / 500 sample from `s`) | 0.286 ± 0.013 | 0.151 ± 0.013 | 0.070 ± 0.006 | 0.038 ± 0.004 | 0.023 ± 0.002 |
| `trace/cli` at the tool's shipped cut | 0.029 ± 0.012 | 0.067 ± 0.022 | 0.021 ± 0.003 | 0.003 ± 0.002 | 0.001 ± 0.001 |
| *reachable share of the truth (`trace/core` read)* | *0.61* | *0.70* | *0.33* | *0.18* | *0.09* |
| *reachable share, Shapley's read* | *0.60* | *0.26* | *0.07* | *0.03* | *0.01* |

## Reading

**Per-sequence axis (the root-cause-analysis reading).**

- Within a trace the trace arms and Granger score a pooled F1 of 0.34–0.51 at request and
  0.23–0.47 at session, with no downward trend at request from `xs` to `xl`: the vocabulary
  grows a hundredfold and the per-trace result holds (0.37–0.38 at `xl`). At session `xl` is the
  first rung with a fall (0.23–0.28 for the trace arms and Granger against 0.36–0.39 at `l`),
  through precision (0.13–0.17): the per-sequence candidate set of a long session trace grows
  with the trace.
- `trace/cli-atomic` is the best trace arm within the trace at every rung and both grains
  through `l` (by 0.01–0.13 at request, 0.00–0.07 at session), through recall: it finds
  0.74–0.97 of a request trace's true links against 0.37–0.72 for `trace/core`, at similar
  precision. At `xl` it is the worst trace arm on both axes: its frozen τ is 0 or the grid floor
  on every seed, so it predicts every candidate pair within the trace (recall 1.00, precision
  0.23) and lands on the predict-all value (0.369 against 0.367). On the type-level axis the
  same arm is the worst trace arm from `m` on.
- At request the lift over predicting every candidate pair is small or negative: the reference
  arms sit below it at `xs` and `s`, 0.03–0.04 above it at `m` and `l` and 0.01 above it at
  `xl`; `trace/cli-atomic` clears it at every rung through `l` and sits on it at `xl`. At
  session predict-all falls to 0.13–0.17 from `s` on and every trace arm clears it by 0.17 or
  more through `l`, by 0.10–0.12 at `xl`.
- Precision within the trace is 0.24–0.42 for the trace arms through `l` and 0.13–0.25 at `xl`:
  of the links an arm draws inside a trace, fewer than half are true.
- The tool's shipped cut finds 0.5–4 % of a trace's true links (pooled F1 ≤ 0.08).

**Type-level axis on the reachable truth.**

- The reachable F1 of the trace arms and Granger is 0.26–0.43 at request and 0.20–0.46 at
  session across the five rungs. The benchmark F1's fall from 0.32 to 0.088 (request) and from
  0.36 to 0.097 (session) is mostly the reachable share falling (0.48 → 0.06, 0.61 → 0.09),
  not the arms finding less of what they can see — though at `xl` the reachable F1 itself is
  down too (0.29 / 0.23 for `trace/core`), through precision.
- **Precision is the limit, not recall.** The trace arms and Granger find 0.56–1.00 of the reachable
  truth; their precision is 0.12–0.36. At `l` request `trace/core` finds 90 % of the reachable edges at a
  precision of 0.19; at `xl` request 94 % at 0.17, and `trace/cli-atomic` and saliency find all of
  them at 0.15 (they predict every scored pair).
- **At request the lift over predicting every scored pair is modest**: 0.03–0.19 across rungs
  for `trace/core` (0.07 at `l`, 0.03 at `xl`, where two arms *are* that predictor). At session
  the trivial predictor falls to 0.08–0.09 from `m` on and the lift is 0.14–0.37 (0.14 at `xl`).
- **Orderings that change on this reading** (at the frozen τ, so read with the first limit
  above): Granger is above `trace/core` at `m` on both grains (+0.014 request, p = 0.24; +0.004
  session, p = 0.026); `trace/cli` is above `trace/cli-atomic` by more than on the benchmark F1
  (+0.042 at `m` request, p = 0.036; +0.079 at `m` session, p < 0.001); at `l` request saliency
  has the highest reachable F1 of the comparable arms (0.361 against 0.312 for `trace/core`),
  through precision (0.23 against 0.19); at `xl` Granger has it on both grains (0.315 against
  0.291 at request, 0.268 against 0.234 at session), through precision again (0.19 / 0.16
  against 0.17 / 0.14), while on the benchmark F1 it is the lowest comparable arm at request.
- The shipped cut is high-precision and nearly empty on this reading too: precision 0.27–0.83,
  recall of the reachable truth 0.01–0.09.

**What does not change.** The benchmark's scorer output stays the type-level headline of every
per-rung findings file and of the paired hypotheses; no verdict there is rescored.
