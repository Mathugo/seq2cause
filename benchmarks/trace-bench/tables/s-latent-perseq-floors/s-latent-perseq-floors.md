# s / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/request | *the arm itself* | 0.488 ± 0.058 | 0.365 ± 0.050 | 0.738 ± 0.074 | 0.691 ± 0.053 | 0.430 ± 0.039 | — | — | — |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` adjacent | 0.409 ± 0.090 | 0.372 ± 0.095 | 0.457 ± 0.082 | 0.598 ± 0.064 | 0.430 ± 0.039 | -0.079 ± 0.060 (p=0.042, n=5) | -0.093 ± 0.062 (p=0.028, n=5) | 1/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` broadcast | 0.421 ± 0.073 | 0.292 ± 0.062 | 0.773 ± 0.118 | 0.593 ± 0.085 | 0.430 ± 0.039 | -0.067 ± 0.047 (p=0.034, n=5) | -0.098 ± 0.089 (p=0.069, n=5) | 3/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/topology` broadcast | 0.838 ± 0.043 | 0.723 ± 0.063 | 1.000 ± 0.000 | 0.926 ± 0.025 | 0.430 ± 0.039 | +0.350 ± 0.019 (p=0.000, n=5) | +0.235 ± 0.038 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/request | *the arm itself* | 0.343 ± 0.060 | 0.321 ± 0.065 | 0.370 ± 0.054 | 0.592 ± 0.024 | 0.397 ± 0.050 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.377 ± 0.074 | 0.342 ± 0.083 | 0.422 ± 0.061 | 0.589 ± 0.046 | 0.397 ± 0.050 | +0.033 ± 0.016 (p=0.010, n=5) | -0.003 ± 0.026 (p=0.803, n=5) | 1/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.380 ± 0.084 | 0.258 ± 0.066 | 0.743 ± 0.139 | 0.574 ± 0.079 | 0.397 ± 0.050 | +0.037 ± 0.038 (p=0.096, n=5) | -0.018 ± 0.057 (p=0.520, n=5) | 2/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.833 ± 0.051 | 0.717 ± 0.073 | 1.000 ± 0.000 | 0.933 ± 0.020 | 0.397 ± 0.050 | +0.490 ± 0.036 (p=0.000, n=5) | +0.341 ± 0.029 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/session | *the arm itself* | 0.423 ± 0.057 | 0.359 ± 0.051 | 0.521 ± 0.084 | 0.791 ± 0.032 | 0.174 ± 0.022 | — | — | — |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` adjacent | 0.323 ± 0.067 | 0.350 ± 0.073 | 0.301 ± 0.064 | 0.624 ± 0.033 | 0.174 ± 0.022 | -0.101 ± 0.026 (p=0.001, n=5) | -0.167 ± 0.016 (p=0.000, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` broadcast | 0.269 ± 0.063 | 0.175 ± 0.045 | 0.600 ± 0.120 | 0.689 ± 0.080 | 0.174 ± 0.022 | -0.154 ± 0.031 (p=0.000, n=5) | -0.102 ± 0.052 (p=0.012, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/topology` broadcast | 0.494 ± 0.063 | 0.426 ± 0.076 | 0.591 ± 0.037 | 0.756 ± 0.024 | 0.174 ± 0.022 | +0.070 ± 0.014 (p=0.000, n=5) | -0.035 ± 0.012 (p=0.003, n=5) | 5/5 |
| trace/core/shipped/frozen/session | *the arm itself* | 0.346 ± 0.063 | 0.326 ± 0.064 | 0.372 ± 0.066 | 0.743 ± 0.022 | 0.173 ± 0.023 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.323 ± 0.067 | 0.349 ± 0.075 | 0.302 ± 0.064 | 0.624 ± 0.032 | 0.173 ± 0.023 | -0.023 ± 0.026 (p=0.119, n=5) | -0.119 ± 0.029 (p=0.001, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.271 ± 0.063 | 0.176 ± 0.044 | 0.602 ± 0.122 | 0.692 ± 0.078 | 0.173 ± 0.023 | -0.075 ± 0.026 (p=0.003, n=5) | -0.051 ± 0.064 (p=0.150, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.493 ± 0.062 | 0.426 ± 0.075 | 0.587 ± 0.034 | 0.755 ± 0.022 | 0.173 ± 0.023 | +0.146 ± 0.050 (p=0.003, n=5) | +0.012 ± 0.033 (p=0.479, n=5) | 5/5 |

