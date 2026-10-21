# xs / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/request | *the arm itself* | 0.505 ± 0.052 | 0.373 ± 0.062 | 0.851 ± 0.196 | 0.607 ± 0.084 | 0.500 ± 0.044 | — | — | — |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` adjacent | 0.538 ± 0.067 | 0.495 ± 0.050 | 0.599 ± 0.135 | 0.677 ± 0.074 | 0.500 ± 0.044 | +0.033 ± 0.111 (p=0.537, n=5) | +0.070 ± 0.151 (p=0.357, n=5) | 2/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` broadcast | 0.529 ± 0.057 | 0.390 ± 0.038 | 0.823 ± 0.111 | 0.678 ± 0.100 | 0.500 ± 0.044 | +0.024 ± 0.083 (p=0.545, n=5) | +0.071 ± 0.167 (p=0.395, n=5) | 4/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/topology` broadcast | 0.807 ± 0.083 | 0.683 ± 0.112 | 1.000 ± 0.000 | 0.884 ± 0.077 | 0.500 ± 0.044 | +0.302 ± 0.042 (p=0.000, n=5) | +0.276 ± 0.037 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/request | *the arm itself* | 0.423 ± 0.061 | 0.380 ± 0.061 | 0.524 ± 0.172 | 0.582 ± 0.022 | 0.456 ± 0.035 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.496 ± 0.080 | 0.455 ± 0.052 | 0.551 ± 0.138 | 0.666 ± 0.080 | 0.456 ± 0.035 | +0.073 ± 0.131 (p=0.284, n=5) | +0.084 ± 0.098 (p=0.127, n=5) | 2/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.481 ± 0.065 | 0.344 ± 0.046 | 0.801 ± 0.115 | 0.662 ± 0.120 | 0.456 ± 0.035 | +0.058 ± 0.107 (p=0.292, n=5) | +0.080 ± 0.135 (p=0.254, n=5) | 4/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.800 ± 0.079 | 0.672 ± 0.109 | 1.000 ± 0.000 | 0.899 ± 0.065 | 0.456 ± 0.035 | +0.377 ± 0.030 (p=0.000, n=5) | +0.317 ± 0.050 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/session | *the arm itself* | 0.443 ± 0.047 | 0.398 ± 0.046 | 0.503 ± 0.073 | 0.728 ± 0.038 | 0.304 ± 0.024 | — | — | — |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` adjacent | 0.414 ± 0.067 | 0.529 ± 0.074 | 0.341 ± 0.064 | 0.642 ± 0.037 | 0.304 ± 0.024 | -0.028 ± 0.033 (p=0.128, n=5) | -0.086 ± 0.014 (p=0.000, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` broadcast | 0.373 ± 0.019 | 0.255 ± 0.021 | 0.700 ± 0.045 | 0.664 ± 0.039 | 0.304 ± 0.024 | -0.070 ± 0.039 (p=0.016, n=5) | -0.064 ± 0.049 (p=0.044, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/topology` broadcast | 0.426 ± 0.052 | 0.395 ± 0.072 | 0.467 ± 0.031 | 0.660 ± 0.024 | 0.304 ± 0.024 | -0.016 ± 0.052 (p=0.518, n=5) | -0.068 ± 0.043 (p=0.024, n=5) | 5/5 |
| trace/core/shipped/frozen/session | *the arm itself* | 0.433 ± 0.049 | 0.394 ± 0.037 | 0.482 ± 0.069 | 0.712 ± 0.025 | 0.303 ± 0.028 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.414 ± 0.066 | 0.531 ± 0.076 | 0.339 ± 0.060 | 0.642 ± 0.034 | 0.303 ± 0.028 | -0.020 ± 0.042 (p=0.354, n=5) | -0.070 ± 0.034 (p=0.010, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.382 ± 0.042 | 0.262 ± 0.042 | 0.715 ± 0.040 | 0.675 ± 0.043 | 0.303 ± 0.028 | -0.052 ± 0.042 (p=0.051, n=5) | -0.037 ± 0.051 (p=0.178, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.406 ± 0.037 | 0.372 ± 0.053 | 0.450 ± 0.027 | 0.647 ± 0.018 | 0.303 ± 0.028 | -0.028 ± 0.065 (p=0.397, n=5) | -0.065 ± 0.036 (p=0.016, n=5) | 5/5 |

