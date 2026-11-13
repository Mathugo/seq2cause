# s / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/request | *the arm itself* | 0.216 ± 0.055 | 0.158 ± 0.041 | 0.345 ± 0.088 | 0.619 ± 0.023 | 0.189 ± 0.024 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.191 ± 0.065 | 0.140 ± 0.055 | 0.312 ± 0.070 | 0.551 ± 0.046 | 0.189 ± 0.024 | -0.024 ± 0.015 (p=0.025, n=5) | -0.068 ± 0.023 (p=0.003, n=5) | 3/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.234 ± 0.084 | 0.144 ± 0.061 | 0.676 ± 0.112 | 0.658 ± 0.090 | 0.189 ± 0.024 | +0.019 ± 0.047 (p=0.428, n=5) | +0.039 ± 0.073 (p=0.296, n=5) | 4/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.389 ± 0.020 | 0.242 ± 0.015 | 1.000 ± 0.000 | 0.829 ± 0.035 | 0.189 ± 0.024 | +0.173 ± 0.038 (p=0.001, n=5) | +0.210 ± 0.053 (p=0.001, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/session | *the arm itself* | 0.309 ± 0.064 | 0.232 ± 0.059 | 0.470 ± 0.044 | 0.805 ± 0.016 | 0.081 ± 0.015 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.289 ± 0.063 | 0.228 ± 0.056 | 0.397 ± 0.069 | 0.675 ± 0.034 | 0.081 ± 0.015 | -0.020 ± 0.021 (p=0.095, n=5) | -0.131 ± 0.031 (p=0.001, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.255 ± 0.062 | 0.154 ± 0.041 | 0.745 ± 0.120 | 0.820 ± 0.077 | 0.081 ± 0.015 | -0.054 ± 0.022 (p=0.006, n=5) | +0.015 ± 0.063 (p=0.616, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.324 ± 0.036 | 0.204 ± 0.029 | 0.793 ± 0.029 | 0.834 ± 0.019 | 0.081 ± 0.015 | +0.014 ± 0.030 (p=0.337, n=5) | +0.029 ± 0.032 (p=0.114, n=5) | 5/5 |

