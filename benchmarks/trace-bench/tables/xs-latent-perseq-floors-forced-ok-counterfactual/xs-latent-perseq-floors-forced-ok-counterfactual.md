# xs / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/request | *the arm itself* | 0.264 ± 0.048 | 0.194 ± 0.044 | 0.556 ± 0.263 | 0.600 ± 0.040 | 0.263 ± 0.047 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.247 ± 0.037 | 0.198 ± 0.023 | 0.343 ± 0.110 | 0.568 ± 0.052 | 0.263 ± 0.047 | -0.017 ± 0.078 (p=0.653, n=5) | -0.031 ± 0.047 (p=0.210, n=5) | 2/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.350 ± 0.050 | 0.223 ± 0.037 | 0.816 ± 0.054 | 0.767 ± 0.032 | 0.263 ± 0.047 | +0.086 ± 0.082 (p=0.080, n=5) | +0.167 ± 0.059 (p=0.003, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.474 ± 0.072 | 0.312 ± 0.059 | 1.000 ± 0.000 | 0.782 ± 0.014 | 0.263 ± 0.047 | +0.209 ± 0.066 (p=0.002, n=5) | +0.182 ± 0.037 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/session | *the arm itself* | 0.329 ± 0.024 | 0.239 ± 0.027 | 0.537 ± 0.063 | 0.749 ± 0.029 | 0.136 ± 0.019 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.339 ± 0.053 | 0.282 ± 0.036 | 0.433 ± 0.114 | 0.683 ± 0.058 | 0.136 ± 0.019 | +0.011 ± 0.050 (p=0.660, n=5) | -0.066 ± 0.033 (p=0.011, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.297 ± 0.052 | 0.182 ± 0.037 | 0.831 ± 0.042 | 0.842 ± 0.030 | 0.136 ± 0.019 | -0.031 ± 0.045 (p=0.193, n=5) | +0.094 ± 0.021 (p=0.001, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.345 ± 0.057 | 0.221 ± 0.040 | 0.792 ± 0.086 | 0.777 ± 0.034 | 0.136 ± 0.019 | +0.016 ± 0.059 (p=0.569, n=5) | +0.028 ± 0.061 (p=0.359, n=5) | 5/5 |

