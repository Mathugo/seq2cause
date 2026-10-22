# m / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/request | *the arm itself* | 0.491 ± 0.013 | 0.348 ± 0.010 | 0.832 ± 0.025 | 0.733 ± 0.014 | 0.409 ± 0.012 | — | — | — |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` adjacent | 0.392 ± 0.020 | 0.357 ± 0.014 | 0.434 ± 0.030 | 0.604 ± 0.015 | 0.409 ± 0.012 | -0.099 ± 0.022 (p=0.001, n=5) | -0.129 ± 0.017 (p=0.000, n=5) | 0/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` broadcast | 0.453 ± 0.026 | 0.310 ± 0.018 | 0.844 ± 0.052 | 0.698 ± 0.041 | 0.409 ± 0.012 | -0.038 ± 0.023 (p=0.022, n=5) | -0.035 ± 0.038 (p=0.108, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/topology` broadcast | 0.867 ± 0.019 | 0.766 ± 0.030 | 1.000 ± 0.000 | 0.948 ± 0.014 | 0.409 ± 0.012 | +0.376 ± 0.020 (p=0.000, n=5) | +0.215 ± 0.023 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/request | *the arm itself* | 0.444 ± 0.015 | 0.372 ± 0.026 | 0.554 ± 0.022 | 0.643 ± 0.013 | 0.409 ± 0.012 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.392 ± 0.020 | 0.357 ± 0.014 | 0.434 ± 0.030 | 0.604 ± 0.015 | 0.409 ± 0.012 | -0.053 ± 0.009 (p=0.000, n=5) | -0.040 ± 0.012 (p=0.002, n=5) | 0/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.453 ± 0.026 | 0.310 ± 0.018 | 0.844 ± 0.052 | 0.698 ± 0.041 | 0.409 ± 0.012 | +0.009 ± 0.018 (p=0.332, n=5) | +0.055 ± 0.036 (p=0.027, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.867 ± 0.019 | 0.766 ± 0.030 | 1.000 ± 0.000 | 0.948 ± 0.014 | 0.409 ± 0.012 | +0.423 ± 0.032 (p=0.000, n=5) | +0.304 ± 0.023 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/session | *the arm itself* | 0.470 ± 0.012 | 0.370 ± 0.004 | 0.647 ± 0.039 | 0.846 ± 0.014 | 0.151 ± 0.008 | — | — | — |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` adjacent | 0.335 ± 0.015 | 0.334 ± 0.020 | 0.336 ± 0.020 | 0.642 ± 0.009 | 0.151 ± 0.008 | -0.136 ± 0.010 (p=0.000, n=5) | -0.205 ± 0.012 (p=0.000, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` broadcast | 0.299 ± 0.029 | 0.186 ± 0.023 | 0.759 ± 0.044 | 0.779 ± 0.022 | 0.151 ± 0.008 | -0.172 ± 0.026 (p=0.000, n=5) | -0.068 ± 0.025 (p=0.004, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/topology` broadcast | 0.589 ± 0.025 | 0.518 ± 0.023 | 0.684 ± 0.036 | 0.814 ± 0.020 | 0.151 ± 0.008 | +0.119 ± 0.025 (p=0.000, n=5) | -0.033 ± 0.014 (p=0.006, n=5) | 5/5 |
| trace/core/shipped/frozen/session | *the arm itself* | 0.412 ± 0.023 | 0.339 ± 0.021 | 0.525 ± 0.032 | 0.828 ± 0.008 | 0.151 ± 0.008 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.335 ± 0.015 | 0.334 ± 0.020 | 0.336 ± 0.020 | 0.642 ± 0.009 | 0.151 ± 0.008 | -0.077 ± 0.015 (p=0.000, n=5) | -0.187 ± 0.014 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.299 ± 0.029 | 0.186 ± 0.023 | 0.759 ± 0.044 | 0.779 ± 0.022 | 0.151 ± 0.008 | -0.113 ± 0.020 (p=0.000, n=5) | -0.049 ± 0.024 (p=0.009, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.589 ± 0.025 | 0.518 ± 0.023 | 0.684 ± 0.036 | 0.814 ± 0.020 | 0.151 ± 0.008 | +0.177 ± 0.044 (p=0.001, n=5) | -0.014 ± 0.025 (p=0.269, n=5) | 5/5 |

