# l / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/request | *the arm itself* | 0.438 ± 0.030 | 0.284 ± 0.027 | 0.971 ± 0.022 | 0.740 ± 0.030 | 0.409 ± 0.018 | — | — | — |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` adjacent | 0.398 ± 0.016 | 0.361 ± 0.015 | 0.443 ± 0.020 | 0.608 ± 0.009 | 0.409 ± 0.018 | -0.041 ± 0.025 (p=0.021, n=5) | -0.132 ± 0.026 (p=0.000, n=5) | 1/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` broadcast | 0.473 ± 0.023 | 0.332 ± 0.020 | 0.823 ± 0.024 | 0.708 ± 0.018 | 0.409 ± 0.018 | +0.035 ± 0.025 (p=0.035, n=5) | -0.032 ± 0.025 (p=0.044, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/topology` broadcast | 0.852 ± 0.014 | 0.742 ± 0.021 | 1.000 ± 0.000 | 0.943 ± 0.010 | 0.409 ± 0.018 | +0.413 ± 0.027 (p=0.000, n=5) | +0.202 ± 0.035 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/request | *the arm itself* | 0.429 ± 0.018 | 0.306 ± 0.025 | 0.722 ± 0.040 | 0.657 ± 0.009 | 0.401 ± 0.023 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.392 ± 0.023 | 0.355 ± 0.021 | 0.438 ± 0.028 | 0.607 ± 0.010 | 0.401 ± 0.023 | -0.037 ± 0.017 (p=0.008, n=5) | -0.050 ± 0.013 (p=0.001, n=5) | 1/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.465 ± 0.028 | 0.324 ± 0.024 | 0.821 ± 0.026 | 0.706 ± 0.021 | 0.401 ± 0.023 | +0.036 ± 0.019 (p=0.014, n=5) | +0.049 ± 0.023 (p=0.009, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.850 ± 0.013 | 0.740 ± 0.019 | 1.000 ± 0.000 | 0.943 ± 0.010 | 0.401 ± 0.023 | +0.422 ± 0.018 (p=0.000, n=5) | +0.286 ± 0.013 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/session | *the arm itself* | 0.385 ± 0.061 | 0.264 ± 0.056 | 0.733 ± 0.032 | 0.847 ± 0.023 | 0.143 ± 0.010 | — | — | — |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` adjacent | 0.343 ± 0.022 | 0.332 ± 0.016 | 0.356 ± 0.030 | 0.652 ± 0.014 | 0.143 ± 0.010 | -0.041 ± 0.050 (p=0.137, n=5) | -0.195 ± 0.016 (p=0.000, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` broadcast | 0.320 ± 0.025 | 0.201 ± 0.017 | 0.781 ± 0.048 | 0.802 ± 0.028 | 0.143 ± 0.010 | -0.065 ± 0.047 (p=0.036, n=5) | -0.045 ± 0.020 (p=0.007, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/topology` broadcast | 0.576 ± 0.012 | 0.493 ± 0.016 | 0.691 ± 0.012 | 0.815 ± 0.007 | 0.143 ± 0.010 | +0.191 ± 0.059 (p=0.002, n=5) | -0.031 ± 0.026 (p=0.056, n=5) | 5/5 |
| trace/core/shipped/frozen/session | *the arm itself* | 0.383 ± 0.055 | 0.259 ± 0.052 | 0.767 ± 0.055 | 0.840 ± 0.013 | 0.143 ± 0.010 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.343 ± 0.022 | 0.332 ± 0.016 | 0.356 ± 0.030 | 0.652 ± 0.014 | 0.143 ± 0.010 | -0.040 ± 0.045 (p=0.119, n=5) | -0.188 ± 0.016 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.320 ± 0.025 | 0.201 ± 0.017 | 0.781 ± 0.048 | 0.802 ± 0.028 | 0.143 ± 0.010 | -0.063 ± 0.044 (p=0.032, n=5) | -0.038 ± 0.025 (p=0.027, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.576 ± 0.012 | 0.493 ± 0.016 | 0.691 ± 0.012 | 0.815 ± 0.007 | 0.143 ± 0.010 | +0.193 ± 0.053 (p=0.001, n=5) | -0.025 ± 0.018 (p=0.034, n=5) | 5/5 |

