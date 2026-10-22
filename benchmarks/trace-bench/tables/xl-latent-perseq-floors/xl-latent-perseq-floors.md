# xl / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/request | *the arm itself* | 0.369 ± 0.014 | 0.226 ± 0.011 | 0.998 ± 0.001 | 0.676 ± 0.036 | 0.367 ± 0.014 | — | — | — |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` adjacent | 0.355 ± 0.005 | 0.307 ± 0.005 | 0.421 ± 0.017 | 0.591 ± 0.010 | 0.367 ± 0.014 | -0.014 ± 0.018 (p=0.142, n=5) | -0.085 ± 0.036 (p=0.006, n=5) | 2/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/bigram` broadcast | 0.417 ± 0.015 | 0.287 ± 0.013 | 0.761 ± 0.026 | 0.657 ± 0.009 | 0.367 ± 0.014 | +0.048 ± 0.011 (p=0.001, n=5) | -0.019 ± 0.031 (p=0.235, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/request | `floor/topology` broadcast | 0.800 ± 0.025 | 0.668 ± 0.034 | 1.000 ± 0.000 | 0.929 ± 0.009 | 0.367 ± 0.014 | +0.431 ± 0.016 (p=0.000, n=5) | +0.253 ± 0.039 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/request | *the arm itself* | 0.380 ± 0.013 | 0.253 ± 0.012 | 0.764 ± 0.009 | 0.638 ± 0.024 | 0.367 ± 0.014 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.355 ± 0.005 | 0.307 ± 0.005 | 0.421 ± 0.017 | 0.591 ± 0.010 | 0.367 ± 0.014 | -0.025 ± 0.017 (p=0.028, n=5) | -0.047 ± 0.022 (p=0.008, n=5) | 2/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.417 ± 0.015 | 0.287 ± 0.013 | 0.761 ± 0.026 | 0.657 ± 0.009 | 0.367 ± 0.014 | +0.037 ± 0.006 (p=0.000, n=5) | +0.018 ± 0.016 (p=0.066, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.800 ± 0.025 | 0.668 ± 0.034 | 1.000 ± 0.000 | 0.929 ± 0.009 | 0.367 ± 0.014 | +0.420 ± 0.020 (p=0.000, n=5) | +0.290 ± 0.030 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/cli-atomic/shipped/frozen/session | *the arm itself* | 0.230 ± 0.060 | 0.134 ± 0.042 | 0.887 ± 0.049 | 0.821 ± 0.034 | 0.132 ± 0.004 | — | — | — |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` adjacent | 0.317 ± 0.004 | 0.292 ± 0.007 | 0.347 ± 0.012 | 0.645 ± 0.005 | 0.132 ± 0.004 | +0.087 ± 0.060 (p=0.031, n=5) | -0.176 ± 0.035 (p=0.000, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/bigram` broadcast | 0.309 ± 0.009 | 0.197 ± 0.007 | 0.716 ± 0.004 | 0.773 ± 0.003 | 0.132 ± 0.004 | +0.079 ± 0.056 (p=0.034, n=5) | -0.049 ± 0.033 (p=0.030, n=5) | 5/5 |
| trace/cli-atomic/shipped/frozen/session | `floor/topology` broadcast | 0.572 ± 0.015 | 0.479 ± 0.018 | 0.709 ± 0.018 | 0.825 ± 0.009 | 0.132 ± 0.004 | +0.342 ± 0.061 (p=0.000, n=5) | +0.004 ± 0.035 (p=0.806, n=5) | 5/5 |
| trace/core/shipped/frozen/session | *the arm itself* | 0.250 ± 0.042 | 0.146 ± 0.029 | 0.885 ± 0.023 | 0.824 ± 0.021 | 0.132 ± 0.004 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.317 ± 0.004 | 0.292 ± 0.007 | 0.347 ± 0.012 | 0.645 ± 0.005 | 0.132 ± 0.004 | +0.067 ± 0.043 (p=0.024, n=5) | -0.179 ± 0.023 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.309 ± 0.009 | 0.197 ± 0.007 | 0.716 ± 0.004 | 0.773 ± 0.003 | 0.132 ± 0.004 | +0.059 ± 0.039 (p=0.028, n=5) | -0.052 ± 0.020 (p=0.004, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.572 ± 0.015 | 0.479 ± 0.018 | 0.709 ± 0.018 | 0.825 ± 0.009 | 0.132 ± 0.004 | +0.322 ± 0.037 (p=0.000, n=5) | +0.001 ± 0.025 (p=0.933, n=5) | 5/5 |

