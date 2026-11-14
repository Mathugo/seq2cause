# xl / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/request | *the arm itself* | 0.206 ± 0.005 | 0.118 ± 0.003 | 0.837 ± 0.007 | 0.630 ± 0.036 | 0.183 ± 0.006 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.157 ± 0.007 | 0.111 ± 0.006 | 0.266 ± 0.016 | 0.530 ± 0.009 | 0.183 ± 0.006 | -0.049 ± 0.008 (p=0.000, n=5) | -0.100 ± 0.033 (p=0.003, n=5) | 0/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.225 ± 0.009 | 0.138 ± 0.006 | 0.625 ± 0.024 | 0.652 ± 0.017 | 0.183 ± 0.006 | +0.019 ± 0.007 (p=0.004, n=5) | +0.023 ± 0.032 (p=0.196, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.382 ± 0.024 | 0.236 ± 0.019 | 1.000 ± 0.000 | 0.832 ± 0.016 | 0.183 ± 0.006 | +0.176 ± 0.021 (p=0.000, n=5) | +0.202 ± 0.038 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/session | *the arm itself* | 0.147 ± 0.035 | 0.081 ± 0.021 | 0.863 ± 0.031 | 0.825 ± 0.037 | 0.067 ± 0.005 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.237 ± 0.005 | 0.171 ± 0.002 | 0.387 ± 0.021 | 0.668 ± 0.011 | 0.067 ± 0.005 | +0.090 ± 0.037 (p=0.006, n=5) | -0.157 ± 0.039 (p=0.001, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.220 ± 0.010 | 0.130 ± 0.007 | 0.715 ± 0.027 | 0.812 ± 0.016 | 0.067 ± 0.005 | +0.073 ± 0.030 (p=0.006, n=5) | -0.013 ± 0.037 (p=0.475, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.322 ± 0.031 | 0.202 ± 0.023 | 0.786 ± 0.029 | 0.841 ± 0.018 | 0.067 ± 0.005 | +0.174 ± 0.024 (p=0.000, n=5) | +0.016 ± 0.040 (p=0.422, n=5) | 5/5 |

