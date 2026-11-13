# l / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/request | *the arm itself* | 0.235 ± 0.022 | 0.139 ± 0.016 | 0.772 ± 0.051 | 0.656 ± 0.014 | 0.189 ± 0.020 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.165 ± 0.016 | 0.118 ± 0.013 | 0.276 ± 0.023 | 0.534 ± 0.013 | 0.189 ± 0.020 | -0.070 ± 0.010 (p=0.000, n=5) | -0.123 ± 0.011 (p=0.000, n=5) | 0/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.241 ± 0.024 | 0.146 ± 0.017 | 0.697 ± 0.032 | 0.679 ± 0.035 | 0.189 ± 0.020 | +0.006 ± 0.003 (p=0.014, n=5) | +0.022 ± 0.025 (p=0.116, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.395 ± 0.024 | 0.247 ± 0.019 | 1.000 ± 0.000 | 0.837 ± 0.014 | 0.189 ± 0.020 | +0.160 ± 0.017 (p=0.000, n=5) | +0.181 ± 0.026 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/session | *the arm itself* | 0.234 ± 0.038 | 0.139 ± 0.026 | 0.773 ± 0.031 | 0.852 ± 0.019 | 0.068 ± 0.008 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.248 ± 0.018 | 0.181 ± 0.013 | 0.398 ± 0.029 | 0.674 ± 0.014 | 0.068 ± 0.008 | +0.014 ± 0.021 (p=0.202, n=5) | -0.178 ± 0.010 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.211 ± 0.016 | 0.122 ± 0.010 | 0.789 ± 0.022 | 0.844 ± 0.015 | 0.068 ± 0.008 | -0.024 ± 0.029 (p=0.147, n=5) | -0.008 ± 0.013 (p=0.269, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.319 ± 0.013 | 0.201 ± 0.010 | 0.784 ± 0.019 | 0.838 ± 0.012 | 0.068 ± 0.008 | +0.085 ± 0.038 (p=0.008, n=5) | -0.014 ± 0.029 (p=0.357, n=5) | 5/5 |

