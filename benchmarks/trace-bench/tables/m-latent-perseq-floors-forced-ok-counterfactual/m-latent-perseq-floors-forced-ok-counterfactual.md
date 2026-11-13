# m / latent — the floors on the per-sequence axis (its own tables; never merged with type-level numbers)

Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the scoreable fraction are therefore the reference row's. Paired difference = floor − reference, five seeds, paired t-test.

## request grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/request | *the arm itself* | 0.252 ± 0.014 | 0.164 ± 0.012 | 0.551 ± 0.047 | 0.668 ± 0.017 | 0.177 ± 0.009 | — | — | — |
| trace/core/shipped/frozen/request | `floor/bigram` adjacent | 0.161 ± 0.007 | 0.112 ± 0.006 | 0.285 ± 0.015 | 0.542 ± 0.007 | 0.177 ± 0.009 | -0.091 ± 0.008 (p=0.000, n=5) | -0.126 ± 0.012 (p=0.000, n=5) | 0/5 |
| trace/core/shipped/frozen/request | `floor/bigram` broadcast | 0.211 ± 0.014 | 0.125 ± 0.008 | 0.683 ± 0.059 | 0.693 ± 0.040 | 0.177 ± 0.009 | -0.041 ± 0.003 (p=0.000, n=5) | +0.024 ± 0.027 (p=0.111, n=5) | 5/5 |
| trace/core/shipped/frozen/request | `floor/topology` broadcast | 0.371 ± 0.023 | 0.228 ± 0.018 | 1.000 ± 0.000 | 0.827 ± 0.018 | 0.177 ± 0.009 | +0.119 ± 0.025 (p=0.000, n=5) | +0.159 ± 0.029 (p=0.000, n=5) | 5/5 |

## session grain

| reference read | row | pooled.directed.f1 | pooled.directed.precision | pooled.directed.recall | pooled.auroc | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |
|---|---|---|---|---|---|---|---|---|---|
| trace/core/shipped/frozen/session | *the arm itself* | 0.297 ± 0.014 | 0.199 ± 0.008 | 0.586 ± 0.039 | 0.869 ± 0.008 | 0.071 ± 0.003 | — | — | — |
| trace/core/shipped/frozen/session | `floor/bigram` adjacent | 0.252 ± 0.016 | 0.186 ± 0.013 | 0.393 ± 0.032 | 0.673 ± 0.017 | 0.071 ± 0.003 | -0.045 ± 0.015 (p=0.002, n=5) | -0.195 ± 0.014 (p=0.000, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/bigram` broadcast | 0.211 ± 0.031 | 0.122 ± 0.021 | 0.795 ± 0.060 | 0.851 ± 0.036 | 0.071 ± 0.003 | -0.087 ± 0.034 (p=0.005, n=5) | -0.017 ± 0.032 (p=0.297, n=5) | 5/5 |
| trace/core/shipped/frozen/session | `floor/topology` broadcast | 0.326 ± 0.027 | 0.206 ± 0.019 | 0.781 ± 0.034 | 0.835 ± 0.022 | 0.071 ± 0.003 | +0.029 ± 0.039 (p=0.173, n=5) | -0.034 ± 0.026 (p=0.045, n=5) | 5/5 |

