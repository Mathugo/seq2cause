# Context family — s / latent (schema seq2causebench/context-table@1)

Five-seed means of test directed F1 at the floor 0.05; pairs are per-seed differences with the paired t-test; τ drift = log10(τ*_test / τ_frozen) where a test-side scoresweep was given, else the F1 drift test − validation at the frozen τ. Verdicts follow the registered fail clauses.

## request

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/request` | 0.2900 | 0.0123 | 0.3855 | 0.0306 | 0.314 |
| `trace/core-cmi-overwrite/fixed-kl/frozen/request` | 0.2632 | 0.0115 | 0.3399 | 0.0178 | 0.505 |
| `trace/core-ig-keep/fixed-kl/frozen/request` | 0.2900 | 0.0123 | 0.3855 | 0.0306 | 0.314 |
| `trace/core-cmi-keep/fixed-kl/frozen/request` | 0.2636 | 0.0111 | 0.3362 | 0.0187 | 0.400 |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/request` | 0.2628 | 0.0128 | 0.3365 | 0.0143 | 0.400 |
| `trace/cli-atomic-cmi-keep/fixed/frozen/request` | 0.2628 | 0.0126 | 0.3365 | 0.0143 | 0.400 |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/request` | 0.0795 | 0.0134 | 0.1473 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/request` | 0.0794 | 0.0133 | 0.1471 | — | — |
| `trace/core/fixed-kl/frozen/request` | 0.2889 | 0.0132 | 0.3823 | 0.0272 | — |
| `trace/cli-atomic/fixed-kl/frozen/request` | 0.2867 | 0.0157 | 0.3790 | 0.0221 | — |
| `trace/cli-atomic/fixed-kl/shipped/request` | 0.0929 | 0.0154 | 0.1769 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0268 | 0.000 | -0.0456 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0264 | 0.000 | -0.0493 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0004 | 0.211 | -0.0037 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | 0.0011 | 0.543 | 0.0031 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0239 | 0.005 | -0.0425 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | -0.0134 | 0.000 | -0.0296 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | -0.0000 | 0.880 | -0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0099 | 0.229 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | not applicable (session grain only) |
| H-atomic-cmi-f1 | fail |
| H-atomic-cmi-shipped | fail |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |

## session

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/session` | 0.3340 | 0.0174 | 0.3793 | 0.0313 | 0.382 |
| `trace/core-cmi-overwrite/fixed-kl/frozen/session` | 0.3030 | 0.0166 | 0.3475 | 0.0346 | 0.200 |
| `trace/core-ig-keep/fixed-kl/frozen/session` | 0.3405 | 0.0115 | 0.3903 | 0.0372 | 0.191 |
| `trace/core-cmi-keep/fixed-kl/frozen/session` | 0.3050 | 0.0127 | 0.3488 | 0.0365 | 0.286 |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/session` | 0.2888 | 0.0154 | 0.3236 | 0.0125 | 0.305 |
| `trace/cli-atomic-cmi-keep/fixed/frozen/session` | 0.2887 | 0.0153 | 0.3235 | 0.0124 | 0.305 |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/session` | 0.1275 | 0.0173 | 0.1584 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/session` | 0.1256 | 0.0194 | 0.1562 | — | — |
| `trace/core/fixed-kl/frozen/session` | 0.3423 | 0.0184 | 0.3905 | 0.0334 | — |
| `trace/cli-atomic/fixed-kl/frozen/session` | 0.3236 | 0.0155 | 0.3642 | 0.0150 | — |
| `trace/cli-atomic/fixed-kl/shipped/session` | 0.1492 | 0.0295 | 0.1880 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0311 | 0.006 | -0.0317 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0355 | 0.002 | -0.0415 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0065 | 0.373 | 0.0110 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0021 | 0.719 | 0.0012 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0083 | 0.048 | -0.0112 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0348 | 0.011 | -0.0407 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | -0.0217 | 0.191 | -0.0296 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | -0.0001 | 0.094 | -0.0001 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0044 | 0.080 | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0035 | 0.213 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | pass |
| H-atomic-cmi-f1 | fail |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |
