# Context family — l / latent (schema seq2causebench/context-table@1)

Five-seed means of test directed F1 at the floor 0.05; pairs are per-seed differences with the paired t-test; τ drift = log10(τ*_test / τ_frozen) where a test-side scoresweep was given, else the F1 drift test − validation at the frozen τ. Verdicts follow the registered fail clauses.

## request

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/request` | 0.1354 | 0.0057 | 0.3103 | 0.0370 | — |
| `trace/core-cmi-overwrite/fixed-kl/frozen/request` | 0.1235 | 0.0073 | 0.2802 | 0.0319 | — |
| `trace/core-ig-keep/fixed-kl/frozen/request` | 0.1354 | 0.0057 | 0.3103 | 0.0370 | — |
| `trace/core-cmi-keep/fixed-kl/frozen/request` | 0.1235 | 0.0073 | 0.2802 | 0.0319 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/request` | 0.1247 | 0.0075 | 0.2775 | 0.0312 | — |
| `trace/cli-atomic-cmi-keep/fixed/frozen/request` | 0.1247 | 0.0075 | 0.2775 | 0.0312 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/request` | 0.0059 | 0.0035 | 0.0475 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/request` | 0.0058 | 0.0035 | 0.0473 | — | — |
| `trace/core/fixed-kl/frozen/request` | 0.1355 | 0.0059 | 0.3123 | 0.0369 | — |
| `trace/cli-atomic/fixed-kl/frozen/request` | 0.1327 | 0.0055 | 0.2863 | 0.0336 | — |
| `trace/cli-atomic/fixed-kl/shipped/request` | 0.0064 | 0.0040 | 0.0520 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0119 | 0.002 | -0.0301 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0119 | 0.002 | -0.0301 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0001 | 0.783 | -0.0020 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0080 | 0.004 | -0.0088 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | -0.0005 | 0.490 | -0.0045 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | -0.0000 | 0.929 | 0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0000 | — | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | not applicable (session grain only) |
| H-atomic-cmi-f1 | pass |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |

## session

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/session` | 0.1579 | 0.0109 | 0.3248 | 0.0391 | — |
| `trace/core-cmi-overwrite/fixed-kl/frozen/session` | 0.1260 | 0.0172 | 0.2433 | 0.0259 | — |
| `trace/core-ig-keep/fixed-kl/frozen/session` | 0.1598 | 0.0120 | 0.3332 | 0.0398 | — |
| `trace/core-cmi-keep/fixed-kl/frozen/session` | 0.1255 | 0.0180 | 0.2387 | 0.0249 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/session` | 0.1207 | 0.0108 | 0.2169 | 0.0210 | — |
| `trace/cli-atomic-cmi-keep/fixed/frozen/session` | 0.1207 | 0.0108 | 0.2169 | 0.0209 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/session` | 0.0066 | 0.0025 | 0.0306 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/session` | 0.0065 | 0.0025 | 0.0302 | — | — |
| `trace/core/fixed-kl/frozen/session` | 0.1610 | 0.0111 | 0.3379 | 0.0406 | — |
| `trace/cli-atomic/fixed-kl/frozen/session` | 0.1477 | 0.0142 | 0.3065 | 0.0345 | — |
| `trace/cli-atomic/fixed-kl/shipped/session` | 0.0064 | 0.0019 | 0.0308 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0319 | 0.001 | -0.0815 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0343 | 0.001 | -0.0945 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0019 | 0.215 | 0.0084 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0005 | 0.266 | -0.0046 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0031 | 0.070 | -0.0131 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0270 | 0.000 | -0.0896 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | 0.0002 | 0.551 | -0.0002 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | -0.0000 | 0.783 | 0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0015 | 0.867 | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0071 | 0.000 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | pass |
| H-atomic-cmi-f1 | fail |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |
