# Context family — xs / latent (schema seq2causebench/context-table@1)

Five-seed means of test directed F1 at the floor 0.05; pairs are per-seed differences with the paired t-test; τ drift = log10(τ*_test / τ_frozen) where a test-side scoresweep was given, else the F1 drift test − validation at the frozen τ. Verdicts follow the registered fail clauses.

## request

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/request` | 0.3167 | 0.0455 | 0.4116 | -0.0013 | 0.600 |
| `trace/core-cmi-overwrite/fixed-kl/frozen/request` | 0.3127 | 0.0503 | 0.4016 | -0.0058 | 0.495 |
| `trace/core-ig-keep/fixed-kl/frozen/request` | 0.3112 | 0.0427 | 0.3954 | -0.0057 | 0.809 |
| `trace/core-cmi-keep/fixed-kl/frozen/request` | 0.3136 | 0.0486 | 0.3928 | 0.0004 | 0.591 |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/request` | 0.3039 | 0.0374 | 0.3635 | 0.0017 | 0.695 |
| `trace/cli-atomic-cmi-keep/fixed/frozen/request` | 0.3041 | 0.0376 | 0.3643 | 0.0014 | 0.705 |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/request` | 0.0610 | 0.0130 | 0.1135 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/request` | 0.0603 | 0.0114 | 0.1122 | — | — |
| `trace/core/fixed-kl/frozen/request` | 0.3213 | 0.0431 | 0.4160 | 0.0018 | — |
| `trace/cli-atomic/fixed-kl/frozen/request` | 0.3070 | 0.0429 | 0.4063 | -0.0093 | — |
| `trace/cli-atomic/fixed-kl/shipped/request` | 0.0481 | 0.0363 | 0.0991 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0040 | 0.512 | -0.0099 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | 0.0024 | 0.725 | -0.0026 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0056 | 0.212 | -0.0161 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0008 | 0.403 | -0.0088 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0046 | 0.086 | -0.0045 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0031 | 0.755 | -0.0429 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | 0.0129 | 0.496 | 0.0143 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | 0.0002 | 0.485 | 0.0009 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0226 | 0.121 | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0115 | 0.172 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | pass |
| H-ctx-keep-perseq | not applicable (session grain only) |
| H-atomic-cmi-f1 | pass |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |

## session

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/session` | 0.3535 | 0.0284 | 0.4374 | -0.0122 | 0.382 |
| `trace/core-cmi-overwrite/fixed-kl/frozen/session` | 0.3380 | 0.0283 | 0.4174 | -0.0132 | 0.295 |
| `trace/core-ig-keep/fixed-kl/frozen/session` | 0.3614 | 0.0241 | 0.4436 | -0.0039 | 0.105 |
| `trace/core-cmi-keep/fixed-kl/frozen/session` | 0.3447 | 0.0292 | 0.4242 | -0.0006 | 0.191 |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/session` | 0.3149 | 0.0317 | 0.3737 | -0.0029 | 0.095 |
| `trace/cli-atomic-cmi-keep/fixed/frozen/session` | 0.3146 | 0.0318 | 0.3732 | -0.0027 | 0.200 |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/session` | 0.0787 | 0.0294 | 0.1176 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/session` | 0.0778 | 0.0286 | 0.1164 | — | — |
| `trace/core/fixed-kl/frozen/session` | 0.3566 | 0.0210 | 0.4428 | -0.0140 | — |
| `trace/cli-atomic/fixed-kl/frozen/session` | 0.3320 | 0.0113 | 0.4128 | -0.0126 | — |
| `trace/cli-atomic/fixed-kl/shipped/session` | 0.0733 | 0.0277 | 0.1160 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0155 | 0.006 | -0.0200 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0167 | 0.005 | -0.0194 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0079 | 0.021 | 0.0062 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0066 | 0.217 | 0.0068 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0031 | 0.500 | -0.0054 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0170 | 0.218 | -0.0391 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | 0.0053 | 0.439 | 0.0016 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | -0.0003 | 0.474 | -0.0005 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0031 | 0.819 | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0143 | 0.221 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | pass |
| H-atomic-cmi-f1 | pass |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |
