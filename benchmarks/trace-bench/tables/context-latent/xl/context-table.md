# Context family — xl / latent (schema seq2causebench/context-table@1)

Five-seed means of test directed F1 at the floor 0.05; pairs are per-seed differences with the paired t-test; τ drift = log10(τ*_test / τ_frozen) where a test-side scoresweep was given, else the F1 drift test − validation at the frozen τ. Verdicts follow the registered fail clauses.

## request

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/request` | 0.0883 | 0.0021 | 0.2912 | 0.0314 | — |
| `trace/core-cmi-overwrite/fixed-kl/frozen/request` | 0.0882 | 0.0020 | 0.2900 | 0.0313 | — |
| `trace/core-ig-keep/fixed-kl/frozen/request` | 0.0883 | 0.0021 | 0.2912 | 0.0314 | — |
| `trace/core-cmi-keep/fixed-kl/frozen/request` | 0.0882 | 0.0020 | 0.2900 | 0.0313 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/request` | 0.0889 | 0.0018 | 0.2614 | 0.0294 | — |
| `trace/cli-atomic-cmi-keep/fixed/frozen/request` | 0.0889 | 0.0018 | 0.2614 | 0.0294 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/request` | 0.0016 | 0.0011 | 0.0241 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/request` | 0.0016 | 0.0011 | 0.0240 | — | — |
| `trace/core/fixed-kl/frozen/request` | 0.0883 | 0.0021 | 0.2918 | 0.0314 | — |
| `trace/cli-atomic/fixed-kl/frozen/request` | 0.0890 | 0.0018 | 0.2632 | 0.0295 | — |
| `trace/cli-atomic/fixed-kl/shipped/request` | 0.0015 | 0.0012 | 0.0234 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0001 | 0.362 | -0.0012 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0001 | 0.362 | -0.0012 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0000 | 0.565 | -0.0006 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0001 | 0.134 | -0.0018 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | 0.0000 | 0.578 | 0.0007 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0000 | — | — |

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
| `trace/core-ig-overwrite/fixed-kl/frozen/session` | 0.0961 | 0.0079 | 0.2225 | 0.0293 | — |
| `trace/core-cmi-overwrite/fixed-kl/frozen/session` | 0.0771 | 0.0101 | 0.1530 | 0.0188 | — |
| `trace/core-ig-keep/fixed-kl/frozen/session` | 0.0969 | 0.0094 | 0.2257 | 0.0292 | — |
| `trace/core-cmi-keep/fixed-kl/frozen/session` | 0.0764 | 0.0117 | 0.1484 | 0.0179 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/session` | 0.0787 | 0.0078 | 0.1599 | 0.0194 | — |
| `trace/cli-atomic-cmi-keep/fixed/frozen/session` | 0.0787 | 0.0078 | 0.1599 | 0.0194 | — |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/session` | 0.0012 | 0.0009 | 0.0120 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/session` | 0.0012 | 0.0009 | 0.0119 | — | — |
| `trace/core/fixed-kl/frozen/session` | 0.0971 | 0.0085 | 0.2343 | 0.0301 | — |
| `trace/cli-atomic/fixed-kl/frozen/session` | 0.0881 | 0.0103 | 0.2003 | 0.0250 | — |
| `trace/cli-atomic/fixed-kl/shipped/session` | 0.0012 | 0.0009 | 0.0124 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0190 | 0.000 | -0.0694 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0205 | 0.000 | -0.0773 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0008 | 0.366 | 0.0032 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0007 | 0.538 | -0.0046 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0010 | 0.183 | -0.0118 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0094 | 0.002 | -0.0405 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | -0.0000 | 0.842 | -0.0005 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | 0.0000 | 0.509 | 0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0001 | 0.995 | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0074 | 0.075 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | pass |
| H-atomic-cmi-f1 | pass |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |
