# Context family — m / latent (schema seq2causebench/context-table@1)

Five-seed means of test directed F1 at the floor 0.05; pairs are per-seed differences with the paired t-test; τ drift = log10(τ*_test / τ_frozen) where a test-side scoresweep was given, else the F1 drift test − validation at the frozen τ. Verdicts follow the registered fail clauses.

## request

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/request` | 0.2043 | 0.0086 | 0.4174 | 0.0464 | 0.391 |
| `trace/core-cmi-overwrite/fixed-kl/frozen/request` | 0.1836 | 0.0085 | 0.3420 | 0.0353 | 0.695 |
| `trace/core-ig-keep/fixed-kl/frozen/request` | 0.2043 | 0.0086 | 0.4174 | 0.0464 | 0.391 |
| `trace/core-cmi-keep/fixed-kl/frozen/request` | 0.1836 | 0.0085 | 0.3420 | 0.0353 | 0.695 |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/request` | 0.1788 | 0.0036 | 0.3094 | 0.0305 | 0.609 |
| `trace/cli-atomic-cmi-keep/fixed/frozen/request` | 0.1788 | 0.0036 | 0.3094 | 0.0305 | 0.609 |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/request` | 0.0284 | 0.0015 | 0.1260 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/request` | 0.0280 | 0.0015 | 0.1245 | — | — |
| `trace/core/fixed-kl/frozen/request` | 0.2044 | 0.0088 | 0.4129 | 0.0453 | — |
| `trace/cli-atomic/fixed-kl/frozen/request` | 0.1957 | 0.0048 | 0.3653 | 0.0384 | — |
| `trace/cli-atomic/fixed-kl/shipped/request` | 0.0305 | 0.0062 | 0.1346 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0207 | 0.001 | -0.0753 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0207 | 0.001 | -0.0753 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0000 | — | 0.0000 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0001 | 0.600 | 0.0045 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0169 | 0.000 | -0.0559 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | -0.0021 | 0.439 | -0.0086 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | -0.0000 | 0.309 | -0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0000 | — | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | 0.0000 | — | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | not applicable (session grain only) |
| H-atomic-cmi-f1 | fail |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |

## session

| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |
|---|---|---|---|---|---|
| `trace/core-ig-overwrite/fixed-kl/frozen/session` | 0.2680 | 0.0099 | 0.4371 | 0.0683 | 0.418 |
| `trace/core-cmi-overwrite/fixed-kl/frozen/session` | 0.2241 | 0.0132 | 0.3529 | 0.0475 | 0.582 |
| `trace/core-ig-keep/fixed-kl/frozen/session` | 0.2681 | 0.0090 | 0.4331 | 0.0667 | 0.523 |
| `trace/core-cmi-keep/fixed-kl/frozen/session` | 0.2233 | 0.0137 | 0.3489 | 0.0457 | 0.582 |
| `trace/cli-atomic-cmi-keep/fixed-kl/frozen/session` | 0.2058 | 0.0064 | 0.3141 | 0.0371 | 0.286 |
| `trace/cli-atomic-cmi-keep/fixed/frozen/session` | 0.2058 | 0.0064 | 0.3141 | 0.0371 | 0.286 |
| `trace/cli-atomic-cmi-keep/fixed-kl/shipped/session` | 0.0419 | 0.0058 | 0.0986 | — | — |
| `trace/cli-atomic-cmi-keep/fixed/shipped/session` | 0.0401 | 0.0056 | 0.0948 | — | — |
| `trace/core/fixed-kl/frozen/session` | 0.2702 | 0.0103 | 0.4424 | 0.0692 | — |
| `trace/cli-atomic/fixed-kl/frozen/session` | 0.2466 | 0.0087 | 0.3815 | 0.0534 | — |
| `trace/cli-atomic/fixed-kl/shipped/session` | 0.0445 | 0.0045 | 0.1151 | — | — |

| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |
|---|---|---|---|
| `H-ctx-cmi-f1:trace/core-cmi-overwrite/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0439 | 0.000 | -0.0841 |
| `H-ctx-cmi-f1:trace/core-cmi-keep/fixed-kl/frozen:trace/core-ig-keep/fixed-kl/frozen` | -0.0448 | 0.000 | -0.0842 |
| `keep-vs-overwrite:trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | 0.0001 | 0.783 | -0.0040 |
| `keep-vs-overwrite:trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0008 | 0.140 | -0.0040 |
| `identity-i:trace/core-ig-overwrite/fixed-kl/frozen:trace/core/fixed-kl/frozen` | -0.0022 | 0.007 | -0.0054 |
| `H-atomic-cmi-f1:trace/cli-atomic-cmi-keep/fixed-kl/frozen:trace/cli-atomic/fixed-kl/frozen` | -0.0408 | 0.000 | -0.0673 |
| `H-atomic-cmi-shipped:trace/cli-atomic-cmi-keep/fixed-kl/shipped:trace/cli-atomic/fixed-kl/shipped` | -0.0026 | 0.292 | -0.0165 |
| `atomic-fixed-vs-fixed-kl:trace/cli-atomic-cmi-keep/fixed/frozen:trace/cli-atomic-cmi-keep/fixed-kl/frozen` | 0.0000 | 0.837 | 0.0000 |
| per-sequence `trace/core-ig-keep/fixed-kl/frozen:trace/core-ig-overwrite/fixed-kl/frozen` | -0.0032 | 0.128 | — |
| per-sequence `trace/core-cmi-keep/fixed-kl/frozen:trace/core-cmi-overwrite/fixed-kl/frozen` | -0.0050 | 0.018 | — |

| hypothesis | verdict |
|---|---|
| H-ctx-cmi-f1 | fail |
| H-ctx-keep-perseq | pass |
| H-atomic-cmi-f1 | fail |
| H-atomic-cmi-shipped | pass |
| H-ctx-cmi-tau | overwrite: pass; keep: pass |
| H-atomic-cmi-tau | pass |
