# Per-trace root-cause axis (pooled; B and seed in summary.json)

| row | `m` hit@1 [CI] / hit@3 / exact (set) / random@1 | `l` hit@1 [CI] / hit@3 / exact (set) / random@1 |
|---|---|---|
| `broadcast/control/rule` | 0.399 [0.246, 0.558] / 0.716 / n/a / 0.199 | 0.493 [0.328, 0.666] / 0.688 / n/a / 0.181 |
| `broadcast/rca/baro` | 0.381 [0.233, 0.530] / 0.767 / n/a / 0.199 | 0.436 [0.261, 0.605] / 0.750 / n/a / 0.181 |
| `broadcast/rca/baro-bocpd` | 0.204 [0.097, 0.323] / 0.477 / n/a / 0.199 | — |
| `broadcast/rca/causalrca/latency-50` | 0.164 [0.082, 0.262] / 0.562 / n/a / 0.199 | — |
| `broadcast/rca/causalrca/latency-95` | 0.116 [0.043, 0.202] / 0.506 / n/a / 0.199 | — |
| `broadcast/rca/circa` | 0.139 [0.061, 0.227] / 0.636 / n/a / 0.199 | — |
| `broadcast/rca/circa-topology` | 0.254 [0.146, 0.369] / 0.708 / n/a / 0.199 | — |
| `broadcast/rca/pc-pagerank` | 0.160 [0.067, 0.269] / 0.627 / n/a / 0.199 | — |
| `broadcast/rca/rcd` | 0.477 [0.292, 0.661] / 0.732 / n/a / 0.199 | 0.398 [0.212, 0.587] / 0.718 / n/a / 0.181 |
| `broadcast/rca/tracerca` | 0.107 [0.027, 0.203] / 0.446 / n/a / 0.199 | — |
| `control/trace-callees/hop0` | 0.351 [0.312, 0.387] / 0.723 / n/a / 0.199 | 0.314 [0.266, 0.362] / 0.681 / n/a / 0.181 |
| `control/trace-callees/hop1` | 0.174 [0.120, 0.229] / 0.384 / n/a / 0.199 | 0.151 [0.108, 0.196] / 0.366 / n/a / 0.181 |
| `control/trace-random` | 0.199 [0.177, 0.225] / 0.569 / n/a / 0.199 | 0.181 [0.168, 0.194] / 0.540 / n/a / 0.181 |
| `floor/trace-topology/hop0` | 0.695 [0.629, 0.755] / 0.956 / 0.133 (4.462) / 0.199 | 0.693 [0.621, 0.763] / 0.954 / 0.106 (4.862) / 0.181 |
| `floor/trace-topology/hop1` | 0.191 [0.092, 0.301] / 0.390 / 0.052 (3.342) / 0.199 | 0.169 [0.068, 0.290] / 0.395 / 0.025 (3.669) / 0.181 |
| `rca/trace-ig-keep/hop0` | 0.659 [0.594, 0.709] / 0.904 / 0.149 (3.985) / 0.199 | 0.603 [0.538, 0.663] / 0.824 / 0.054 (4.865) / 0.181 |
| `rca/trace-ig-keep/hop1` | 0.258 [0.188, 0.327] / 0.494 / 0.093 (3.046) / 0.199 | 0.149 [0.088, 0.217] / 0.444 / 0.057 (4.082) / 0.181 |
| `rca/trace-ig-keep/source` | 0.619 [0.559, 0.674] / 0.850 / 0.483 (1.562) / 0.199 | 0.612 [0.552, 0.671] / 0.711 / 0.526 (1.268) / 0.181 |
| `rca/trace-cmi-keep/hop0` | 0.627 [0.555, 0.687] / 0.856 / 0.110 (4.267) / 0.199 | 0.559 [0.484, 0.626] / 0.774 / 0.049 (4.824) / 0.181 |
| `rca/trace-cmi-keep/hop1` | 0.174 [0.115, 0.235] / 0.430 / 0.054 (3.338) / 0.199 | 0.119 [0.069, 0.174] / 0.368 / 0.045 (3.889) / 0.181 |
| `rca/trace-cmi-keep/source` | 0.631 [0.570, 0.684] / 0.806 / 0.520 (1.429) / 0.199 | 0.594 [0.527, 0.652] / 0.684 / 0.486 (1.321) / 0.181 |
| `control/trace-earliest/earliest` | 0.428 [0.310, 0.557] / 0.881 / 0.008 (5.924) / 0.199 | 0.404 [0.282, 0.544] / 0.815 / 0.000 (6.279) / 0.181 |

## Hypotheses

- **H-trace-perseq-random**: fail
- **H-trace-perseq-topology**: fail
- **H-trace-perseq-broadcast**: fail
- **H-trace-perseq-self**: fail
- **H-trace-earliest**: pass
- **H-l-top3**: fail
- **H-l-over-free**: pass
- **H-l-hold**: pass
- **H-l-vs-rcd**: pass
- **H-l-ig-ge-cmi**: pass
- **H-l-no-blind-kind**: pass
- **H-l-floor-gap**: pass
