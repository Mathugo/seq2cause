| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 | misses |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/baro` | 10 | 0.300 | 0.800 | 1.000 | 0.720 [0.540, 0.880] | 0.600 | 0 |
| s | `rca/baro` | 15 | 0.267 | 0.467 | 0.667 | 0.467 [0.253, 0.680] | 0.250 | 0 |
| m | `rca/baro` | 25 | 0.120 | 0.160 | 0.240 | 0.176 [0.048, 0.320] | 0.058 | 0 |
| xs | `rca/baro-bocpd` | 10 | 0.100 | 0.100 | 0.200 | 0.140 [0.000, 0.380] | 0.600 | 7 |
| s | `rca/baro-bocpd` | 15 | 0.133 | 0.267 | 0.600 | 0.333 [0.173, 0.533] | 0.250 | 0 |
| m | `rca/baro-bocpd` | 25 | 0.040 | 0.040 | 0.040 | 0.040 [0.000, 0.120] | 0.058 | 8 |
| xs | `control/rule` | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.820] | 0.600 | 0 |
| s | `control/rule` | 15 | 0.467 | 0.667 | 0.800 | 0.680 [0.480, 0.867] | 0.250 | 0 |
| m | `control/rule` | 25 | 0.160 | 0.280 | 0.320 | 0.272 [0.120, 0.440] | 0.058 | 0 |
| xs | `rca/causalrca` (latency-50) | 10 | 0.220 | 0.620 | 0.990 | 0.596 [0.470, 0.734] | 0.600 | 0 |
| s | `rca/causalrca` (latency-50) | 15 | 0.080 | 0.227 | 0.487 | 0.248 [0.113, 0.400] | 0.250 | 0 |
| m | `rca/causalrca` (latency-50) | 25 | 0.016 | 0.036 | 0.068 | 0.040 [0.007, 0.084] | 0.058 | 0 |
| xs | `rca/causalrca` (latency-95) | 10 | 0.280 | 0.630 | 0.990 | 0.622 [0.474, 0.780] | 0.600 | 0 |
| s | `rca/causalrca` (latency-95) | 15 | 0.107 | 0.193 | 0.333 | 0.204 [0.083, 0.356] | 0.250 | 0 |
| m | `rca/causalrca` (latency-95) | 25 | 0.012 | 0.036 | 0.060 | 0.034 [0.003, 0.080] | 0.058 | 0 |

| rung | arm | alerting services per case, mean [min, max] |
|---|---|---|
| xs | `control/rule` | 2.50 [0, 5] |
| s | `control/rule` | 8.80 [3, 12] |
| m | `control/rule` | 29.04 [3, 51] |

Graph axis — four test seeds (1–4); seed 0 is the graph axis's validation split. Mean over the test cases.

| rung | series | τ | cases | precision | recall | F1 | AUROC | `floor/topology` F1 | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| xs | latency-50 | 0.1 | 8 | 0.249 | 0.304 | 0.266 | 0.516 | 1.000 | 1.000 |
| s | latency-50 | 0.03 | 12 | 0.258 | 0.153 | 0.181 | 0.500 | 1.000 | 1.000 |
| m | latency-50 | 0.03 | 20 | 0.185 | 0.431 | 0.257 | 0.507 | 0.997 | 0.999 |
| xs | latency-95 | 0.15 | 8 | 0.132 | 0.148 | 0.136 | 0.433 | 1.000 | 1.000 |
| s | latency-95 | 0.03 | 12 | 0.319 | 0.198 | 0.233 | 0.517 | 1.000 | 1.000 |
| m | latency-95 | 0.03 | 20 | 0.163 | 0.380 | 0.226 | 0.479 | 0.997 | 0.999 |

| hypothesis | verdict |
|---|---|
| H-rca-random | fail |
| H-rca-bocpd | fail |
| H-rca-scale | pass |
| H-rca-rule | fail |
| H-graph-topology | pass |
