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

| rung | arm | alerting services per case, mean [min, max] |
|---|---|---|
| xs | `control/rule` | 2.50 [0, 5] |
| s | `control/rule` | 8.80 [3, 12] |
| m | `control/rule` | 29.04 [3, 51] |

| hypothesis | verdict |
|---|---|
| H-rca-random | fail |
| H-rca-bocpd | fail |
| H-rca-scale | pass |
| H-rca-rule | fail |
| H-graph-topology | not run |
