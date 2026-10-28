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
| xs | `rca/circa` | 10 | 0.100 | 0.500 | 1.000 | 0.620 [0.480, 0.760] | 0.600 | 0 |
| s | `rca/circa` | 15 | 0.133 | 0.467 | 0.733 | 0.427 [0.267, 0.613] | 0.250 | 0 |
| m | `rca/circa` | 25 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.058 | 0 |
| xs | `rca/rcd` | 10 | 0.630 | 0.880 | 0.880 | 0.818 [0.638, 0.960] | 0.600 | 0 |
| s | `rca/rcd` | 15 | 0.800 | 0.933 | 0.940 | 0.900 [0.758, 1.000] | 0.250 | 0 |
| m | `rca/rcd` | 25 | 0.404 | 0.552 | 0.584 | 0.518 [0.340, 0.687] | 0.058 | 0 |
| xs | `rca/circa-topology` | 10 | 0.100 | 0.500 | 1.000 | 0.500 [0.360, 0.680] | 0.600 | 0 |
| s | `rca/circa-topology` | 15 | 0.133 | 0.267 | 0.333 | 0.253 [0.067, 0.453] | 0.250 | 0 |
| m | `rca/circa-topology` | 25 | 0.000 | 0.000 | 0.080 | 0.024 [0.000, 0.056] | 0.058 | 0 |
| xs | `rca/tracerca` | 10 | 0.200 | 0.700 | 0.900 | 0.640 [0.440, 0.820] | 0.600 | 0 |
| s | `rca/tracerca` | 15 | 0.400 | 0.600 | 0.600 | 0.520 [0.280, 0.747] | 0.250 | 0 |
| m | `rca/tracerca` | 25 | 0.000 | 0.080 | 0.200 | 0.088 [0.024, 0.168] | 0.058 | 0 |
| xs | `rca/pc-pagerank` | 10 | 0.200 | 0.800 | 1.000 | 0.660 [0.480, 0.820] | 0.600 | 0 |
| s | `rca/pc-pagerank` | 15 | 0.067 | 0.333 | 0.467 | 0.293 [0.120, 0.493] | 0.250 | 0 |
| m | `rca/pc-pagerank` | 25 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.058 | 0 |
| xs | `rca/trace` | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| s | `rca/trace` | 15 | 0.133 | 0.267 | 0.600 | 0.333 [0.160, 0.520] | 0.250 | 0 |
| m | `rca/trace` | 25 | 0.000 | 0.040 | 0.200 | 0.080 [0.016, 0.160] | 0.058 | 0 |
| xs | `rca/trace-core` | 10 | 0.100 | 0.600 | 1.000 | 0.560 [0.380, 0.740] | 0.600 | 0 |
| s | `rca/trace-core` | 15 | 0.133 | 0.267 | 0.467 | 0.307 [0.120, 0.507] | 0.250 | 0 |
| m | `rca/trace-core` | 25 | 0.000 | 0.200 | 0.240 | 0.144 [0.048, 0.256] | 0.058 | 0 |

| rung | arm | graph edges per case, mean [min, max] |
|---|---|---|
| xs | `rca/circa` | 16.4 [13, 19] |
| s | `rca/circa` | 103.7 [96, 112] |
| m | `rca/circa` | 362.1 [336, 385] |
| xs | `rca/circa-topology` | 43.8 [39, 51] |
| s | `rca/circa-topology` | 248.4 [228, 258] |
| m | `rca/circa-topology` | 3061.2 [2718, 3228] |
| xs | `rca/pc-pagerank` | 16.4 [13, 19] |
| s | `rca/pc-pagerank` | 103.7 [96, 112] |
| m | `rca/pc-pagerank` | 362.1 [336, 385] |
| xs | `rca/trace` | 2101.0 [814, 4272] |
| s | `rca/trace` | 2854.3 [2048, 3642] |
| m | `rca/trace` | 4274.6 [3342, 5845] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| s | `rca/trace-core` | 2134.1 [1630, 2856] |
| m | `rca/trace-core` | 2829.7 [1893, 5044] |

| rung | arm | alerting services per case, mean [min, max] |
|---|---|---|
| xs | `control/rule` | 2.50 [0, 5] |
| s | `control/rule` | 8.80 [3, 12] |
| m | `control/rule` | 29.04 [3, 51] |

| rung | arm | window traces per case, mean [min, max] | probed, mean [min, max] | services reached, mean |
|---|---|---|---|---|
| xs | `rca/trace` | 1589 [1046, 2144] | 435 [333, 500] | 5.0 |
| s | `rca/trace` | 26019 [17429, 43836] | 498 [486, 500] | 12.0 |
| m | `rca/trace` | 20935 [10722, 32016] | 500 [500, 500] | 51.2 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-core` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-core` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |

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
| H-rca-circa-random | fail |
| H-rca-circa-rule | fail |
| H-rca-circa-topology-random | fail |
| H-rca-circa-topology-rule | fail |
| H-rca-rcd-random | pass |
| H-rca-rcd-rule | pass |
| H-rca-tracerca-random | fail |
| H-rca-tracerca-rule | fail |
| H-rca-pc-pagerank-random | fail |
| H-rca-pc-pagerank-rule | fail |
| H-rca-trace-random | fail |
| H-rca-trace-rule | fail |
| H-rca-trace-core-random | fail |
| H-rca-trace-core-rule | fail |
| H-circa-topology-gain | fail |
| H-graph-topology | pass |
