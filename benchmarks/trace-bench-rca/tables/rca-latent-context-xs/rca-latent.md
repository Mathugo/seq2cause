| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 | misses |
|---|---|---|---|---|---|---|---|---|
| xs | `rca/baro` | 10 | 0.300 | 0.800 | 1.000 | 0.720 [0.540, 0.880] | 0.600 | 0 |
| xs | `rca/baro-bocpd` | 10 | 0.100 | 0.100 | 0.200 | 0.140 [0.000, 0.380] | 0.600 | 7 |
| xs | `control/rule` | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.820] | 0.600 | 0 |
| xs | `rca/causalrca` (latency-50) | 10 | 0.220 | 0.620 | 0.990 | 0.596 [0.470, 0.734] | 0.600 | 0 |
| xs | `rca/causalrca` (latency-95) | 10 | 0.280 | 0.630 | 0.990 | 0.622 [0.474, 0.780] | 0.600 | 0 |
| xs | `rca/circa` | 10 | 0.100 | 0.500 | 1.000 | 0.620 [0.480, 0.760] | 0.600 | 0 |
| xs | `rca/rcd` | 10 | 0.630 | 0.880 | 0.880 | 0.818 [0.638, 0.960] | 0.600 | 0 |
| xs | `rca/circa-topology` | 10 | 0.100 | 0.500 | 1.000 | 0.500 [0.360, 0.680] | 0.600 | 0 |
| xs | `rca/tracerca` | 10 | 0.200 | 0.700 | 0.900 | 0.640 [0.440, 0.820] | 0.600 | 0 |
| xs | `rca/pc-pagerank` | 10 | 0.200 | 0.800 | 1.000 | 0.660 [0.480, 0.820] | 0.600 | 0 |
| xs | `rca/trace` | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| xs | `rca/trace` (hop0) | 10 | 0.000 | 0.500 | 1.000 | 0.460 [0.320, 0.620] | 0.600 | 0 |
| xs | `rca/trace` (source) | 10 | 0.200 | 0.600 | 1.000 | 0.580 [0.380, 0.780] | 0.600 | 0 |
| xs | `rca/trace-core` | 10 | 0.100 | 0.600 | 1.000 | 0.560 [0.380, 0.740] | 0.600 | 0 |
| xs | `rca/trace-core` (hop0) | 10 | 0.100 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| xs | `rca/trace-core` (source) | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.840] | 0.600 | 0 |
| xs | `rca/trace-ig-overwrite` (hop1) | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| xs | `rca/trace-ig-overwrite` (hop0) | 10 | 0.000 | 0.600 | 1.000 | 0.500 [0.340, 0.660] | 0.600 | 0 |
| xs | `rca/trace-ig-overwrite` (source) | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.840] | 0.600 | 0 |
| xs | `rca/trace-ig-overwrite` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| xs | `rca/trace-ig-overwrite` (hop0-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.840 [0.660, 0.980] | 0.600 | 0 |
| xs | `rca/trace-ig-overwrite` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop1) | 10 | 0.000 | 0.600 | 1.000 | 0.520 [0.360, 0.680] | 0.600 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop0) | 10 | 0.000 | 0.600 | 1.000 | 0.500 [0.340, 0.660] | 0.600 | 0 |
| xs | `rca/trace-cmi-overwrite` (source) | 10 | 0.300 | 0.600 | 1.000 | 0.660 [0.460, 0.840] | 0.600 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop0-subpre) | 10 | 0.600 | 0.800 | 1.000 | 0.820 [0.620, 0.980] | 0.600 | 0 |
| xs | `rca/trace-cmi-overwrite` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| xs | `rca/trace-ig-keep` (hop1) | 10 | 0.200 | 0.600 | 1.000 | 0.600 [0.420, 0.800] | 0.600 | 0 |
| xs | `rca/trace-ig-keep` (hop0) | 10 | 0.100 | 0.600 | 1.000 | 0.560 [0.380, 0.740] | 0.600 | 0 |
| xs | `rca/trace-ig-keep` (source) | 10 | 0.400 | 0.600 | 1.000 | 0.680 [0.480, 0.880] | 0.600 | 0 |
| xs | `rca/trace-ig-keep` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| xs | `rca/trace-ig-keep` (hop0-subpre) | 10 | 0.600 | 0.800 | 1.000 | 0.820 [0.620, 0.980] | 0.600 | 0 |
| xs | `rca/trace-ig-keep` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| xs | `rca/trace-cmi-keep` (hop1) | 10 | 0.100 | 0.600 | 1.000 | 0.580 [0.400, 0.760] | 0.600 | 0 |
| xs | `rca/trace-cmi-keep` (hop0) | 10 | 0.100 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| xs | `rca/trace-cmi-keep` (source) | 10 | 0.400 | 0.600 | 1.000 | 0.700 [0.500, 0.880] | 0.600 | 0 |
| xs | `rca/trace-cmi-keep` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| xs | `rca/trace-cmi-keep` (hop0-subpre) | 10 | 0.600 | 0.800 | 1.000 | 0.820 [0.620, 0.980] | 0.600 | 0 |
| xs | `rca/trace-cmi-keep` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop1) | 10 | 0.000 | 0.700 | 1.000 | 0.600 [0.440, 0.740] | 0.600 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop0) | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.380, 0.700] | 0.600 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (source) | 10 | 0.400 | 0.600 | 1.000 | 0.700 [0.520, 0.880] | 0.600 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.780 [0.560, 0.940] | 0.600 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop0-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.780 [0.560, 0.940] | 0.600 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| xs | `control/trace-earliest` | 10 | 0.000 | 0.500 | 1.000 | 0.440 [0.340, 0.540] | 0.600 | 0 |

| rung | arm | graph edges per case, mean [min, max] |
|---|---|---|
| xs | `rca/circa` | 16.4 [13, 19] |
| xs | `rca/circa-topology` | 43.8 [39, 51] |
| xs | `rca/pc-pagerank` | 16.4 [13, 19] |
| xs | `rca/trace` | 2101.0 [814, 4272] |
| xs | `rca/trace` | 2101.0 [814, 4272] |
| xs | `rca/trace` | 2101.0 [814, 4272] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |

| rung | arm | alerting services per case, mean [min, max] |
|---|---|---|
| xs | `control/rule` | 2.50 [0, 5] |

| rung | arm | window traces per case, mean [min, max] | probed, mean [min, max] | services reached, mean |
|---|---|---|---|---|
| xs | `rca/trace` | 1589 [1046, 2144] | 435 [333, 500] | 5.0 |
| xs | `rca/trace` | 1589 [1046, 2144] | 435 [333, 500] | 5.0 |
| xs | `rca/trace` | 1589 [1046, 2144] | 435 [333, 500] | 4.9 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |

Graph axis — four test seeds (1–4); seed 0 is the graph axis's validation split. Mean over the test cases.

| rung | series | τ | cases | precision | recall | F1 | AUROC | `floor/topology` F1 | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| xs | latency-50 | 0.1 | 8 | 0.249 | 0.304 | 0.266 | 0.516 | 1.000 | 1.000 |
| xs | latency-95 | 0.15 | 8 | 0.132 | 0.148 | 0.136 | 0.433 | 1.000 | 1.000 |

| hypothesis | verdict |
|---|---|
| H-rca-random | fail |
| H-rca-bocpd | fail |
| H-rca-rule | pass |
| H-rca-circa-random | fail |
| H-rca-circa-rule | fail |
| H-rca-circa-topology-random | fail |
| H-rca-circa-topology-rule | fail |
| H-rca-rcd-random | pass |
| H-rca-rcd-rule | pass |
| H-rca-tracerca-random | fail |
| H-rca-tracerca-rule | pass |
| H-rca-pc-pagerank-random | fail |
| H-rca-pc-pagerank-rule | pass |
| H-rca-trace-random | fail |
| H-rca-trace-rule | fail |
| H-rca-trace-core-random | fail |
| H-rca-trace-core-rule | fail |
| H-rca-trace-ig-overwrite-random | fail |
| H-rca-trace-ig-overwrite-rule | fail |
| H-rca-trace-cmi-overwrite-random | fail |
| H-rca-trace-cmi-overwrite-rule | fail |
| H-rca-trace-ig-keep-random | fail |
| H-rca-trace-ig-keep-rule | fail |
| H-rca-trace-cmi-keep-random | fail |
| H-rca-trace-cmi-keep-rule | fail |
| H-rca-trace-atomic-cmi-keep-random | fail |
| H-rca-trace-atomic-cmi-keep-rule | fail |
| H-circa-topology-gain | fail |
| H-graph-topology | pass |
| H-rca-hop0 | fail |
| H-rca-ctx-rule | fail |
| H-rca-ctx-rcd | fail |
| H-rca-source | pass |
| H-rca-presub | pass |
| H-rca-atomic-cmi | pass |
