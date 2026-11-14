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
| xs | `rca/trace` (hop0) | 10 | 0.000 | 0.500 | 1.000 | 0.460 [0.320, 0.620] | 0.600 | 0 |
| s | `rca/trace` (hop0) | 15 | 0.133 | 0.267 | 0.333 | 0.227 [0.053, 0.427] | 0.250 | 0 |
| m | `rca/trace` (hop0) | 25 | 0.200 | 0.200 | 0.240 | 0.208 [0.072, 0.376] | 0.058 | 0 |
| xs | `rca/trace` (source) | 10 | 0.200 | 0.600 | 1.000 | 0.580 [0.380, 0.780] | 0.600 | 0 |
| s | `rca/trace` (source) | 15 | 0.267 | 0.467 | 0.867 | 0.533 [0.347, 0.707] | 0.250 | 0 |
| m | `rca/trace` (source) | 25 | 0.160 | 0.200 | 0.480 | 0.256 [0.120, 0.400] | 0.058 | 0 |
| xs | `rca/trace-core` | 10 | 0.100 | 0.600 | 1.000 | 0.560 [0.380, 0.740] | 0.600 | 0 |
| s | `rca/trace-core` | 15 | 0.133 | 0.267 | 0.467 | 0.307 [0.120, 0.507] | 0.250 | 0 |
| m | `rca/trace-core` | 25 | 0.000 | 0.200 | 0.240 | 0.144 [0.048, 0.256] | 0.058 | 0 |
| xs | `rca/trace-core` (hop0) | 10 | 0.100 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| s | `rca/trace-core` (hop0) | 15 | 0.133 | 0.200 | 0.333 | 0.227 [0.067, 0.427] | 0.250 | 0 |
| m | `rca/trace-core` (hop0) | 25 | 0.200 | 0.240 | 0.280 | 0.232 [0.088, 0.400] | 0.058 | 0 |
| xs | `rca/trace-core` (source) | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.840] | 0.600 | 0 |
| s | `rca/trace-core` (source) | 15 | 0.133 | 0.467 | 0.800 | 0.467 [0.280, 0.640] | 0.250 | 0 |
| m | `rca/trace-core` (source) | 25 | 0.160 | 0.240 | 0.480 | 0.280 [0.136, 0.432] | 0.058 | 0 |
| xs | `rca/trace-ig-overwrite` (hop1) | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| s | `rca/trace-ig-overwrite` (hop1) | 15 | 0.133 | 0.267 | 0.467 | 0.307 [0.120, 0.507] | 0.250 | 0 |
| m | `rca/trace-ig-overwrite` (hop1) | 25 | 0.000 | 0.160 | 0.240 | 0.136 [0.048, 0.240] | 0.058 | 0 |
| xs | `rca/trace-ig-overwrite` (hop0) | 10 | 0.000 | 0.600 | 1.000 | 0.500 [0.340, 0.660] | 0.600 | 0 |
| s | `rca/trace-ig-overwrite` (hop0) | 15 | 0.133 | 0.267 | 0.333 | 0.240 [0.067, 0.440] | 0.250 | 0 |
| m | `rca/trace-ig-overwrite` (hop0) | 25 | 0.200 | 0.200 | 0.280 | 0.224 [0.080, 0.392] | 0.058 | 0 |
| xs | `rca/trace-ig-overwrite` (source) | 10 | 0.300 | 0.600 | 1.000 | 0.640 [0.440, 0.840] | 0.600 | 0 |
| s | `rca/trace-ig-overwrite` (source) | 15 | 0.133 | 0.467 | 0.800 | 0.467 [0.280, 0.640] | 0.250 | 0 |
| m | `rca/trace-ig-overwrite` (source) | 25 | 0.160 | 0.240 | 0.480 | 0.280 [0.136, 0.432] | 0.058 | 0 |
| xs | `rca/trace-ig-overwrite` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| s | `rca/trace-ig-overwrite` (hop1-subpre) | 15 | 0.267 | 0.533 | 0.800 | 0.533 [0.333, 0.707] | 0.250 | 0 |
| m | `rca/trace-ig-overwrite` (hop1-subpre) | 25 | 0.040 | 0.160 | 0.240 | 0.152 [0.048, 0.272] | 0.058 | 0 |
| xs | `rca/trace-ig-overwrite` (hop0-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.840 [0.660, 0.980] | 0.600 | 0 |
| s | `rca/trace-ig-overwrite` (hop0-subpre) | 15 | 0.267 | 0.600 | 0.733 | 0.507 [0.307, 0.693] | 0.250 | 0 |
| m | `rca/trace-ig-overwrite` (hop0-subpre) | 25 | 0.120 | 0.200 | 0.280 | 0.200 [0.080, 0.352] | 0.058 | 0 |
| xs | `rca/trace-ig-overwrite` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| s | `rca/trace-ig-overwrite` (source-subpre) | 15 | 0.400 | 0.867 | 0.933 | 0.760 [0.600, 0.893] | 0.250 | 0 |
| m | `rca/trace-ig-overwrite` (source-subpre) | 25 | 0.240 | 0.440 | 0.560 | 0.424 [0.264, 0.592] | 0.058 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop1) | 10 | 0.000 | 0.600 | 1.000 | 0.520 [0.360, 0.680] | 0.600 | 0 |
| s | `rca/trace-cmi-overwrite` (hop1) | 15 | 0.067 | 0.333 | 0.400 | 0.267 [0.107, 0.453] | 0.250 | 0 |
| m | `rca/trace-cmi-overwrite` (hop1) | 25 | 0.000 | 0.200 | 0.240 | 0.144 [0.048, 0.256] | 0.058 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop0) | 10 | 0.000 | 0.600 | 1.000 | 0.500 [0.340, 0.660] | 0.600 | 0 |
| s | `rca/trace-cmi-overwrite` (hop0) | 15 | 0.133 | 0.200 | 0.400 | 0.240 [0.080, 0.427] | 0.250 | 0 |
| m | `rca/trace-cmi-overwrite` (hop0) | 25 | 0.200 | 0.240 | 0.280 | 0.232 [0.088, 0.400] | 0.058 | 0 |
| xs | `rca/trace-cmi-overwrite` (source) | 10 | 0.300 | 0.600 | 1.000 | 0.660 [0.460, 0.840] | 0.600 | 0 |
| s | `rca/trace-cmi-overwrite` (source) | 15 | 0.133 | 0.467 | 0.800 | 0.480 [0.293, 0.653] | 0.250 | 0 |
| m | `rca/trace-cmi-overwrite` (source) | 25 | 0.160 | 0.200 | 0.440 | 0.248 [0.112, 0.400] | 0.058 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| s | `rca/trace-cmi-overwrite` (hop1-subpre) | 15 | 0.200 | 0.533 | 0.733 | 0.467 [0.280, 0.653] | 0.250 | 0 |
| m | `rca/trace-cmi-overwrite` (hop1-subpre) | 25 | 0.040 | 0.160 | 0.280 | 0.160 [0.056, 0.280] | 0.058 | 0 |
| xs | `rca/trace-cmi-overwrite` (hop0-subpre) | 10 | 0.600 | 0.800 | 1.000 | 0.820 [0.620, 0.980] | 0.600 | 0 |
| s | `rca/trace-cmi-overwrite` (hop0-subpre) | 15 | 0.267 | 0.667 | 0.733 | 0.547 [0.333, 0.733] | 0.250 | 0 |
| m | `rca/trace-cmi-overwrite` (hop0-subpre) | 25 | 0.120 | 0.240 | 0.360 | 0.232 [0.104, 0.384] | 0.058 | 0 |
| xs | `rca/trace-cmi-overwrite` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| s | `rca/trace-cmi-overwrite` (source-subpre) | 15 | 0.467 | 0.867 | 0.933 | 0.773 [0.613, 0.907] | 0.250 | 0 |
| m | `rca/trace-cmi-overwrite` (source-subpre) | 25 | 0.160 | 0.440 | 0.600 | 0.408 [0.264, 0.560] | 0.058 | 0 |
| xs | `rca/trace-ig-keep` (hop1) | 10 | 0.200 | 0.600 | 1.000 | 0.600 [0.420, 0.800] | 0.600 | 0 |
| s | `rca/trace-ig-keep` (hop1) | 15 | 0.133 | 0.267 | 0.467 | 0.307 [0.120, 0.507] | 0.250 | 0 |
| m | `rca/trace-ig-keep` (hop1) | 25 | 0.000 | 0.160 | 0.240 | 0.136 [0.048, 0.240] | 0.058 | 0 |
| xs | `rca/trace-ig-keep` (hop0) | 10 | 0.100 | 0.600 | 1.000 | 0.560 [0.380, 0.740] | 0.600 | 0 |
| s | `rca/trace-ig-keep` (hop0) | 15 | 0.133 | 0.267 | 0.333 | 0.240 [0.067, 0.440] | 0.250 | 0 |
| m | `rca/trace-ig-keep` (hop0) | 25 | 0.200 | 0.200 | 0.280 | 0.224 [0.080, 0.392] | 0.058 | 0 |
| xs | `rca/trace-ig-keep` (source) | 10 | 0.400 | 0.600 | 1.000 | 0.680 [0.480, 0.880] | 0.600 | 0 |
| s | `rca/trace-ig-keep` (source) | 15 | 0.133 | 0.467 | 0.800 | 0.467 [0.280, 0.640] | 0.250 | 0 |
| m | `rca/trace-ig-keep` (source) | 25 | 0.160 | 0.240 | 0.480 | 0.280 [0.136, 0.432] | 0.058 | 0 |
| xs | `rca/trace-ig-keep` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| s | `rca/trace-ig-keep` (hop1-subpre) | 15 | 0.267 | 0.533 | 0.800 | 0.533 [0.333, 0.707] | 0.250 | 0 |
| m | `rca/trace-ig-keep` (hop1-subpre) | 25 | 0.040 | 0.160 | 0.240 | 0.152 [0.048, 0.272] | 0.058 | 0 |
| xs | `rca/trace-ig-keep` (hop0-subpre) | 10 | 0.600 | 0.800 | 1.000 | 0.820 [0.620, 0.980] | 0.600 | 0 |
| s | `rca/trace-ig-keep` (hop0-subpre) | 15 | 0.267 | 0.600 | 0.733 | 0.507 [0.307, 0.693] | 0.250 | 0 |
| m | `rca/trace-ig-keep` (hop0-subpre) | 25 | 0.120 | 0.200 | 0.280 | 0.200 [0.080, 0.352] | 0.058 | 0 |
| xs | `rca/trace-ig-keep` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| s | `rca/trace-ig-keep` (source-subpre) | 15 | 0.400 | 0.867 | 0.933 | 0.760 [0.600, 0.893] | 0.250 | 0 |
| m | `rca/trace-ig-keep` (source-subpre) | 25 | 0.240 | 0.440 | 0.560 | 0.424 [0.264, 0.592] | 0.058 | 0 |
| xs | `rca/trace-cmi-keep` (hop1) | 10 | 0.100 | 0.600 | 1.000 | 0.580 [0.400, 0.760] | 0.600 | 0 |
| s | `rca/trace-cmi-keep` (hop1) | 15 | 0.067 | 0.267 | 0.400 | 0.253 [0.093, 0.440] | 0.250 | 0 |
| m | `rca/trace-cmi-keep` (hop1) | 25 | 0.000 | 0.200 | 0.240 | 0.144 [0.048, 0.256] | 0.058 | 0 |
| xs | `rca/trace-cmi-keep` (hop0) | 10 | 0.100 | 0.600 | 1.000 | 0.540 [0.360, 0.720] | 0.600 | 0 |
| s | `rca/trace-cmi-keep` (hop0) | 15 | 0.133 | 0.267 | 0.333 | 0.240 [0.067, 0.440] | 0.250 | 0 |
| m | `rca/trace-cmi-keep` (hop0) | 25 | 0.200 | 0.240 | 0.280 | 0.232 [0.088, 0.400] | 0.058 | 0 |
| xs | `rca/trace-cmi-keep` (source) | 10 | 0.400 | 0.600 | 1.000 | 0.700 [0.500, 0.880] | 0.600 | 0 |
| s | `rca/trace-cmi-keep` (source) | 15 | 0.133 | 0.467 | 0.800 | 0.467 [0.280, 0.640] | 0.250 | 0 |
| m | `rca/trace-cmi-keep` (source) | 25 | 0.160 | 0.200 | 0.440 | 0.248 [0.112, 0.400] | 0.058 | 0 |
| xs | `rca/trace-cmi-keep` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.800 [0.620, 0.940] | 0.600 | 0 |
| s | `rca/trace-cmi-keep` (hop1-subpre) | 15 | 0.200 | 0.533 | 0.800 | 0.480 [0.293, 0.653] | 0.250 | 0 |
| m | `rca/trace-cmi-keep` (hop1-subpre) | 25 | 0.040 | 0.160 | 0.280 | 0.160 [0.056, 0.280] | 0.058 | 0 |
| xs | `rca/trace-cmi-keep` (hop0-subpre) | 10 | 0.600 | 0.800 | 1.000 | 0.820 [0.620, 0.980] | 0.600 | 0 |
| s | `rca/trace-cmi-keep` (hop0-subpre) | 15 | 0.267 | 0.600 | 0.600 | 0.493 [0.267, 0.707] | 0.250 | 0 |
| m | `rca/trace-cmi-keep` (hop0-subpre) | 25 | 0.120 | 0.240 | 0.360 | 0.232 [0.104, 0.384] | 0.058 | 0 |
| xs | `rca/trace-cmi-keep` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| s | `rca/trace-cmi-keep` (source-subpre) | 15 | 0.467 | 0.867 | 0.933 | 0.773 [0.613, 0.907] | 0.250 | 0 |
| m | `rca/trace-cmi-keep` (source-subpre) | 25 | 0.160 | 0.440 | 0.600 | 0.408 [0.264, 0.560] | 0.058 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop1) | 10 | 0.000 | 0.700 | 1.000 | 0.600 [0.440, 0.740] | 0.600 | 0 |
| s | `rca/trace-atomic-cmi-keep` (hop1) | 15 | 0.067 | 0.200 | 0.467 | 0.240 [0.080, 0.427] | 0.250 | 0 |
| m | `rca/trace-atomic-cmi-keep` (hop1) | 25 | 0.000 | 0.080 | 0.240 | 0.104 [0.032, 0.192] | 0.058 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop0) | 10 | 0.000 | 0.600 | 1.000 | 0.540 [0.380, 0.700] | 0.600 | 0 |
| s | `rca/trace-atomic-cmi-keep` (hop0) | 15 | 0.133 | 0.200 | 0.400 | 0.240 [0.067, 0.427] | 0.250 | 0 |
| m | `rca/trace-atomic-cmi-keep` (hop0) | 25 | 0.200 | 0.240 | 0.280 | 0.232 [0.088, 0.400] | 0.058 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (source) | 10 | 0.400 | 0.600 | 1.000 | 0.700 [0.520, 0.880] | 0.600 | 0 |
| s | `rca/trace-atomic-cmi-keep` (source) | 15 | 0.200 | 0.467 | 0.867 | 0.493 [0.307, 0.667] | 0.250 | 0 |
| m | `rca/trace-atomic-cmi-keep` (source) | 25 | 0.160 | 0.200 | 0.440 | 0.248 [0.112, 0.392] | 0.058 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop1-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.780 [0.560, 0.940] | 0.600 | 0 |
| s | `rca/trace-atomic-cmi-keep` (hop1-subpre) | 15 | 0.133 | 0.467 | 0.467 | 0.373 [0.160, 0.587] | 0.250 | 0 |
| m | `rca/trace-atomic-cmi-keep` (hop1-subpre) | 25 | 0.040 | 0.120 | 0.240 | 0.136 [0.040, 0.248] | 0.058 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (hop0-subpre) | 10 | 0.500 | 0.800 | 1.000 | 0.780 [0.560, 0.940] | 0.600 | 0 |
| s | `rca/trace-atomic-cmi-keep` (hop0-subpre) | 15 | 0.267 | 0.667 | 0.667 | 0.533 [0.320, 0.733] | 0.250 | 0 |
| m | `rca/trace-atomic-cmi-keep` (hop0-subpre) | 25 | 0.120 | 0.200 | 0.240 | 0.200 [0.072, 0.352] | 0.058 | 0 |
| xs | `rca/trace-atomic-cmi-keep` (source-subpre) | 10 | 0.600 | 0.900 | 1.000 | 0.880 [0.740, 0.980] | 0.600 | 0 |
| s | `rca/trace-atomic-cmi-keep` (source-subpre) | 15 | 0.533 | 0.867 | 0.867 | 0.800 [0.613, 0.947] | 0.250 | 0 |
| m | `rca/trace-atomic-cmi-keep` (source-subpre) | 25 | 0.200 | 0.400 | 0.560 | 0.408 [0.248, 0.568] | 0.058 | 0 |
| xs | `control/trace-earliest` | 10 | 0.000 | 0.500 | 1.000 | 0.440 [0.340, 0.540] | 0.600 | 0 |
| s | `control/trace-earliest` | 15 | 0.000 | 0.333 | 0.467 | 0.267 [0.107, 0.440] | 0.250 | 0 |
| m | `control/trace-earliest` | 25 | 0.200 | 0.240 | 0.280 | 0.232 [0.088, 0.400] | 0.058 | 0 |

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
| xs | `rca/trace` | 2101.0 [814, 4272] |
| s | `rca/trace` | 2849.5 [2036, 3642] |
| m | `rca/trace` | 4268.5 [3348, 5845] |
| xs | `rca/trace` | 2101.0 [814, 4272] |
| s | `rca/trace` | 2849.5 [2036, 3642] |
| m | `rca/trace` | 4268.5 [3348, 5845] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| s | `rca/trace-core` | 2134.1 [1630, 2856] |
| m | `rca/trace-core` | 2829.7 [1893, 5044] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| s | `rca/trace-core` | 2136.1 [1615, 2843] |
| m | `rca/trace-core` | 2827.6 [1886, 5012] |
| xs | `rca/trace-core` | 2016.4 [874, 4243] |
| s | `rca/trace-core` | 2136.1 [1615, 2843] |
| m | `rca/trace-core` | 2827.6 [1886, 5012] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| s | `rca/trace-ig-overwrite` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-overwrite` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| s | `rca/trace-ig-overwrite` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-overwrite` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| s | `rca/trace-ig-overwrite` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-overwrite` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| s | `rca/trace-ig-overwrite` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-overwrite` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| s | `rca/trace-ig-overwrite` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-overwrite` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-overwrite` | 2057.7 [656, 4262] |
| s | `rca/trace-ig-overwrite` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-overwrite` | 2653.6 [1779, 4568] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| s | `rca/trace-cmi-overwrite` | 2694.2 [1925, 4237] |
| m | `rca/trace-cmi-overwrite` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| s | `rca/trace-cmi-overwrite` | 2694.2 [1925, 4237] |
| m | `rca/trace-cmi-overwrite` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| s | `rca/trace-cmi-overwrite` | 2694.2 [1925, 4237] |
| m | `rca/trace-cmi-overwrite` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| s | `rca/trace-cmi-overwrite` | 2694.2 [1925, 4237] |
| m | `rca/trace-cmi-overwrite` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| s | `rca/trace-cmi-overwrite` | 2694.2 [1925, 4237] |
| m | `rca/trace-cmi-overwrite` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-overwrite` | 2046.4 [933, 4202] |
| s | `rca/trace-cmi-overwrite` | 2694.2 [1925, 4237] |
| m | `rca/trace-cmi-overwrite` | 3576.9 [2549, 5761] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| s | `rca/trace-ig-keep` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-keep` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| s | `rca/trace-ig-keep` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-keep` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| s | `rca/trace-ig-keep` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-keep` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| s | `rca/trace-ig-keep` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-keep` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| s | `rca/trace-ig-keep` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-keep` | 2653.6 [1779, 4568] |
| xs | `rca/trace-ig-keep` | 2489.1 [678, 4262] |
| s | `rca/trace-ig-keep` | 2189.1 [1615, 2652] |
| m | `rca/trace-ig-keep` | 2653.6 [1779, 4568] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| s | `rca/trace-cmi-keep` | 2973.2 [1925, 4237] |
| m | `rca/trace-cmi-keep` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| s | `rca/trace-cmi-keep` | 2973.2 [1925, 4237] |
| m | `rca/trace-cmi-keep` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| s | `rca/trace-cmi-keep` | 2973.2 [1925, 4237] |
| m | `rca/trace-cmi-keep` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| s | `rca/trace-cmi-keep` | 2973.2 [1925, 4237] |
| m | `rca/trace-cmi-keep` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| s | `rca/trace-cmi-keep` | 2973.2 [1925, 4237] |
| m | `rca/trace-cmi-keep` | 3576.9 [2549, 5761] |
| xs | `rca/trace-cmi-keep` | 2515.5 [1085, 4202] |
| s | `rca/trace-cmi-keep` | 2973.2 [1925, 4237] |
| m | `rca/trace-cmi-keep` | 3576.9 [2549, 5761] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| s | `rca/trace-atomic-cmi-keep` | 3518.3 [2855, 4738] |
| m | `rca/trace-atomic-cmi-keep` | 4838.4 [3516, 6860] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| s | `rca/trace-atomic-cmi-keep` | 3518.3 [2855, 4738] |
| m | `rca/trace-atomic-cmi-keep` | 4838.4 [3516, 6860] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| s | `rca/trace-atomic-cmi-keep` | 3518.3 [2855, 4738] |
| m | `rca/trace-atomic-cmi-keep` | 4838.4 [3516, 6860] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| s | `rca/trace-atomic-cmi-keep` | 3518.3 [2855, 4738] |
| m | `rca/trace-atomic-cmi-keep` | 4838.4 [3516, 6860] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| s | `rca/trace-atomic-cmi-keep` | 3518.3 [2855, 4738] |
| m | `rca/trace-atomic-cmi-keep` | 4838.4 [3516, 6860] |
| xs | `rca/trace-atomic-cmi-keep` | 4458.8 [2552, 6213] |
| s | `rca/trace-atomic-cmi-keep` | 3518.3 [2855, 4738] |
| m | `rca/trace-atomic-cmi-keep` | 4838.4 [3516, 6860] |

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
| xs | `rca/trace` | 1589 [1046, 2144] | 435 [333, 500] | 5.0 |
| s | `rca/trace` | 26019 [17429, 43836] | 498 [486, 500] | 12.0 |
| m | `rca/trace` | 20935 [10722, 32016] | 500 [500, 500] | 51.7 |
| xs | `rca/trace` | 1589 [1046, 2144] | 435 [333, 500] | 4.9 |
| s | `rca/trace` | 26019 [17429, 43836] | 498 [486, 500] | 11.7 |
| m | `rca/trace` | 20935 [10722, 32016] | 500 [500, 500] | 42.6 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-core` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-core` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-core` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-core` | 20935 [10722, 32016] | 500 [500, 500] | 51.5 |
| xs | `rca/trace-core` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| s | `rca/trace-core` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-core` | 20935 [10722, 32016] | 500 [500, 500] | 43.4 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-ig-overwrite` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-ig-overwrite` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.5 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| s | `rca/trace-ig-overwrite` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-ig-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 43.5 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-ig-overwrite` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-ig-overwrite` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.5 |
| xs | `rca/trace-ig-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| s | `rca/trace-ig-overwrite` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-ig-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 43.5 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-cmi-overwrite` | 26019 [17429, 43836] | 500 [496, 500] | 12.0 |
| m | `rca/trace-cmi-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-cmi-overwrite` | 26019 [17429, 43836] | 500 [496, 500] | 12.0 |
| m | `rca/trace-cmi-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.4 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| s | `rca/trace-cmi-overwrite` | 26019 [17429, 43836] | 500 [496, 500] | 11.8 |
| m | `rca/trace-cmi-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 42.7 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-cmi-overwrite` | 26019 [17429, 43836] | 500 [496, 500] | 12.0 |
| m | `rca/trace-cmi-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 5.0 |
| s | `rca/trace-cmi-overwrite` | 26019 [17429, 43836] | 500 [496, 500] | 12.0 |
| m | `rca/trace-cmi-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 51.4 |
| xs | `rca/trace-cmi-overwrite` | 1589 [1046, 2144] | 500 [495, 500] | 4.9 |
| s | `rca/trace-cmi-overwrite` | 26019 [17429, 43836] | 500 [496, 500] | 11.8 |
| m | `rca/trace-cmi-overwrite` | 20935 [10722, 32016] | 500 [500, 500] | 42.7 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-ig-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-ig-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.5 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| s | `rca/trace-ig-keep` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-ig-keep` | 20935 [10722, 32016] | 500 [500, 500] | 43.5 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-ig-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-ig-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-ig-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.5 |
| xs | `rca/trace-ig-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| s | `rca/trace-ig-keep` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-ig-keep` | 20935 [10722, 32016] | 500 [500, 500] | 43.5 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.4 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| s | `rca/trace-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 42.7 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.0 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.4 |
| xs | `rca/trace-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| s | `rca/trace-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 42.7 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-atomic-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-atomic-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.4 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-atomic-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-atomic-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.7 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| s | `rca/trace-atomic-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-atomic-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 42.4 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-atomic-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-atomic-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.4 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 5.0 |
| s | `rca/trace-atomic-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 12.0 |
| m | `rca/trace-atomic-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 51.7 |
| xs | `rca/trace-atomic-cmi-keep` | 1589 [1046, 2144] | 500 [500, 500] | 4.9 |
| s | `rca/trace-atomic-cmi-keep` | 26019 [17429, 43836] | 500 [500, 500] | 11.8 |
| m | `rca/trace-atomic-cmi-keep` | 20935 [10722, 32016] | 500 [500, 500] | 42.4 |

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
| H-rca-presub | fail |
| H-rca-atomic-cmi | pass |
