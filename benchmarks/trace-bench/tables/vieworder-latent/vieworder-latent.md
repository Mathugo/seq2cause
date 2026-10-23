# latent — the view-order statistic (describes the view, not an arm)

Share of co-occurrences of a directed truth edge (at the default floor) in which the cause token comes before the effect token in the view's row order, pooled over truth edges and sampled test sequences (`plans/floors.md`, addendum 2026-10-22). Five-seed mean [min, max]; the unweighted per-edge mean beside it. Reported beside every request-grain orientation number of both views, never gated.

## request grain

| rung | ordering | share_cause_first | edge_mean_share (mean) | sequences per seed | seeds |
|---|---|---|---|---|---|
| xs | end | 0.891 [0.842, 0.927] | 0.904 | 1223–1304 | 5 |
| xs | start | 0.102 [0.029, 0.164] | 0.129 | 1223–1304 | 5 |
| s | end | 0.858 [0.753, 0.922] | 0.900 | 20000 | 5 |
| s | start | 0.049 [0.032, 0.071] | 0.121 | 20000 | 5 |
| m | end | 0.947 [0.924, 0.979] | 0.965 | 20000 | 5 |
| m | start | 0.038 [0.032, 0.040] | 0.099 | 20000 | 5 |
| l | end | not run: no score tier on disk for this rung | — | — | 0 |
| l | start | not run: no score tier on disk for this rung | — | — | 0 |
| xl | end | not run: no score tier on disk for this rung | — | — | 0 |
| xl | start | not run: no score tier on disk for this rung | — | — | 0 |

## session grain

| rung | ordering | share_cause_first | edge_mean_share (mean) | sequences per seed | seeds |
|---|---|---|---|---|---|
| xs | end | 0.513 [0.500, 0.528] | 0.534 | 443–487 | 5 |
| xs | start | 0.507 [0.490, 0.544] | 0.474 | 443–487 | 5 |
| s | end | 0.505 [0.484, 0.518] | 0.664 | 10000 | 5 |
| s | start | 0.491 [0.483, 0.498] | 0.451 | 10000 | 5 |
| m | end | 0.560 [0.557, 0.563] | 0.794 | 10000 | 5 |
| m | start | 0.455 [0.444, 0.467] | 0.335 | 10000 | 5 |
| l | end | not run: no score tier on disk for this rung | — | — | 0 |
| l | start | not run: no score tier on disk for this rung | — | — | 0 |
| xl | end | not run: no score tier on disk for this rung | — | — | 0 |
| xl | start | not run: no score tier on disk for this rung | — | — | 0 |
