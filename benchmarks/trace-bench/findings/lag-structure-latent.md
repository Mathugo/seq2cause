# Findings — lag structure of the directed truth, and what the arms recover beyond lag 1 (xs … l / latent, five seeds)

Score-side analysis of 2026-10-04 over the landed test reads of the four rungs
(`results/{xs,s,m,l}/latent/seed={0..4}/`); no method run, no new freeze, nothing here is a
pre-registered hypothesis. It answers three questions the per-rung findings leave open: what share
of the directed truth is an immediate callee → caller edge, how the remaining edges are spread
over lag, and how many of the edges that need a lag ≥ 2 reading the arms actually find.

## Definitions

- **Truth** — the directed target edges at the default floor 0.05 (`truth_sets`), per grain;
  bidirected edges are not counted here.
- **Read sample** — the test split of `views/end-<grain>` in file order, every row at `xs` and
  the first 20,000 (request) / 10,000 (session) rows from `s`, truncated to 64 tokens and encoded
  with the view's vocabulary: the sample every test read probed (`--ordering end --sequence-sample
  head --max-len 64 --max-lag 63`).
- **Lag** — `q − j` for a cause token at position `j` and an effect token at `q > j` of one
  sequence: the harness's lag (`per_lag_recall`), not a call-graph distance.
- **Call-linked instance** — an occurrence pair of a truth edge whose cause span's parent is the
  effect span (`parent_pos[j] == q`), read on the score side.
- **Min lag** of a token pair — the smallest lag at which it occurs in the read sample;
  **reachable** = it occurs at all. A min-lag-1 edge can be found from adjacent tokens alone; a
  min-lag ≥ 2 edge can only be found by a reading that reaches past the neighbour.
- **Base rate** — the truth share among the scorer-universe pairs of the same min-lag class, i.e.
  the precision of predicting every such pair.

Tables give five-seed means (± sd over seeds); precision and base rate are pooled over the five
seeds. Every prediction count below reconciles with the landed `score.json` of its cell
(true positives, true + false positives, truth size and out-of-universe predictions equal on all
240 cells: 4 rungs × 5 seeds × 2 grains × 6 cells).

## Summary

1. **By call graph, every request-grain truth edge is an immediate callee → caller edge** — 100 %
   on all 20 corpora: the effect's operation calls the cause's operation directly; no edge spans
   two or more call hops and none runs caller → callee. At the session grain the share is
   69 / 80 / 95 / 98 % (`xs` / `s` / `m` / `l`); the rest are cross-request journey and retry
   edges.
2. **By position, fewer than half of those callee → caller instances are adjacent**: 43–44 % sit
   at lag 1 from `s` up (53 % at `xs`), then 21 % at lag 2, 14 % at lag 3, 9 % at lag 4, 6 % at
   lag 5, 5 % at lag 6–7 and 2 % at lag ≥ 8 (mean lag 2.4). At the type level, of the truth edges
   the test read reaches, 76–77 % (67 % at `xs`) occur adjacent at least once and **23–24 %
   (33 %) only ever occur at lag ≥ 2** — 14–15 % first at lag 2, 5 % at lag 3, 2 % at lag 4, 2 %
   at lag 5–7, 0.2 % at lag ≥ 8. At the session grain the lag ≥ 2 share is 33–36 % (44 % at `xs`)
   with a longer tail (6–9 % at lag ≥ 8).
3. **The tool's own cut finds almost none of the lag ≥ 2 edges; a truth-tuned threshold finds a
   quarter to three quarters of them, at 1.2–3× the base-rate precision.** `trace/cli` under the
   shipped cut finds **0** min-lag ≥ 2 edges at the request grain on all 20 corpora;
   `trace/cli-atomic` finds 2 / 20 / 25 / 14 of the 67 / 284 / 1,150 / 2,056 reachable ones. At
   the frozen τ `trace/core` finds 37 / 72 / 581 / 1,587 (50 / 26 / 50 / 77 % recall) and
   `trace/cli-atomic` 40 / 144 / 744 / 1,728, at 7–16 % precision against a 5–12 % base rate.
   Lag ≥ 2 edges are 10–27 % of a trace arm's true positives at the request grain.

## 1. Call-graph class of the truth

Request grain, all four rungs: 100.0 ± 0.0 % of the directed truth edges `src → dst` have the
`dst` operation as a direct caller of the `src` operation in the corpus topology
(`instantiation.json`); 0 edges at two or more hops, 0 in the caller → callee direction, 0 off the
call graph. The "ancestor ≠ parent" separations noted in the per-rung findings are
instance-level (the same token pair occurring across an ancestor link, ≤ 0.25 % of forward
co-occurrences), not multi-hop truth edges.

Session grain (share of the directed truth):

| rung | directed truth edges | callee → caller, inherited from the request target | BFF → client page (callee → caller) | cross-request (journey / retry) |
|---|---|---|---|---|
| `xs` | 670 | 59.6 ± 5.4 % | 9.2 ± 2.0 % | 31.2 ± 3.8 % |
| `s` | 3,238 | 75.7 ± 1.1 % | 4.4 ± 0.4 % | 19.9 ± 1.2 % |
| `m` | 26,825 | 93.6 ± 0.5 % | 1.0 ± 0.0 % | 5.4 ± 0.5 % |
| `l` | 73,791 | 97.5 ± 0.2 % | 0.6 ± 0.0 % | 1.9 ± 0.2 % |

The cross-request class is where the session target departs from callee → caller: its edges run
from a BFF or client-page token of one request to tokens of a later request, mostly in the
caller → callee direction. Its share shrinks with the rung because the request-inherited truth
grows with the operation count while the journey structure does not.

## 2. Lag of the callee → caller instances (instance level)

Call-linked instances of truth edges in the read sample, request grain, by lag. This is a property
of the `end` ordering and does not depend on the sample size.

| rung | call-linked instances | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 | lag 6–7 | lag ≥ 8 | mean lag | caller first |
|---|---|---|---|---|---|---|---|---|---|---|
| `xs` | 3,392 | 53.3 ± 12.1 % | 21.4 ± 4.6 % | 10.9 ± 3.4 % | 6.9 ± 2.6 % | 3.8 ± 1.4 % | 2.9 ± 1.2 % | 0.9 ± 0.4 % | 2.01 | 0.91 ± 0.22 % |
| `s` | 62,135 | 42.6 ± 6.6 % | 20.6 ± 2.0 % | 13.6 ± 1.3 % | 9.7 ± 2.9 % | 6.4 ± 2.3 % | 5.6 ± 2.2 % | 1.5 ± 0.8 % | 2.44 | 0.83 ± 0.17 % |
| `m` | 64,071 | 43.4 ± 3.0 % | 20.8 ± 0.9 % | 13.6 ± 0.3 % | 9.1 ± 0.8 % | 5.8 ± 1.0 % | 5.3 ± 1.3 % | 1.9 ± 0.7 % | 2.42 | 0.80 ± 0.06 % |
| `l` | 61,044 | 43.8 ± 2.8 % | 21.2 ± 0.9 % | 13.5 ± 0.4 % | 8.8 ± 0.7 % | 5.5 ± 0.8 % | 5.1 ± 1.1 % | 2.0 ± 0.8 % | 2.40 | 0.81 ± 0.09 % |

In the `end` ordering only the last-ending child of a caller can sit immediately before it; every
earlier-ending child is separated from the caller by the later siblings and their subtrees. The
call-linked instances of the session view have the same profile (lag-1 share within one point at
every rung). "Caller first" counts linked instances where the caller's token precedes its
callee's in the `end` ordering — unavailable to any forward reading. Of all forward
co-occurrences of truth pairs, 88 / 94 / 96 / 95 % are call-linked.

Cross-request instances of the session-only edges, by lag — no adjacency peak, 50–62 % of the
mass at lag ≥ 5:

| rung | instances | lag 1 | lag 2 | lag 3 | lag 4 | lag 5–7 | lag 8–15 | lag ≥ 16 |
|---|---|---|---|---|---|---|---|---|
| `xs` | 3,544 | 17.0 ± 2.9 % | 14.2 ± 1.7 % | 10.9 ± 0.6 % | 8.4 ± 0.5 % | 18.1 ± 1.6 % | 22.9 ± 2.3 % | 8.5 ± 3.3 % |
| `s` | 67,071 | 9.2 ± 3.3 % | 10.4 ± 2.7 % | 9.9 ± 1.0 % | 9.0 ± 0.9 % | 21.0 ± 1.0 % | 28.5 ± 1.7 % | 12.0 ± 4.1 % |
| `m` | 45,261 | 8.7 ± 1.1 % | 9.7 ± 0.8 % | 10.7 ± 0.8 % | 9.0 ± 0.3 % | 20.0 ± 1.5 % | 29.9 ± 1.1 % | 12.0 ± 2.4 % |
| `l` | 33,489 | 10.8 ± 2.3 % | 11.9 ± 2.1 % | 10.6 ± 1.2 % | 8.4 ± 0.4 % | 19.3 ± 1.0 % | 27.5 ± 1.5 % | 11.5 ± 4.3 % |

## 3. Min lag of the truth edges (type level)

Shares in the lag columns are of the reachable edges. Unlike section 2, this table depends on the
read sample: a larger sample moves edges from "not reached" into the table and from higher to
lower min lag.

| grain | rung | truth edges | reachable (share of truth) | min lag 1 | min lag ≥ 2 | min lag 2 | 3 | 4 | 5–7 | ≥ 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| request | `xs` | 403 | 204 (50.2 ± 3.8 %) | 137 (**67.2 ± 6.2 %**) | 67 (32.8 ± 6.2 %) | 20.1 ± 5.0 % | 6.4 ± 2.7 % | 3.5 ± 1.6 % | 2.7 ± 0.9 % | 0.1 ± 0.2 % |
| request | `s` | 2,451 | 1,180 (48.1 ± 2.3 %) | 896 (**76.0 ± 2.9 %**) | 284 (24.0 ± 2.9 %) | 13.8 ± 1.9 % | 5.3 ± 1.4 % | 2.4 ± 1.2 % | 2.3 ± 1.1 % | 0.2 ± 0.1 % |
| request | `m` | 25,113 | 5,039 (20.1 ± 1.0 %) | 3,889 (**77.2 ± 0.6 %**) | 1,150 (22.8 ± 0.6 %) | 14.5 ± 0.6 % | 4.6 ± 0.3 % | 1.9 ± 0.2 % | 1.6 ± 0.2 % | 0.2 ± 0.1 % |
| request | `l` | 71,972 | 8,452 (11.8 ± 0.7 %) | 6,397 (**75.7 ± 0.7 %**) | 2,056 (24.3 ± 0.7 %) | 15.3 ± 0.5 % | 4.9 ± 0.3 % | 2.2 ± 0.1 % | 1.7 ± 0.1 % | 0.2 ± 0.1 % |
| session | `xs` | 670 | 419 (62.1 ± 3.9 %) | 235 (**55.8 ± 2.9 %**) | 183 (44.2 ± 2.9 %) | 17.5 ± 3.3 % | 7.5 ± 1.9 % | 5.4 ± 1.3 % | 8.1 ± 1.2 % | 5.6 ± 1.5 % |
| session | `s` | 3,238 | 2,315 (71.4 ± 2.3 %) | 1,477 (**63.9 ± 3.0 %**) | 838 (36.1 ± 3.0 %) | 10.0 ± 1.7 % | 4.9 ± 1.2 % | 4.1 ± 0.8 % | 8.5 ± 0.7 % | 8.5 ± 1.9 % |
| session | `m` | 26,825 | 8,979 (33.5 ± 1.7 %) | 5,983 (**66.7 ± 0.5 %**) | 2,995 (33.3 ± 0.5 %) | 12.2 ± 0.7 % | 4.5 ± 0.2 % | 2.8 ± 0.2 % | 6.4 ± 0.4 % | 7.4 ± 0.5 % |
| session | `l` | 73,791 | 13,081 (17.7 ± 1.1 %) | 8,660 (**66.2 ± 1.0 %**) | 4,421 (33.8 ± 1.0 %) | 13.4 ± 0.4 % | 4.8 ± 0.2 % | 3.1 ± 0.1 % | 6.1 ± 0.3 % | 6.3 ± 0.9 % |

Truth edges with a token the view's vocabulary never mints (17 / 3 / 20 / 35 % of the request
truth) are among the unreached and are unreachable for any arm (`annotate.json:
unreachable_tokens`).

## 4. What the arms recover, by min-lag class

Cells are on the `shipped` path. "Frozen τ" is the validation-selected threshold (selected against
the truth); "the tool's cut" is the shipped pooled-percentile rule with lag decay, which uses no
truth. `baseline/granger` is listed as the reference that ties the trace arms on F1.

Request grain:

| rung | cell | min lag 1: found / reachable | recall | precision (base rate) | **min lag ≥ 2: found / reachable** | recall | precision (base rate) | recall at min lag 2 / 3 / 4 / ≥ 5 |
|---|---|---|---|---|---|---|---|---|
| `xs` | `trace/core`, frozen τ | 115 / 137 | 84.7 ± 9.1 % | 37.8 % (34.0 %) | **37 / 67** | 49.8 ± 33.7 % | 16.4 % (12.3 %) | 58 / 63 / 47 / 34 % |
| `xs` | `trace/cli`, frozen τ | 118 / 137 | 85.8 ± 6.9 % | 35.9 % (34.0 %) | **37 / 67** | 51.6 ± 23.3 % | 16.5 % (12.3 %) | 62 / 60 / 32 / 31 % |
| `xs` | `trace/cli-atomic`, frozen τ | 108 / 137 | 80.1 ± 11.0 % | 37.3 % (34.0 %) | **40 / 67** | 55.9 ± 22.7 % | 15.2 % (12.3 %) | 62 / 68 / 50 / 31 % |
| `xs` | `baseline/granger`, frozen τ | 107 / 137 | 78.3 ± 5.2 % | 39.4 % (34.0 %) | **34 / 67** | 46.4 ± 22.1 % | 17.6 % (12.3 %) | 54 / 52 / 47 / 24 % |
| `xs` | `trace/cli`, the tool's cut | 7 / 137 | 5.2 ± 2.8 % | 52.1 % (34.0 %) | **0 / 67** | 0.0 ± 0.0 % | — (12.3 %) | 0 / 0 / 0 / 0 % |
| `xs` | `trace/cli-atomic`, the tool's cut | 9 / 137 | 6.6 ± 6.0 % | 42.7 % (34.0 %) | **2 / 67** | 2.5 ± 2.3 % | 55.6 % (12.3 %) | 5 / 0 / 0 / 0 % |
| `s` | `trace/core`, frozen τ | 681 / 896 | 76.3 ± 5.1 % | 30.4 % (25.2 %) | **72 / 284** | 25.9 ± 11.4 % | 13.8 % (6.2 %) | 37 / 15 / 9 / 0 % |
| `s` | `trace/cli`, frozen τ | 666 / 896 | 74.7 ± 5.4 % | 30.8 % (25.2 %) | **71 / 284** | 25.7 ± 13.5 % | 16.0 % (6.2 %) | 36 / 16 / 10 / 0 % |
| `s` | `trace/cli-atomic`, frozen τ | 636 / 896 | 71.0 ± 2.1 % | 30.9 % (25.2 %) | **144 / 284** | 50.6 ± 10.7 % | 15.2 % (6.2 %) | 58 / 51 / 40 / 20 % |
| `s` | `baseline/granger`, frozen τ | 710 / 896 | 79.1 ± 4.4 % | 29.0 % (25.2 %) | **124 / 284** | 43.2 ± 13.4 % | 12.2 % (6.2 %) | 54 / 39 / 30 / 13 % |
| `s` | `trace/cli`, the tool's cut | 67 / 896 | 7.4 ± 3.6 % | 55.0 % (25.2 %) | **0 / 284** | 0.0 ± 0.0 % | 0.0 % (6.2 %) | 0 / 0 / 0 / 0 % |
| `s` | `trace/cli-atomic`, the tool's cut | 109 / 896 | 12.1 ± 2.1 % | 39.9 % (25.2 %) | **20 / 284** | 7.1 ± 3.0 % | 36.6 % (6.2 %) | 9 / 6 / 5 / 3 % |
| `m` | `trace/core`, frozen τ | 3,503 / 3,889 | 90.0 ± 1.9 % | 33.3 % (25.5 %) | **581 / 1,150** | 50.2 ± 9.5 % | 13.5 % (4.6 %) | 63 / 38 / 24 / 13 % |
| `m` | `trace/cli`, frozen τ | 3,527 / 3,889 | 90.7 ± 0.8 % | 32.4 % (25.5 %) | **620 / 1,150** | 54.2 ± 5.1 % | 13.8 % (4.6 %) | 66 / 40 / 27 / 20 % |
| `m` | `trace/cli-atomic`, frozen τ | 3,498 / 3,889 | 89.9 ± 1.4 % | 34.3 % (25.5 %) | **744 / 1,150** | 64.7 ± 3.8 % | 9.3 % (4.6 %) | 71 / 61 / 50 / 38 % |
| `m` | `baseline/granger`, frozen τ | 3,435 / 3,889 | 88.3 ± 1.2 % | 35.4 % (25.5 %) | **570 / 1,150** | 49.5 ± 2.2 % | 14.2 % (4.6 %) | 60 / 38 / 26 / 20 % |
| `m` | `trace/cli`, the tool's cut | 303 / 3,889 | 7.8 ± 1.8 % | 82.6 % (25.5 %) | **0 / 1,150** | 0.0 ± 0.0 % | — (4.6 %) | 0 / 0 / 0 / 0 % |
| `m` | `trace/cli-atomic`, the tool's cut | 369 / 3,889 | 9.5 ± 2.0 % | 51.9 % (25.5 %) | **25 / 1,150** | 2.1 ± 0.4 % | 21.0 % (4.6 %) | 3 / 1 / 1 / 2 % |
| `l` | `trace/core`, frozen τ | 6,053 / 6,397 | 94.6 ± 1.3 % | 29.7 % (28.1 %) | **1,587 / 2,056** | 77.4 ± 7.5 % | 7.7 % (5.3 %) | 79 / 77 / 75 / 67 % |
| `l` | `trace/cli`, frozen τ | 6,028 / 6,397 | 94.2 ± 1.2 % | 29.9 % (28.1 %) | **1,621 / 2,056** | 78.8 ± 8.8 % | 7.4 % (5.3 %) | 81 / 78 / 77 / 68 % |
| `l` | `trace/cli-atomic`, frozen τ | 6,193 / 6,397 | 96.8 ± 1.7 % | 29.6 % (28.1 %) | **1,728 / 2,056** | 84.2 ± 8.7 % | 6.5 % (5.3 %) | 85 / 83 / 83 / 76 % |
| `l` | `baseline/granger`, frozen τ | 5,745 / 6,397 | 89.8 ± 2.2 % | 32.9 % (28.1 %) | **1,464 / 2,056** | 71.2 ± 2.6 % | 7.9 % (5.3 %) | 73 / 70 / 70 / 65 % |
| `l` | `trace/cli`, the tool's cut | 214 / 6,397 | 3.4 ± 1.7 % | 70.2 % (28.1 %) | **0 / 2,056** | 0.0 ± 0.0 % | — (5.3 %) | 0 / 0 / 0 / 0 % |
| `l` | `trace/cli-atomic`, the tool's cut | 219 / 6,397 | 3.4 ± 2.2 % | 57.0 % (28.1 %) | **14 / 2,056** | 0.7 ± 0.4 % | 39.3 % (5.3 %) | 1 / 0 / 0 / 0 % |

Session grain:

| rung | cell | min lag 1: found / reachable | recall | precision (base rate) | **min lag ≥ 2: found / reachable** | recall | precision (base rate) | recall at min lag 2 / 3 / 4 / ≥ 5 |
|---|---|---|---|---|---|---|---|---|
| `xs` | `trace/core`, frozen τ | 201 / 235 | 85.1 ± 2.6 % | 43.4 % (36.6 %) | **44 / 183** | 23.6 ± 4.1 % | 19.1 % (10.2 %) | 37 / 29 / 25 / 4 % |
| `xs` | `trace/cli`, frozen τ | 196 / 235 | 83.7 ± 5.1 % | 43.5 % (36.6 %) | **47 / 183** | 25.8 ± 7.8 % | 18.9 % (10.2 %) | 37 / 32 / 30 / 7 % |
| `xs` | `trace/cli-atomic`, frozen τ | 174 / 235 | 73.6 ± 3.2 % | 44.5 % (36.6 %) | **62 / 183** | 34.4 ± 8.1 % | 17.3 % (10.2 %) | 45 / 44 / 39 / 13 % |
| `xs` | `baseline/granger`, frozen τ | 196 / 235 | 83.1 ± 3.7 % | 44.7 % (36.6 %) | **50 / 183** | 27.7 ± 6.9 % | 18.4 % (10.2 %) | 40 / 37 / 29 / 5 % |
| `xs` | `trace/cli`, the tool's cut | 10 / 235 | 4.4 ± 2.0 % | 59.0 % (36.6 %) | **0 / 183** | 0.0 ± 0.0 % | — (10.2 %) | 0 / 0 / 0 / 0 % |
| `xs` | `trace/cli-atomic`, the tool's cut | 22 / 235 | 9.6 ± 4.0 % | 44.9 % (36.6 %) | **6 / 183** | 3.1 ± 2.9 % | 22.3 % (10.2 %) | 7 / 1 / 2 / 0 % |
| `s` | `trace/core`, frozen τ | 1,257 / 1,477 | 85.2 ± 1.4 % | 32.0 % (21.4 %) | **72 / 838** | 8.6 ± 2.6 % | 11.5 % (2.9 %) | 26 / 8 / 2 / 0 % |
| `s` | `trace/cli`, frozen τ | 1,242 / 1,477 | 84.2 ± 1.3 % | 32.3 % (21.4 %) | **73 / 838** | 8.7 ± 2.3 % | 15.2 % (2.9 %) | 25 / 10 / 3 / 0 % |
| `s` | `trace/cli-atomic`, frozen τ | 1,191 / 1,477 | 80.7 ± 1.9 % | 30.4 % (21.4 %) | **205 / 838** | 24.3 ± 3.0 % | 13.7 % (2.9 %) | 55 / 38 / 21 / 4 % |
| `s` | `baseline/granger`, frozen τ | 1,228 / 1,477 | 83.2 ± 1.2 % | 32.8 % (21.4 %) | **58 / 838** | 6.9 ± 1.8 % | 13.3 % (2.9 %) | 20 / 6 / 2 / 0 % |
| `s` | `trace/cli`, the tool's cut | 157 / 1,477 | 10.5 ± 5.4 % | 25.6 % (21.4 %) | **47 / 838** | 4.9 ± 5.2 % | 2.8 % (2.9 %) | 8 / 7 / 3 / 4 % |
| `s` | `trace/cli-atomic`, the tool's cut | 306 / 1,477 | 20.3 ± 6.7 % | 37.1 % (21.4 %) | **78 / 838** | 8.8 ± 6.6 % | 8.2 % (2.9 %) | 15 / 16 / 12 / 4 % |
| `m` | `trace/core`, frozen τ | 5,540 / 5,983 | 92.6 ± 0.6 % | 38.4 % (22.8 %) | **685 / 2,995** | 22.9 ± 1.3 % | 14.3 % (1.5 %) | 52 / 19 / 9 / 1 % |
| `m` | `trace/cli`, frozen τ | 5,443 / 5,983 | 91.0 ± 0.4 % | 38.7 % (22.8 %) | **642 / 2,995** | 21.5 ± 1.3 % | 18.7 % (1.5 %) | 49 / 18 / 9 / 1 % |
| `m` | `trace/cli-atomic`, frozen τ | 5,342 / 5,983 | 89.3 ± 0.6 % | 36.2 % (22.8 %) | **918 / 2,995** | 30.6 ± 3.0 % | 10.0 % (1.5 %) | 56 / 36 / 22 / 8 % |
| `m` | `baseline/granger`, frozen τ | 5,461 / 5,983 | 91.3 ± 0.6 % | 38.9 % (22.8 %) | **711 / 2,995** | 23.7 ± 0.3 % | 15.2 % (1.5 %) | 53 / 22 / 10 / 1 % |
| `m` | `trace/cli`, the tool's cut | 277 / 5,983 | 4.6 ± 0.7 % | 47.7 % (22.8 %) | **21 / 2,995** | 0.7 ± 0.7 % | 1.6 % (1.5 %) | 1 / 1 / 1 / 0 % |
| `m` | `trace/cli-atomic`, the tool's cut | 569 / 5,983 | 9.5 ± 1.1 % | 47.2 % (22.8 %) | **87 / 2,995** | 2.9 ± 1.5 % | 6.1 % (1.5 %) | 5 / 3 / 3 / 1 % |
| `l` | `trace/core`, frozen τ | 7,704 / 8,660 | 89.0 ± 2.4 % | 31.7 % (25.7 %) | **1,791 / 4,421** | 40.4 ± 6.7 % | 8.7 % (1.5 %) | 59 / 52 / 40 / 17 % |
| `l` | `trace/cli`, frozen τ | 7,624 / 8,660 | 88.1 ± 3.9 % | 31.8 % (25.7 %) | **1,681 / 4,421** | 37.9 ± 4.1 % | 7.2 % (1.5 %) | 56 / 48 / 37 / 15 % |
| `l` | `trace/cli-atomic`, frozen τ | 7,361 / 8,660 | 85.1 ± 5.8 % | 34.0 % (25.7 %) | **1,408 / 4,421** | 32.0 ± 7.0 % | 5.9 % (1.5 %) | 48 / 39 / 29 / 13 % |
| `l` | `baseline/granger`, frozen τ | 7,464 / 8,660 | 86.4 ± 6.7 % | 34.8 % (25.7 %) | **1,934 / 4,421** | 43.9 ± 2.9 % | 9.0 % (1.5 %) | 60 / 52 / 44 / 24 % |
| `l` | `trace/cli`, the tool's cut | 82 / 8,660 | 0.9 ± 0.3 % | 25.4 % (25.7 %) | **40 / 4,421** | 0.8 ± 1.0 % | 2.6 % (1.5 %) | 1 / 1 / 1 / 0 % |
| `l` | `trace/cli-atomic`, the tool's cut | 190 / 8,660 | 2.2 ± 0.7 % | 34.0 % (25.7 %) | **55 / 4,421** | 1.2 ± 0.9 % | 2.4 % (1.5 %) | 2 / 1 / 1 / 1 % |

Reading the two tables:

- **At the request grain the tool's cut on `trace/cli` returns adjacent pairs only.** It predicts
  no pair of min lag ≥ 2 on any of the 20 corpora, so it finds none. At the session grain it does
  predict such pairs from `s` up (0 to 121 true positives per seed), at or near the base rate:
  precision 2.8 / 1.6 / 2.6 % against 2.9 / 1.5 / 1.5 % at `s` / `m` / `l`. `trace/cli-atomic`
  under the same cut finds 1–9 % of the lag ≥ 2 edges, at a precision well above the base rate at
  the request grain (21–56 % against 5–12 %).
- **At the frozen τ the arms do reach past lag 1.** Recall on min-lag
  ≥ 2 edges is 26–84 % (request) and 9–40 % (session) and falls steeply with lag at `s` and `m`
  (`m` request, core: 63 / 38 / 24 / 13 % at min lag 2 / 3 / 4 / ≥ 5); precision on that class is
  1.2–3.0× the base rate at the request grain and 1.7–12× at the session grain, where the base
  rate is 1.5–10 %. On the min-lag-1 class precision is 1.05–1.35× the base rate (request).
- **`trace/cli-atomic` is the arm that reads furthest**, as H-construction's lag-graded reading in
  the per-rung findings says: at `s` it finds twice the lag ≥ 2 edges of core / cli (144 against
  72 at request, 205 against 72 at session) at a comparable precision; at `m` 1.2–1.4× at a lower
  precision (9–10 % against 14–19 %); at `l` the three arms converge (request) or the order
  reverses (session).
- **At `l` the request-grain recall of 77–84 % is a threshold effect**: the frozen τ sits at or
  near the bottom of its grid there (`findings/l-latent.md`) and the arms predict 41–48 k of the
  62 k co-occurring universe pairs, so most reachable edges of every lag are returned and
  precision on the lag ≥ 2 class (6.5–7.7 %) is close to its base rate (5.3 %).

## Consequences

- **H-lag's first clause could not pass.** Lag-1 recall is bounded by the min-lag-1 share of the
  reachable truth: 0.67 / 0.76 / 0.77 / 0.76 of the coverage ceiling at the request grain and
  0.56 / 0.64 / 0.67 / 0.66 at the session grain, below the 0.8× the clause asks for on every
  corpus (exact for cells frozen at context 1; see the first caveat otherwise). Of the min-lag-1
  edges themselves the frozen core / cli cells return 75–95 % (request) and 84–93 % (session).
- **A model-free adjacent-pair rule is level with `trace/core` on F1 at four of the eight
  rung × grain cells and ahead at one.** Predicting every adjacent token pair of the read sample
  (no model, no threshold, no truth) scores directed F1 0.335 ± 0.033 / 0.299 ± 0.018 /
  0.193 ± 0.005 / 0.135 ± 0.005 at the request grain and 0.354 ± 0.026 / 0.292 ± 0.010 /
  0.225 ± 0.006 / 0.161 ± 0.005 at the session grain. Paired per seed, `trace/core` frozen minus
  the rule is −0.014 (p = 0.33) / **−0.010 (p = 0.026, 0/5 wins)** / **+0.012 (p = 0.007, 5/5)** /
  +0.001 (p = 0.80) at the request grain and +0.002 (p = 0.69) / **+0.050 (p < 0.001, 5/5)** /
  **+0.045 (p < 0.001, 5/5)** / −0.000 (p = 0.98) at the session grain (paired t-test, n = 5). The
  arms' margin, where there is one, comes from the lag ≥ 2 class and from pruning adjacent pairs.
  The rule is not a registered arm; it is reported as a reference for how much of the headline
  the lag-1 class carries.

## Caveats

- Min lag is measured over whole sequences (context 1). A cell frozen at context `c ≥ 2` does not
  score the first `c − 1` tokens, so its own reachable set is smaller than the "reachable" column:
  equal at `c = 1`, 0.99× at `m` / `l` session, 0.95–0.99× at `s`, 0.86–0.98× at `xs` (one seed
  0.70×, a request cell frozen at `c = 3`). Recall figures for those cells are understated
  by that factor at most.
- Type-level shares (section 3, section 4) are relative to the test read's sample; the validation
  sweeps probe half as many sequences from `s` (the registered sample-size note).
- Counts are of directed truth edges only; the bidirected truth and the SID / AID axes are outside
  this analysis.

## Reproduction

Inputs: per corpus, `graphs/scoring-target{,-session}.json`, `instantiation.json`, the test
split and `model-vocab.json` of `views/end-{request,session}` (dataset revision `v0.3.0`), and
per test read the `prediction-<grain>-<arm>-shipped-{frozen,shipped}.json` files of its discover
folders. Steps: (1) directed truth at the floor; (2) call-graph hops from
`instantiation.json:topology.edges` plus the scenario steps' client → BFF links; (3) one pass over
the read sample accumulating, per ordered cross-operation token pair, its minimum lag, and per
truth edge its forward co-occurrences by lag with the parent link of each; (4) each prediction
restricted to the scorer's universe (`support_op_pairs`) and split by the pair's min lag. The
scripts are score-side lab tooling kept outside the tracked tree.
