# The `forced-ok-counterfactual` truth on xs (2026-11-11)

First build of the second per-sequence truth rule (`plans/per-sequence-rules.md`, addendum
2026-11-11) on the five xs corpora, test split, the stored sequences of the read that holds
`trace/core/shipped/frozen/<grain>`. Records per corpus and grain in
`findings/forced-ok-counterfactual-xs/`. No arm is re-scored here: the rows below need no stored
score matrix. Counts are pooled over the five seeds; F1 is the five-seed mean of the pooled
per-corpus value, as in the landed tables.

## Checks that ran

- The re-run of every shard of every corpus (20 shards) equals its oracle span and session tables
  row for row and reproduces the linkage table's client records.
- The narrowed evaluation of the sampled requests equals the shard re-run.
- Forcing an operation a request did not invoke changes nothing in it (392 runs).
- Under `first-caller-parent` the call-graph and frequency rows reproduce the landed records
  (tp, fp, fn) on all ten reads.
- Every truth file loads through the window-index gate; a second build of seed 0 is byte-identical.

## What is scored

| | request | session |
|---|---|---|
| stored sequences | 6,314 | 2,325 |
| with a candidate, `first-caller-parent` | 5,795 | 2,262 |
| with a candidate, `forced-ok-counterfactual` | 2,823 | 1,492 |
| left without a candidate: fully healthy / other | 3,174 / 317 | 681 / 152 |
| with a truth pair | 1,321 | 1,049 |
| candidate pairs, landed → new | 49,952 → 21,596 | 180,597 → 54,010 |
| dropped: OK target | 28,356 | 107,191 |
| dropped: client retry | 0 | 18,092 |
| dropped: position without a request | 0 | 1,304 |

The positions without a request are in 12 device-level rows of the session view, which hold
client records only.

## Old against new truth

| | request | session |
|---|---|---|
| directed truth pairs, landed → new | 15,018 → 3,397 | 32,009 → 4,059 |
| landed pairs kept | 2,139 | 2,821 |
| landed pairs lost: OK target | 8,651 | 24,030 |
| landed pairs lost: forcing the source does not change the target | 4,228 | 4,430 |
| landed pairs lost: client retry / position without a request | 0 / 0 | 726 / 2 |
| new pairs gained | 1,258 | 1,238 |

The new directed pairs:

| | request | session |
|---|---|---|
| truth under a single forcing / only with the other callees forced | 2,892 / 505 | 3,554 / 505 |
| source: final attempt / retried first attempt / edge | 2,054 / 1,343 / 0 | 1,987 / 1,333 / 739 |
| critical edge / non-critical edge (time path) | 3,133 / 264 | 3,071 / 249 |
| token pair is a population-level edge: any strength / at 0.05 | 3,384 / 2,835 | 4,047 / 3,513 |
| within-request by the oracle, cross-request by the view's root links | — | 112 |

Directed re-run pairs that cannot be scored:

| reason | request | session |
|---|---|---|
| the client error record is not in the row | 907 | 899 |
| source ordered after the target | 777 | 776 |
| a position outside the window (the context position, truncation) | 260 | 369 |
| not a candidate (client tokens are outside the request-grain universe) | 772 | 0 |
| the view's outcome differs from the engine's | 3 | 3 |

## The model-free rows under both rules

Pooled directed F1 (precision, recall), five-seed mean.

| grain | row | `first-caller-parent` | `forced-ok-counterfactual` |
|---|---|---|---|
| request | call graph (reference) | 0.800 (0.672, 1.000) | 0.474 (0.312, 1.000) |
| request | frequency baseline | 0.481 (0.344, 0.801) | 0.350 (0.223, 0.816) |
| request | predict every pair | 0.456 | 0.263 |
| session | call graph (reference) | 0.406 (0.372, 0.450) | 0.345 (0.221, 0.792) |
| session | frequency baseline | 0.382 (0.262, 0.715) | 0.297 (0.182, 0.831) |
| session | predict every pair | 0.303 | 0.136 |

- At the request grain the call graph still finds every truth pair, and two thirds of the pairs
  it marks did not carry the failure in that request.
- At the session grain its recall rises from 0.45 to 0.79. The new truth has no cross-request
  pairs, which a call graph cannot hold; how much of the rise that accounts for was not measured.
- **Limitation (added 2026-11-12, owner's decision: kept as it is).** The pair (edge position, the
  error client record of its own request) is scored at the session grain only: 739 of the 4,059
  session truth pairs here (18 %; 18–22 % by size), and never a candidate at the request grain. The
  call graph has no client link, so the reference misses exactly these pairs and its session recall
  is capped there (0.79 here, 0.78–0.79 by size). The two grains do not score the same pair types
  (`plans/per-sequence-rules.md`, addendum 2026-11-12).

## Not done

- The arms: `seqscore` does not take the rule yet, and the stored matrices are not pulled.
- s to xl: one shard re-run takes 0.3–0.5 s at xs; the larger sizes are not profiled.
