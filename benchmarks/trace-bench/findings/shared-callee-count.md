# Shared callees: the sample-level pairs the recorded-parent rule drops (count, 2026-11-11)

In trace-bench an operation with several callers is invoked at most once per request and records
one parent, the first caller that calls it. Every caller present in the request still depends on
it: the simulator reads the final outcome and the time of every invoked callee of an operation,
whichever caller invoked it. The per-sequence truth (`plans/per-sequence-rules.md` §2) marks a
position pair `(j, q)` directed truth only when `j`'s recorded parent is `q`, so the pair
(shared callee, another caller in the same request) is never truth, and a method that marks it is
counted wrong. The per-trace root-cause reference `floor/trace-topology` follows the same parent
links downwards and has the same gap. This file counts how often that happens. Nothing was
re-scored and no landed number changes.

## Sample-level pairs

`scripts/shared_callee_count.py`, one record per corpus and grain in `findings/shared-callee-count/`.
The sample is the landed test read of the cell `trace/core/shipped/frozen/<grain>` (the cell the
paper's sample-level columns use), re-derived from its recorded selection; the truth is re-induced
with `seqtruth.induce`, and the script refuses unless its truth-pair, candidate and sequence totals
equal the landed `seqscore.json`. All 35 runs below passed that check.

A *dropped truth pair* is a candidate pair `(j, q)` inside one request where the operation of `q`
is a call-graph caller of the operation of `j`, the operation of `j`'s recorded parent is another
one, and the token pair is a directed target edge at the 0.05 floor. These are the pairs a
call-graph rule would add to the truth. Counts are pooled over the five seeds; shares of truth
pairs are of the landed truth pairs.

### Request grain

| size | corpora | sequences with candidates | with a dropped truth pair | truth pairs | dropped truth pairs | share of truth pairs (range over seeds) | share of candidate pairs | critical / non-critical edge | with a retried operation | reversed order, token pair a truth edge |
|---|---|---|---|---|---|---|---|---|---|---|
| xs | 5 | 5,795 | 871 (15.0 %) | 15,018 | 1,384 | 9.2 % (1.2–14.7) | 2.77 % | 1,132 / 252 | 553 | 1,928 |
| s | 5 | 98,956 | 13,131 (13.3 %) | 310,677 | 16,483 | 5.3 % (2.5–8.0) | 1.29 % | 10,370 / 6,113 | 2,234 | 55,319 |
| m | 5 | 98,332 | 7,536 (7.7 %) | 320,357 | 9,243 | 2.9 % (2.3–3.4) | 0.74 % | 6,715 / 2,528 | 1,441 | 16,822 |
| l | 5 | 97,971 | 6,976 (7.1 %) | 305,222 | 8,795 | 2.9 % (2.0–3.3) | 0.72 % | 6,971 / 1,824 | 2,057 | 17,112 |

### Session grain

| size | corpora | sequences with candidates | with a dropped truth pair | truth pairs | dropped truth pairs | share of truth pairs (range over seeds) | share of candidate pairs | critical / non-critical edge | with a retried operation | reversed order, token pair a truth edge |
|---|---|---|---|---|---|---|---|---|---|---|
| xs | 5 | 2,262 | 653 (28.9 %) | 32,009 | 1,374 | 4.3 % (0.1–7.5) | 0.76 % | 1,025 / 349 | 557 | 1,890 |
| s | 5 | 49,585 | 15,631 (31.5 %) | 851,504 | 25,386 | 3.0 % (1.8–4.8) | 0.28 % | 15,023 / 10,363 | 5,955 | 89,196 |
| m | 5 | 49,427 | 11,101 (22.5 %) | 771,237 | 16,698 | 2.2 % (1.9–2.6) | 0.18 % | 12,274 / 4,424 | 4,074 | 26,841 |

- Every size and grain measured is above 1 % of the truth pairs.
- The call graph is the shipped `topology/prior.json`; edge criticality is from each corpus's
  `instantiation.json`, whose edge set equals the prior's on all 20 corpora. Between 18 % and
  41 % of the dropped truth pairs sit on non-critical edges, where the callee's time makes the caller slow.
- *Reversed order*: the same dependence where the shared callee is ordered after the other caller
  in the end-ordered view. Only `j < q` pairs are candidates, so no truth rule reaches these; they
  outnumber the dropped truth pairs at every size.
- Not run: the session grain at l (the session target needs about 13 GB on an 18 GB laptop) and
  both grains at xl (no local corpus). The count reads the correlator's parent column, as the
  landed truth does; it was not repeated on the oracle's parent links.

What a call-graph rule would do to the landed pooled F1 of the same read, from the landed counts
(five-seed mean). The reference's new value assumes it predicts every dropped truth pair, so it is
an upper estimate; TRACE's range is from none to all of them being among its predictions.

| grain | size | call graph (reference) | predict every pair | TRACE |
|---|---|---|---|---|
| request | xs | 0.800 → ≤ 0.840 | 0.456 → 0.486 | 0.423 → 0.407–0.484 |
| request | s | 0.833 → ≤ 0.858 | 0.397 → 0.413 | 0.343 → 0.335–0.383 |
| request | m | 0.867 → ≤ 0.881 | 0.409 → 0.418 | 0.444 → 0.439–0.462 |
| request | l | 0.850 → ≤ 0.864 | 0.401 → 0.411 | 0.429 → 0.425–0.442 |
| session | xs | 0.406 → ≤ 0.434 | 0.303 → 0.314 | 0.433 → 0.426–0.462 |
| session | s | 0.493 → ≤ 0.511 | 0.173 → 0.178 | 0.346 → 0.342–0.369 |
| session | m | 0.589 → ≤ 0.602 | 0.151 → 0.154 | 0.412 → 0.408–0.425 |

## Per-trace root-cause reference

`../trace-bench-rca/scripts/shared_callee_rca_count.py`, one record per corpus in
`../trace-bench-rca/findings/shared-callee-count/`. Rule R and the scored set are recomputed with
the benchmark's own functions; every scored trace is ranked on the call tree (the landed rule) and
on the call tree plus every call-graph edge between two operations present in the trace. The
scored-trace totals and the call-tree hit@1 equal the landed reference at every size. The root-cause
label and TRACE's own rows read no parent link and do not change.

| size | scored traces | a failed span reaches more positions | reached services differ (hop0) | first-ranked service differs (hop0) | hit@1 hop0, call tree → with call-graph edges | hit@1 hop1, call tree → with call-graph edges |
|---|---|---|---|---|---|---|
| xs | 3,641 | 65.5 % | 2.2 % | 16.6 % | 0.817 → 0.925 | 0.589 → 0.607 |
| s | 50,979 | 57.9 % | 9.6 % | 11.5 % | 0.808 → 0.790 | 0.446 → 0.367 |
| m | 16,758 | 36.1 % | 4.6 % | 6.0 % | 0.695 → 0.682 | 0.191 → 0.171 |
| l | 5,234 | 32.0 % | 4.4 % | 4.5 % | 0.693 → 0.668 | 0.169 → 0.143 |

- hit@1 is the mean over cases of the per-case rate, as in the landed tables.
- With the call-graph edges the reference rises by 0.11 at xs and falls by 0.01–0.03 from s to l:
  a shared callee that failed without being the root cause is promoted too.
- Not run: xl (the label table needs about 11 GB on an 18 GB laptop).
