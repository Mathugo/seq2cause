# Per-sequence axis — rules (pre-registered 2026-09-24, before `seqtruth.py` / `seqscore.py`)

PRD Goal 6, carried decision 4, scenarios 34–37. The per-sequence axis
scores what the method emits for **one** sequence, over position pairs,
against a truth induced for that sequence from two benchmark-shipped facts:
the call links in the views' parent column and the grain's type-level target
at the floor. Its numbers never share a table with a type-level number.
Every formula is imported from `tracebench.score` (`prf`, `auroc`,
`average_precision`, `build_universe`); the instrument adds only the
induction of the truth and the aggregation over sequences.

## 1. Inputs (score side only)

`seqscore` reads a `discover` run's stored matrices (`matrices-<probe>-<noise>-<grain>.npz`:
the full strict-upper triangle of every probed sequence, one column per
path), the run's recorded sequence selection (`--num-sequences`,
`--sequence-sample`, seed, `--max-len`, `c`), and re-reads the same
sequences of the same split with the columns `trace_id, ops, outcomes,
parent_pos` through the benchmark's accessor. The parent column is read
here and nowhere else (scenario 34). Matrix index `t` (0-based within the
window) corresponds to `ops` index `c + t − 1` (BOS occupies position 0 and
counts toward `c`).

## 2. Truth induction

- **Request grain.** Position pair `(j, q)`, `j < q`, is a directed truth edge
  iff `parent_pos[o_j] == o_q` (under the `end` ordering the callee `j`
  precedes its caller `q`) **and** `(tok_j, tok_q)` is a directed edge of the
  request target at the floor.
- **Session grain.** `root(x)` follows `parent_pos` to `−1`. Within one
  request (`root(j) == root(q)`) the request rule applies. Across requests
  (`j < q`, `root(j) ≠ root(q)`) the pair is a directed truth edge iff
  `(tok_j, tok_q)` is a directed edge of the session target at the floor that
  is **absent** from the request target at the floor (the journey and retry
  edges only the session grain carries). This cross-request rule is an
  approximation over occurrence pairs and is stated as such wherever the
  numbers appear.
- **Bidirected target pairs** (latent variant) induce an *adjacency* between
  every occurrence pair `(j, q)` of the two tokens, with no call-link
  requirement (a confounder does not follow the call tree); scored on the
  skeleton basis only.
- **Links to truncated positions** (a parent beyond `--max-len`) are counted
  as `n_links_lost` and dropped.
- **Ancestor-link column (secondary diagnostic, never the headline):** the
  request rule with "`q` is any ancestor of `j` through the parent chain"
  in place of the direct parent, reported beside the parent-link numbers,
  because the full staircase targets total effects and the atomic
  construction direct effects.

## 3. Candidates and edge sets

- Candidate pairs: `(j, q)`, `j < q`, `q − j ≤ max_lag`, both tokens real
  (no specials), `(tok_j, tok_q)` in the scorer's universe
  (`build_universe(alphabet, target)`), so the type-level projection of the
  same cells reaches exactly the scorer's universe (scenario 37).
- Edge set, frozen cut: `S[j, q] > τ` (strict, the frozen τ of the cell).
- Edge set, shipped cut (cli arms): `apply_tau_by_lag(S, tau_by_lag)` with
  the read's recorded `tau_by_lag`, restricted to the candidates.

## 4. Metrics and aggregation

- A sequence is **scoreable** when at least one truth pair (directed or
  adjacency) lies among its candidates. Per scoreable sequence: directed and
  skeleton `prf`; `auroc` and `average_precision` over its candidates
  (directed on the raw scores; skeleton on `S[j, q]`, the only orientation
  the matrix carries); per-lag recall by truth lag.
- **Distribution:** mean, standard deviation, p10, p50, p90 over scoreable
  sequences.
- **Pooled:** one `prf` over the union of every sequence's tp / fp / fn
  (unscoreable sequences contribute false positives only); one `auroc` /
  `average_precision` over all candidate pairs of the split (unscoreable
  sequences' pairs enter as negatives). A sequence with no truth pair enters
  **no** per-sequence AUC.
- Beside every headline: `n_sequences`, `n_scoreable`, `scoreable_fraction`,
  `n_skipped_short`, `n_links_lost`, and the **predict-all audit value**
  `2p / (n + p)` (pooled: `p` truth pairs among `n` candidates over the
  split; and its per-sequence mean). A headline at the predict-all value is
  read as a collapsed threshold.

## 5. Output

`seqscore.json` per cell and grain: `schema, axis: "per-sequence", grain,
arm, path, cut, corpus_id, model_sha256, floor, tau | tau_by_lag,
n_sequences, n_scoreable, scoreable_fraction, n_skipped_short, n_links_lost,
per_sequence{directed{precision, recall, f1}, skeleton{…}, auroc, ap: each
{mean, std, p10, p50, p90}}, pooled{directed prf, skeleton prf, auroc, ap},
predict_all{pooled, per_sequence_mean}, per_lag_recall{lag: value},
ancestor{…the same block…}, rules_version`. Per-sequence tables are their own
report document per grain (`tables/<rung>-<variant>-perseq.*`); a request
table and a session table are never merged.

## 6. Hypotheses

- **H-perseq.** The pooled per-sequence directed F1 at the frozen cut is
  **above the predict-all value** on every scoreable cell of every rung.
  *Fail:* any cell at or below it.
- **H-perseq-fraction.** The scoreable fraction on the request grain is
  **≥ 0.3** at every rung (half of request traces are root-only). Reported;
  a lower value is a finding about the substrate, not a failure of the arm.

## Addenda

### 2026-09-25 — after the xs test read (`xs/latent/seed=0`)

- **The ancestor-link column equals the parent-link column in every one of the 38 cells.** The
  truth induction keeps only candidate pairs whose token pair is a target directed edge, and at
  xs those are direct call pairs, so no strict-ancestor pair survives the filter and
  `ancestor_directed` ≡ `directed` by construction. The column stays (it separates on a corpus
  whose target graph carries a transitive edge) but is reported as redundant wherever the two
  coincide; it is never a headline (§ above).
- **Predict-all at the request grain is high** (0.44–0.52 pooled): request windows hold few
  candidates and most are truth pairs, so the pooled per-sequence F1 of the reference arms sits
  below it on this seed. Reported per H-perseq, not adjusted.

### 2026-09-26 — after the xs test reads of seeds 1–4

- **The ancestor-link column equals the parent-link column on four of the five xs corpora, not
  five.** On `xs/latent/seed=4` the target graph carries at least one transitive edge, so a
  strict-ancestor candidate survives the filter on every one of its 38 cells and the two columns
  separate — by at most 0.0013 pooled F1 (the ancestor column is lower on the trace arms and
  Granger, higher on saliency). The 2026-09-25 note's "at xs" is a per-corpus fact; the column is
  reported wherever it differs and stays out of the headline.
- **Predict-all at the request grain stays above the reference arms on five seeds** (pooled
  predict-all 0.45–0.50 vs pooled F1 0.42–0.51): only `trace/cli-atomic` clears it, on two seeds.
  H-perseq therefore fails at request on the five-seed record; at session every reference arm and
  Granger clear predict-all on every seed while saliency (2 of 5) and Shapley (0 of 5) do not.
- **`s/latent/seed=0` (2026-09-27):** the ancestor-link column equals the parent-link column on
  all 38 cells. Predict-all at request (pooled 0.449) again exceeds every reference arm's pooled
  F1 except `trace/cli-atomic` (0.531); at session (0.187) every reference arm, Granger and Shapley
  clear it while saliency does not. One seed, provisional (`findings/s-latent-s0.md`).


### 2026-10-04 — reading order, and reachable-only columns on the type-level axis (owner decision)

Not a pre-registered hypothesis and not a change to any landed number: a rule for how the two
axes are read, and four derived columns.

- **The per-sequence axis leads the root-cause-analysis reading.** The question a root-cause
  analysis asks — which of the events *in the trace in front of me* caused which — is the one
  this axis scores: position pairs of one sequence against the truth induced for that sequence
  (§2). Wherever a result is framed as a root-cause-analysis result, the per-sequence table is
  given first, with its predict-all value and scoreable fraction (§4). The type-level axis
  answers a different question — how much of the system's graph a fixed number of traces
  recovers — and keeps the benchmark's own scorer output as its headline.
- **Reachable-only precision, recall and F1** are reported beside the benchmark's on every
  type-level cell (`report`, columns `reachable.precision`, `reachable.recall`, `reachable.f1`,
  `reachable.predict_all_f1`). The *reachable* truth of a cell is the directed truth edges whose
  token pair the cell's own read scored at all — `coverage.reachable_recall_ceiling ×
  coverage.truth_directed`, an integer. The columns are derived from the landed `score.json` and
  `annotate.json`, never from a second scorer: the benchmark's true and false positives are
  kept, only the unreachable false negatives leave the denominator. So precision equals the
  benchmark's, recall equals the benchmark recall over the ceiling, and the predict-all value is
  the F1 of predicting every scored pair of the universe. The two tables never merge with the
  per-sequence tables (scenario 35 stands).
- **τ is not re-selected.** The frozen τ of every cell is the one selected on the benchmark F1
  (arm plan §4); the reachable columns are read at that τ. Under a low ceiling the benchmark F1
  favours predicting more, so a reachable F1 read at the frozen τ is a lower bound on what a
  selection on the reachable metric would reach, and an arm ordering on it is partly an
  ordering of where each arm's τ landed. No freeze is redone.
- **Comparability.** Two cells are comparable on the reachable columns only where their reads
  scored the same sample. The Shapley baseline probes its own smaller sample from `s` on
  (D-SB-14), so its reachable truth is a smaller, more frequent subset and its reachable columns
  are not comparable with the other arms'; they are reported and marked.
- **Backfill.** `report` was run again on `xs`, `s`, `m` and `l` on 2026-10-04 (schema
  `report@2`); every previously landed value of the four tables is unchanged, the per-sequence
  documents are byte-equal in content, and the pre-registered pairs now carry the paired test on
  the reachable columns too. Cross-rung reading: `findings/rca-reading-latent.md`.
