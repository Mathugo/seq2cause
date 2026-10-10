# Floor-arm plan (pre-registered 2026-10-09, before the floor code)

Committed before `floors.py` exists. Registers the `floor` arm family the harness PRD's Decision
Log entry of 2026-10-08 names (owner decision in the publication PRD, Decision Log 2026-10-08,
plan questions 3 and 15): two model-free arms that fill the Arm Register's `trivial/*` slot
("not run here") and run under the registered protocol beside the reference arms and baselines
on every rung. Every number the paper quotes for a floor comes from a read governed by this
file; the probes of 2026-09-28 and 2026-10-03 are provenance, not results. Changes are dated
addenda at the end, never edits in place. Deviation ids are RUN.md's D-SB-n.

## 1. Arms

| arm | class | reads (declared inputs, score side unless stated) | statistic | paths | cuts | sample |
|---|---|---|---|---|---|---|
| `floor/topology` | `floor` | `topology/prior.json` (the shipped deployment topology, callee → caller) and `graphs/alphabet.json` (token ↔ op); never a view, never through `Corpus` | `p_call` of the call edge, parsed from the edge's `evidence` field (`topology.edges[i]; p_call=0.xxxx`, four decimals) | `none` | `frozen` | none — sample-free (§5) |
| `floor/bigram` | `floor` | the views only, through `Corpus` (`views/<ordering>-<grain>/`), method side; token strings from the view rows' `(op, outcome)` integers, never the alphabet | `bigram_share`: the share of sampled sequences containing `a` in which `b` immediately follows `a` | `none` | `frozen` | the arms' (`plans/caps.md`: whole split at xs; 10k / 5k validation, 20k / 10k test from s; `head`; `--max-len 64`, D-SB-7) |

Neither arm has a `c`, `N` or `g` axis; the records carry `0`. Neither predicts a bidirected edge
(`structural_limitation` as for every arm here). The model hash of every floor record is the
literal `none`.

## 2. Score rules

- **§2 topology.** For every edge of `topology/prior.json` (`from` = callee column, `to` = caller
  column; columns are `<service>:<endpoint>` and map one-to-one onto the alphabet's
  `(service, name)`), every ordered token pair `(t_callee, t_caller)` with `t_callee` an alphabet
  token of the callee op and `t_caller` one of the caller op is predicted with score `p_call`
  (a repeated op pair keeps the larger `p_call`). Pairs the scorer's universe does not contain
  are dropped by the scorer and counted (`predictions_outside_universe`). The prior lists no
  client op, so no client-op pair is ever predicted. The same prediction is scored at both
  grains (the universes differ).
- **§2 bigram.** On the sampled sequences of the split, in the view's row order, with
  `tok_j = <op_j>:<outcome_j>`: `n_pair(a, b)` = sequences holding at least one position `j` with
  `tok_j = a`, `tok_{j+1} = b` and `op_j ≠ op_{j+1}`; `n_tok(a)` = sequences holding `a` at least
  once; `score(a → b) = n_pair(a, b) / n_tok(a)`. Within-operation pairs are dropped at
  projection as for every arm (D-SB-10). Each sequence is read as the arms read it: its first
  `--max-len` (64) real tokens (D-SB-7). Not chosen: dividing by occurrences of `a`, or by all
  sequences.
- **§2 token space.** Both arms predict `<op_id>:<outcome>` tokens, the scorer's own space: the
  topology arm from `graphs/alphabet.json`, the bigram arm from the view rows' `(op, outcome)`
  integers and the outcome names. Neither uses the view vocabulary, which folds unminted outcome
  variants into their base op. A floor's score table carries its own token list.

## 3. Grids (validation sweep, blind)

- **§3 τ.** One grid for both arms, `floor_taus`, over probabilities including 0:
  `0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99`
  (20 values). Selection is strict (`score > τ`), so `τ = 0` is the cut "every pair the arm
  scores" — for the topology arm the probe's cell, for the bigram arm every adjacent pair observed.
  No negative τ (both scores are non-negative). Freeze rule as in `plans/reference-arms.md` §4:
  argmax of validation directed F1 at the default floor; ties → larger τ (no `N`, `c` to break on).
  A frozen τ at either end of the grid is flagged `at_grid_edge` and recorded, not refused. For
  the topology arm the full ranking is the `τ = 0` cut, so its `every_scored_pair_f1` equals that
  row's F1 by construction.
- **§3 floor.** The scorer's default floor (0.05); the coverage rule is reported, never gated.

## 4. Protocol

Per corpus and grain, the arms' protocol with one stage in place of `sweep`/`discover`:
`floors --split val` (both arms in one call) → `scoresweep --floor-taus` → `freeze
--pretrain-results none` (its own dated file `freezes/<date>-<rung>-<variant>-s<k>-floors.json`;
the arms' freeze of the corpus is untouched) → one committed freeze → `floors --split test
--freeze … --cells …` → `annotate --pretrain-results none --per-lag-files 0` → landing under
`results/<rung>/latent/seed=<k>/floor/<arm>/none/frozen/<grain>/` with the read's records under
`test-read/floors-<grain>/`, ledger entries of stage `floors` (`results_sha256`,
`model_sha256 = "none"`, `read`, `cells`), and the `report` rows and pairs of §6. Five seeds per
rung, xs … xl; xs, s, m locally; l and xl on CPU hosts (one replica per seed, owner-applied).
Staging rule: `--prior-rung-record smallest-rung` at xs; above xs the floors name the **arms'**
previous-rung ledger entry of the same grain (`sweep` for the validation read, `discover` for
the test read) — the entries every arm run named — so the rungs' floor runs are independent of
each other and l and xl can run in parallel; the record carries the matched entry. Every read
records `ordering`; the registered reads are on the `end` view (the view every arm job reads);
the start view is the addendum below. The freeze document keeps schema `freeze@3` with three
additive keys (`family`, `ordering`, `floor_taus`); a floor freeze is named
`…-s<k>-floors.json` and holds floor cells only. Corpora are read from the published release
only (`chadyuk/trace-bench`, `v0.3.0`, through `pull`; the score tier carries `topology/`).

## 5. The sample-free caveat (topology)

The topology arm reads no sequence, so its validation and test reads are the same computation
and differ only in the τ the freeze fixes. Its "test" number is therefore not a held-out number
and is not sample-paired with any arm; every table and findings line that quotes it says so
beside the number ("sample-free: validation and test reads coincide except for τ"). Its τ is
still chosen on the validation read and fixed before the test read, for uniformity of record.
The bigram arm is sample-bound like the arms (same splits, same caps, same `head` rule; the
2026-09-27 sample-size note of `plans/caps.md` applies to it).

## 6. Report rows and pairs

Both floor cells join the required set of every rung's `report` (absent → a `--reasons` entry,
e.g. xl "not run" under the owner's decision of 2026-10-08). Pairs added to every rung's report
(frozen cut, both grains; the pair shares corpus and seed, not a model — D-SB-15):
`floor/topology/none/frozen : trace/core/shipped/frozen`, `: baseline/granger/shipped/frozen`,
`: baseline/saliency/none/frozen`; the same three for `floor/bigram/none/frozen`.

## 7. Hypotheses (from the probes; scored pass or fail per rung in `findings/floors-latent.md`)

- **H-floor-topology.** At the frozen cut and the default floor, `floor/topology`'s directed F1 at
  the request grain is **above every learned arm's** on every rung (five-seed mean of the paired
  difference, same corpus and seed). *Fail:* any rung where a learned arm ties or wins.
- **H-floor-bigram.** At the frozen cut, `floor/bigram`'s directed F1 is **within 0.02 of
  `trace/core/shipped/frozen`'s** at both grains on every rung (paired, five seeds). *Fail:* a
  paired difference beyond 0.02 in either direction with p < 0.05.
- **H-floor-auroc.** The learned arms keep an advantage over `floor/bigram` on session-grain
  directed AUROC (≥ 0.02, paired). *Fail:* the floor ties or wins.

## 8. Deviations (RUN.md rows, numbered before the first floor run)

- **D-SB-15** — `report`'s pair assertion on equal `(corpus_id, seed, model_sha256)` (PRD
  scenario 9): a pair with a model-free floor arm is compared on `(corpus_id, seed)` only and the
  pair record says `model_free: true`.
- **D-SB-16** — the model binding of `scoresweep`, `freeze` and `annotate`: floor records carry
  the literal `model_sha256 = "none"`, `freeze` and `annotate` take `--pretrain-results none` for
  them (oracle, budget and regime fields absent, `in_regime: null`), a floor freeze is a separate
  dated file, and any mixture of model-bound and model-free cells in one freeze or one pretrain
  binding is refused.

## 9. Provenance of the probes (not results)

- 2026-09-28, `publications/trace-bench/.claude/analysis/topology-baseline-26-09-28/`
  (`hop1-up`, `p_call` read numerically from `instantiation.json`, scored with
  `tracebench.score.score_corpus` at floor 0.05, xs and s latent, five seeds; no τ sweep). The
  registered arm reads the shipped prior instead; its four-decimal `p_call` equals the numeric
  value to 1e-4, so the two differ only at τ ties in the fourth decimal.
- 2026-10-03, bigram and view-order probe; scripts were session-scratch (the recipe survives in
  the lab's notes). Its denominator was not written down; §2 fixes it.

## Addenda

### 2026-10-09 — the start-ordered-view read (owner decision 2026-10-08, publication PRD Decision Log, plan question 1)

- **Scope.** xs and s, **request grain only** (the session grain carries no clock artefact:
  cause-before-effect share ≈ 0.50 in the probe), every reference arm and shipped baseline and
  both floors, on `--ordering start`.
- **Protocol.** The registered protocol run anew on the start view: `pretrain` (a new backbone on
  the start-ordered sequences; re-reading an end-trained model is a distribution shift, not a
  read) → `sweep` → `scoresweep` → `freeze` → `discover --split test` → `annotate`; the floors
  through §4 with `--ordering start`. Grids, caps, sample rule and freeze rule as registered.
- **Records.** Every record carries `ordering: start`; freezes are named
  `freezes/<date>-<rung>-latent-s<k>-start.json` and `…-floors-start.json`; results land under a
  sibling root `results-start/<rung>/latent/seed=<k>/…` with the same layout, ledger entries
  carry `ordering`, tables under `tables/<rung>-latent-start/`; `report` runs on that root
  unchanged. A test read refuses a freeze whose `ordering` is not its own.
- **Topology arm.** Reads no view; its start-view cell equals its end-view cell by construction
  and is run and recorded for uniformity, labelled so.
- **View-order statistic.** Reported beside every request-grain orientation number: over the
  split's sampled sequences, the share of co-occurrences of a truth-directed token pair (at the
  default floor) in which the cause precedes the effect in the view's row order, per rung, grain
  and ordering (score side; the 2026-10-03 probe gave 0.90–0.92 end / 0.05–0.09 start at request).
- **Fallback (decided).** If the s chain is not launched by 23 October: arms on xs; on s the
  floors' start-view read and the statistic; logged in the publication PRD Decision Log.
