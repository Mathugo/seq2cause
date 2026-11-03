# External-arm plan: CASCADE with the deployment topology (pre-registered 2026-11-03, before the arm code)

Committed before `cascadearm.py` exists. Registers the `external` arm family: a published
discovery method that is not seq2cause, run on the trace-bench corpora under the registered
protocol beside the reference arms, the baselines and the floors. Its first member is CASCADE
(Cüppers, Xu, Ahmed, Vreeken, NeurIPS 2024; corpus slug `cascade-cuppers-2024`; public code
`github.com/joschac/cascade`, commit `b717308d`), the matching learner whose paper reports
142 of 147 edges on Huawei alarms when constrained to the device topology — run here with the
shipped deployment topology as a hard structural prior, and once without it. Owner decisions
(publication PRD Decision Log, 2026-11-03): the arm enters the CLeaR 2027 submission under one
results freeze moved to Friday 6 November 2026; xs, s and m with five seeds each, or the arm is
absent; the no-prior configuration at xs only; fits on the registered validation and test
samples. Every number the paper quotes for this family comes from a read governed by this
file; the lab's probe of 2026-11-03 (a hand-built outcome rule on the topology's pairs) is
provenance, not a result. Changes are dated addenda at the end, never edits in place.
Deviation ids are RUN.md's D-SB-n.

## 1. Arms

| arm | class | reads (declared inputs) | statistic | paths | cuts | sample |
|---|---|---|---|---|---|---|
| `cascade/topology` | `external` | **method side:** the views only, through `Corpus` (`views/end-<grain>/sequences/split=<val\|test>`, columns `trace_id, ops, outcomes, n_spans, start_ns, offsets, durations`); **score side:** `topology/prior.json` and `graphs/alphabet.json` through `floorprior.topology_read` — the same reader and the same `(callee token, caller token)` pair set as `floor/topology` — handed to the method as the set of pairs it may consider. A declared **topology-informed** arm, as `rca/circa-topology` is (`benchmarks/trace-bench-rca/plans/rca-arms.md`, addendum 2026-10-25). | `cascade_dag`: 1.0 for every edge of the DAG CASCADE returns, no other pair scored | `none` | `frozen` | the arms' (`plans/caps.md`: whole split at xs; 10k / 5k validation, 20k / 10k test from s; `head`; `--max-len 64`, D-SB-7) |
| `cascade/none` | `external` | the views only, as above; no prior | `cascade_dag` | `none` | `frozen` | the same; **xs only** (owner decision) |

Neither arm has a backbone, a probe pass, a `c` or a `g` axis; the records carry `0` for both.
The `N` slot carries the method's one swept setting, `max_delay` in milliseconds (§3, D-SB-28).
Neither predicts a bidirected edge (`structural_limitation` as for every arm here). The model
hash of every record is the literal `none` (the harness's model-free binding, D-SB-16); the
method's identity is recorded beside it as `method_sha256` (§2).

## 2. The method and its inputs

- **§2 the code.** CASCADE at commit `b717308d` with the lab's patch `0001-allowed-pairs`
  (`algos/cascade-cuppers-2024/patches/`, recorded in that directory's RUN.md): `cascade(events,
  …, allowed_pairs)` threads a set of `(cause, effect)` event-type pairs into
  `get_delays.get_event_delays` (which then computes delays for those pairs only and returns the
  long-format table over them, indexed `cause · N + effect`) and into `topological_search`
  (whose per-iteration recompute maps that index to the positional candidate array);
  `find_topological_order.get_next_node` replaces `list.index` by a dictionary with the same
  order. With `allowed_pairs = None` the output equals the shipped code's on the shipped example
  data (verified before the first run). Shipped settings kept: `precision 2`, `allow_instant
  True`, the shifted geometric delay distribution, `align_mode next`, `instant_idf True`,
  `optimize_dst False`, `mle_bic False`. The method runs in its own conda environment (`cascade`:
  python 3.11 and the repository's `requirements.txt`, numpy 1.23.5) as a subprocess of the arm
  (`seq2causebench.cascade_runner`, driven by the `cascade` environment's interpreter named on
  the command line); the arm records `method_sha256` = sha256 over the patched `src/*.py` files
  of the pinned clone, the commit and the patch name.
- **§2 events.** The sampled sequences of the split (`--num-sequences`, `--sequence-sample`,
  `--seed` as the arms; D-SB-14), each read as its first `--max-len` (64) real tokens (D-SB-7),
  in the view's row order. One event per token: type = `<op>:<outcome>` from the view rows'
  `(op, outcome)` integers and the outcome names (the floors' token space, `plans/floors.md`
  §2; never the view vocabulary), time = the span's **end** in integer milliseconds relative to
  the earliest sampled event (`start_ns + offsets[j] + durations[j]`, nanoseconds floored to
  milliseconds). Events are sorted by time (ties keep the view's row order); types are
  re-indexed densely `0 .. N−1` over the types present in the sample and mapped back on read-out.
  The end time is the direction the target encodes (a callee completes before its caller;
  `plans/reference-arms.md` §2, D-SB-7), so a truth-direction delay is non-negative. Both grains
  are read from their own view (`end-request`, `end-session`), one fit per grain and setting.
- **§2 the mask (`cascade/topology`).** `allowed_pairs` = the ordered token pairs `floor/topology`
  predicts (`plans/floors.md` §2: every `(t_callee, t_caller)` of every direct call edge of
  `topology/prior.json`), restricted to the types present in the sample and mapped to their
  dense ids. Direction is kept (callee → caller); no reverse pair, no self pair, no client-op
  pair is allowed. A pair outside the mask can never be scored, so the arm's recall at either
  grain is at most the topology floor's (§5). `cascade/none` passes no mask.
- **§2 read-out.** The returned adjacency over dense ids maps back to token strings; every edge
  is scored 1.0; within-op pairs are dropped at projection as for every arm (D-SB-10); the
  table carries its own token list and is written in the sweep's shape (one file per setting,
  `scores-<probe>-none-c0-N<max_delay_ms>-<grain>.npz`, `c = g = 0`, `N = max_delay_ms`, the
  model hash `none`), so `scoresweep`, `freeze` and `annotate` run unchanged.

## 3. Grids (validation sweep, blind)

- **§3 settings.** `max_delay` ∈ {1 000, 10 000, 60 000} ms (1 s, 10 s, 60 s), one fit each on the
  validation sample, carried as `N`. Nothing else is swept: `precision`, the delay family and the
  instant-effect flag are the published defaults (§2).
- **§3 τ.** The floors' grid `floor_taus` (`plans/floors.md` §3, 20 values `0 … 0.99`), strict
  (`score > τ`). Every scored pair has score 1.0, so every τ of the grid keeps the whole DAG and
  the swept rows of one setting are equal by construction; the freeze's tie-break (larger τ)
  then records τ = 0.99, and `every_scored_pair_f1` equals the cell's F1. The grid is swept for
  uniformity of record, not because it selects.
- **§3 freeze rule.** Per cell `cascade/<arm>/none/frozen/<grain>`: the argmax of validation
  directed F1 at the default floor over `(N, τ)`; ties → smaller `N` (the shorter window), then
  larger τ — the harness's own rule (`plans/reference-arms.md` §4, `freeze.select_cells`). A
  frozen `N` at either end of the settings grid is recorded, not refused.
- **§3 floor.** The scorer's default floor (0.05); the coverage rule is reported, never gated.

## 4. Protocol

Per corpus and grain, the floors' protocol with the arm's own stage in place of `floors`:
`cascade --split val` (every arm of the rung in one call; three fits per arm) → `scoresweep
--arms cascade/… --floor-taus` → `freeze --pretrain-results none` (its own dated file
`freezes/<date>-<rung>-<variant>-s<k>-external.json`, family `external`; the arms' and the
floors' freezes of the corpus are untouched) → one committed freeze → `cascade --split test
--freeze … --cells <one cell>=<τ> --max-delay-ms <the frozen N>` (**one arm per test read**,
because a test read binds one `(c, N, g)`; one fit) → `annotate --pretrain-results none
--per-lag-files 0` → landing under `results/<rung>/latent/seed=<k>/cascade/<arm>/none/frozen/<grain>/`
with the read's records under `test-read/cascade-<arm>-<grain>/` and the validation records
under `cascade-val-sweep/`, ledger entries of stage `cascade` (`results_sha256`, `model_sha256 =
"none"`, `method_sha256`, `read`, `cells`), and the `report` rows and pairs of §6. Five seeds per
rung; **xs, s and m** (owner decision 2026-11-03: a rung is reported with five seeds or not at
all, and the arm enters the submission only with all three); l and xl are not run under this
plan. Staging rule as the floors' (`plans/floors.md` §4): `smallest-rung` at xs; above xs the
read names the **arms'** previous-rung ledger entry of the same grain (`sweep` for the
validation read, `discover` for the test read). Every read records `ordering = end`, the
sample, the settings, `method_sha256`, the interpreter and the wall clock of every fit. Corpora
are read from the published release only (`chadyuk/trace-bench`, `v0.3.0`); the score tier
carries `topology/`. xs and s run locally; m runs locally or on CPU hosts (one replica per
seed, owner-applied), sized from the measured s wall clock.

**Timing gate (owner's rule, stated in advance).** The s seed-0 validation fit is timed
first. If the m reads cannot land five seeds by Friday 6 November 18:00 local, the arm is
absent from the submission and the fact is recorded here as an addendum and in the paper's
reproducibility appendix ("released and unevaluated for this method"); nothing landed moves.

## 5. Caveats carried beside every number

- **Hard mask (`cascade/topology`).** The method can only return pairs the topology names, so its
  directed recall at either grain is at most `floor/topology`'s (request: 1.0; session: the
  topology's coverage of the journey and retry edges, 0.60 at xs … 0.94 at m), and every gain
  over the floor is a precision gain. The session-grain journey and retry edges are unreachable
  by construction.
- **Sample-bound, sample-paired.** Both arms fit on the arms' validation and test samples (the
  bigram floor's rule; `plans/floors.md` §5), so the pairs of §6 are sample-paired with the
  arms and the bigram floor and not with the topology floor; the sample-size note of
  `plans/caps.md` (validation 10k / 5k vs test 20k / 10k from s) travels with every s and m
  number.
- **A hard output.** CASCADE returns a graph, not a ranking: its ranking is its edge set, so its
  directed AUROC and average precision are those of a binary predictor and are reported, never
  compared with a ranked arm's.
- **Causal sufficiency.** CASCADE assumes no hidden confounding; the latent corpora violate it by
  design (bidirected truth edges are never predicted, as for every arm here).
- **Five pairs: low power** (`report`'s own note).

## 6. Report rows and pairs

The external cells are **not** in the required set (`cascade/none` runs at xs only). Pairs
added to the report of every rung the family ran on (frozen cut, both grains; a pair shares
corpus and seed, not a model — D-SB-15):
`cascade/topology/none/frozen : floor/topology/none/frozen`, `: floor/bigram/none/frozen`,
`: trace/core/shipped/frozen`, `: baseline/granger/shipped/frozen`; at xs also
`cascade/none/none/frozen : floor/bigram/none/frozen`, `: trace/core/shipped/frozen` and
`cascade/topology/none/frozen : cascade/none/none/frozen`.

## 7. Hypotheses (scored pass or fail per rung in `findings/cascade-latent.md`)

- **H-cascade-topology-request.** At the frozen cut and the default floor, `cascade/topology`'s
  directed F1 at the request grain is **at least 0.02 above `floor/topology`'s** on every rung
  (five-seed mean of the paired difference, same corpus and seed, p < 0.05). *Fail:* any rung
  where the mean gain is below 0.02 or p ≥ 0.05.
- **H-cascade-topology-session.** The same at the session grain. *Fail:* as above (the recall
  ceiling of §5 makes this the harder test).
- **H-cascade-vs-core.** `cascade/topology`'s directed F1 is **above `trace/core/shipped/frozen`'s**
  at both grains on every rung (paired, p < 0.05). *Fail:* any rung and grain where it is not.
- **H-cascade-none-xs.** At xs, `cascade/none`'s request-grain directed F1 is **at least 0.05
  above `floor/bigram`'s** (paired, p < 0.05); its difference to `trace/core/shipped/frozen` is
  reported with no threshold. *Fail:* the gain is below 0.05 or p ≥ 0.05.
- **H-cascade-mask-gain.** At xs, `cascade/topology` − `cascade/none` on request-grain directed
  F1 is **positive** (paired, p < 0.05): the prior helps the method. *Fail:* otherwise.

## 8. Deviations (RUN.md rows, numbered before the first run)

- **D-SB-27** — the `external` arm family: a published method outside seq2cause, run by the
  harness as a subprocess in its own environment, model-free in the harness's binding
  (`model_sha256 = "none"`; `freeze` and `annotate` take `--pretrain-results none`; its freeze
  is a separate dated file `…-s<k>-external.json`), with the method's identity carried as
  `method_sha256`; the topology prior reaches it through the score-side `floorprior` reader, as
  the topology floor's does, never through `Corpus`.
- **D-SB-28** — the record's `N` slot carries the method's swept setting (`max_delay` in ms) and
  `c = g = 0`, so the sweep-shaped score tables, the freeze's `(c, N, τ)` argmax and the test
  read's `(c, N, g)` binding run unchanged; a test read fits one arm at its frozen setting.

## 9. Provenance (not results)

- 2026-11-03, the lab's probe (memory `topology-floor-slack-is-noncritical-edges`): on the
  train split, a per-call-edge failure-lift rule over the topology's pairs reached request F1
  0.90 (xs) … 0.95 (s) against the floor's 0.82 … 0.85, with the outcome templates read off the
  truth — an upper bound on the room, not a method. The session grain gained less (+0.05 …
  +0.09) and adding adjacent pairs did not help.
- CASCADE's own Network Alarms result (142 / 147 edges, no spurious edges, device topology as a
  hard constraint; its Appendix B.4) is the published precedent for the masked configuration.
