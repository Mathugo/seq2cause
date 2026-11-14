# External-arm plan: THP (gCastle `TTPM`) with the deployment topology (pre-registered 2026-11-04, before any read)

The second member of the `external` arm family (`plans/cascade.md` and its addendum of
2026-11-04, D-SB-27 … D-SB-29): the **topological Hawkes process** of Cai, Wu, Qiao, Hao, Zhang
and Zhang (IEEE TNNLS 35(1), 2024; corpus slug `thps-cai-2024`), the method whose defining
assumption is a *known undirected topology over the sources of events* and whose published
contribution is that folding that topology into the Hawkes intensity turns topological
propagation from a confounder into a modelled term. Its reference implementation is gCastle's
`castle.algorithms.TTPM` (Huawei Noah's Ark Lab; `github.com/huawei-noah/trustworthyAI`,
`gcastle/`), which takes the topology as `topology_matrix` *and* a separate `priori_knowledge`
(forbidden / required edges) — the two seams this arm uses: the deployment topology as the
method's own topology (node = op) and the topology floor's pair set as a hard mask. Owner
decisions (publication PRD Decision Log, 2026-11-04): xs and s with five seeds each or the arm is
absent; m reported where it lands by the Phase A gate (Monday 10 November); no unmasked
configuration (`cascade/none` at xs is the family's unmasked precedent). Every number the paper
quotes for this arm comes from a read governed by this file. Changes are dated addenda at the
end, never edits in place. Deviation ids are `D-TH-n` (this arm's own, listed in §8 and in
`algos/thps-cai-2024/RUN.md`); the family's are RUN.md's `D-SB-27 … D-SB-29`.

## 1. Arms

| arm | class | reads (declared inputs) | statistic | paths | cuts | sample |
|---|---|---|---|---|---|---|
| `thp/topology` | `external` | **method side:** the views only, through `Corpus` (`views/end-<grain>/sequences/split=<val\|test>`, columns `trace_id, ops, outcomes, n_spans, start_ns, offsets, durations`); **score side:** the topology prior and the alphabet through `floorprior.topology_read` — the same reader and the same `(callee token, caller token)` pair set as `floor/topology` and `cascade/topology` — handed to the method twice: as the undirected op-level topology those pairs span (`op_topology`, the method's `topology_matrix`) and as the set of type pairs it may consider (every other pair forbidden in its `PrioriKnowledge`). A declared **topology-informed** arm. | `ttpm_bic_edge`: 1.0 for every edge of the DAG TTPM's BIC hill-climb returns, no other pair scored | `none` | `frozen` | the arms' (`plans/caps.md`: whole split at xs; 10k / 5k validation, 20k / 10k test from s; `head`; `--max-len 64`, D-SB-7) |

No backbone, no probe pass, no `c` or `g` axis (both recorded as `0`). The `N` slot carries the
method's one swept setting, `decay_ms` = 1/δ of the exponential kernel in milliseconds (§3). The
model hash of every record is the literal `none`; the method's identity is `method_sha256` (§2)
with `method_commit` beside it. No bidirected edge is ever predicted (`structural_limitation`).

## 2. The method and its inputs

- **§2 the code.** gCastle 1.0.4 at commit `8675af0bc532c7f138158ded62e8cf303e9af655` of
  `huawei-noah/trustworthyAI` (`pip install "gcastle @ git+…@8675af0b…#subdirectory=gcastle"`),
  **with the lab's patch `0001-decomposable-score`** (`scripts/envs/thp/`, applied onto the
  installed `castle/algorithms/ttpm/ttpm.py` by `scripts/envs/thp-env.sh`; owner decision
  2026-11-04, §9, D-TH-7): the same search, the same output, at a cost that lets it run to
  convergence. `ttpm.py` (patched) and `castle/common/priori_knowledge.py` are what
  `method_sha256` hashes (`hash_globs` of the registry), with the commit from the install's
  `direct_url.json`, the registered `hash_extras` (the patch name) and the runner's own bytes.
  The method runs in its own environment (`thp`: python 3.11, numpy 2.4.6, pandas 3.1.0,
  networkx 3.6.1, scipy 1.17.1, scikit-learn 1.9.1, torch 2.15.0 CPU — torch only because
  `castle.algorithms` imports it; `scripts/envs/thp-env.sh` + `thp-requirements.txt` rebuild it
  from scratch with `venv` + pip) as a subprocess of the generic read
  (`seq2causebench.externalarm --method thp`, runner `seq2causebench.thp_runner`, interpreter
  and site-packages named on the command line as `--method-python` / `--method-src`).
- **§2 what TTPM does with the inputs** (read off the pinned `ttpm.py`; line numbers are its).
  `learn` takes a frame with columns `event`, `timestamp`, `node`; `_start_init` (l.131–176)
  casts the timestamp to float, **groups identical `(event, timestamp, node)` rows into one
  row with a count** (l.138–140), sorts by `(node, timestamp)`, **drops every row whose `node`
  is not a node of the topology graph** (l.144: `isin(self._topo.nodes)`; `nx.Graph(matrix)`
  holds every row index of the matrix, isolated ops included, so with `op_topology` nothing is
  dropped), and sizes its matrix over the event names it then sees, sorted (l.147–150). The
  intensity of type *v* at node *n* sums, per hop *k* ≤ `max_hop`, the exponentially decayed
  counts of every type at the nodes exactly *k* hops from *n* (l.259–302), **strictly earlier**
  events only (l.294: `nei_start_t < start_t` — two events at the same millisecond never excite
  each other); the kernel is `exp(-δ · Δt)` with Δt in the timestamp's own unit (l.286, 295).
  There is **no binning beyond the timestamp resolution**: the harness hands integer
  milliseconds, so Δt = 1 ms is the paper's "minimal interval". The search (l.208–249) is a
  hill-climb from the graph with only the diagonal on (self-excitation is always a parent,
  l.224, 336) over single-edge add / delete / reverse moves (l.401–430), each candidate refitted
  by the closed-form EM over *every* type (l.304–392) and scored by BIC
  `L − (N + ε·|E|·(K+1)) · log(Σ counts) / 2` (l.386–390; `|E|` includes the diagonal); a cyclic
  candidate scores −10¹⁴ (l.321–326), so **the output is a DAG by construction**; the search
  stops when no neighbour improves or after `max_iter` sweeps. A pair the `PrioriKnowledge`
  forbids (or requires) is never moved (l.418), **but its candidate is still handed to the EM**
  (l.234–236 refit the unchanged matrix once per such pair, deterministic and never an
  improvement) — the shipped search costs one full EM per *ordered pair of types* per sweep,
  not per open pair, and **every sweep accepts exactly one move** (the best improving
  neighbour of the sweep's starting graph), so `max_iter` bounds the edge count. The patch of
  §9 keeps the search and the numbers and removes the cost (§4).
- **§2 events.** The family's (`plans/cascade.md` §2): the arms' sample of the split, each
  sequence's first 64 real tokens, one event per token, type `<op>:<outcome>` re-indexed densely
  over the sample, time = the span's end in integer milliseconds relative to the earliest
  sampled event, sorted (ties in row order). New for this method (addendum 2026-11-04): the
  **node of every event is its op** (`events.npz["node"]`, the dense op id; `n_ops`), and
  `op_topology` is the undirected op adjacency the allowed pairs span (`A[op(a), op(b)] =
  A[op(b), op(a)] = 1` for every allowed `(a, b)`). The runner passes `event` = the dense type
  id (an integer: TTPM sorts the names it sees, so position *i* is type *i*; the runner checks
  the round trip on the matrix labels), `timestamp` = the millisecond, `node` = the op, and
  `topology_matrix` = `op_topology`. Every allowed pair joins two ops that are adjacent in
  `op_topology`, so `max_hop = 1` is exactly the reach the mask needs (with `max_hop = 0` a
  cross-op parent contributes nothing and no masked pair could ever be found).
- **§2 the mask.** `allowed.npy` = the ordered token pairs `floor/topology` predicts, restricted
  to the types present in the sample, as dense ids. The runner builds `PrioriKnowledge(n_types)`
  and adds **every off-diagonal pair not in the mask as a forbidden edge** (`n_forbidden_pairs`
  in the sidecar; at xs seed 0 request grain: 77 types, 551 allowed, 5 301 forbidden); the
  hill-climb can then add an allowed pair only. The registry's `mask_mode = refuse`: an edge
  outside the mask on read-out is a defect, and the read refuses. `delta = 1 / decay_ms`.
- **§2 read-out.** `ttpm.causal_matrix` (row = cause) with its diagonal stripped, as a float32
  0/1 matrix; the runner asserts the labels are `0 … n−1` in order, the values are 0/1 and the
  graph is acyclic, and writes the sidecar (`n_types`, `n_events`, `n_ops`, `n_rows` after
  TTPM's grouping, `n_allowed_pairs`, `n_forbidden_pairs`, `n_nonzero`, `n_edges_outside_mask`,
  `wall_seconds`, `setting`, `delta`, `method_args`, `method_version` = gcastle version @
  commit, `method_env` = the interpreter's `pip freeze`). The harness then maps dense ids back
  to tokens, drops within-op pairs (D-SB-10; none can occur under the mask), scores every edge
  1.0 and writes the sweep-shaped table `scores-thp-topology-none-c0-N<decay_ms>-<grain>.npz`
  (`c = g = 0`, model hash `none`, `method_sha256`, `method_commit`, `setting_name =
  decay_ms`), so `scoresweep`, `freeze` and `annotate` run unchanged.

## 3. Grids (validation sweep, blind)

- **§3 settings.** `decay_ms` ∈ {100, 1 000, 10 000} (δ = 0.01, 0.001, 0.0001 per millisecond:
  kernel e-folding times of 0.1 s, 1 s, 10 s), one fit each on the validation sample, carried
  as `N`. The paper hand-sets δ (0.11 on its alarm data, App. F) and never tunes it; the three
  values bracket the callee-end → caller-end delays of the traces (milliseconds to seconds).
- **§3 fixed method-args** (the runner's registered keys and defaults; recorded verbatim in
  every record): `max_hop = 1` (§2: the mask's reach; the paper's K is "the farthest topological
  distance the events can affect", chosen by prior knowledge), `penalty = BIC` (the paper's
  score, Eq. 10), **`max_iter = 5000`** (a safety cap: the hill-climb stops by itself when no
  move improves the score — the paper's Algorithm 1 — and one sweep accepts one move, so the
  cap is also a cap on the edge count; gCastle's default of 20 would have bounded the arm at
  20 edges, §9; owner decision 2026-11-04, D-TH-8), `epsilon = 1` (gCastle's default BIC
  coefficient). Nothing else is swept; gCastle exposes no other knob. Every record carries
  `n_sweeps_run`, `converged` (true when the search stopped before the cap) and `max_iter`; a
  fit that hits the cap is recorded, not refused, and reported with `converged = false` beside
  its number.
- **§3 τ.** The floors' grid `floor_taus` (`plans/floors.md` §3), strict (`score > τ`). A hard
  output: every scored pair has score 1.0, every τ keeps the whole DAG, the rows of one setting
  are equal by construction; the freeze's tie-break (larger τ) records the grid's top and
  `every_scored_pair_f1` equals the cell's F1. Swept for uniformity of record only.
- **§3 freeze rule.** Per cell `thp/topology/none/frozen/<grain>`: the argmax of validation
  directed F1 at the default floor over `(N, τ)`; ties → smaller `N` (the faster decay), then
  larger τ — `freeze.select_cells`. A frozen `N` at either end of the grid is recorded, not
  refused.
- **§3 floor.** The scorer's default floor (0.05); the coverage rule is reported, never gated.

## 4. Protocol

The family's (`plans/cascade.md` §4 with the addendum's landing): per corpus and grain,
`externalarm --method thp --arms thp/topology --split val --settings 100 1000 10000
--method-args max_hop=1 penalty=BIC max_iter=5000 epsilon=1` (three fits) → `scoresweep --arms
thp/topology --floor-taus` → `freeze --pretrain-results none` (its own dated file
`freezes/<date>-<rung>-latent-s<k>-external-thp.json`, family `external`, `method = thp`) → one
committed freeze → `externalarm --method thp --split test --freeze … --cells <one cell>=<τ>
--settings <the frozen N>` (one arm, one fit) → `annotate --pretrain-results none
--per-lag-files 0` → landing under `results/<rung>/latent/seed=<k>/thp/topology/none/frozen/<grain>/`
(test-read records under `test-read/thp-topology-<grain>/`, validation records under
`thp-val-sweep/`), ledger entries of stage `thp` (`results_sha256`, `model_sha256 = "none"`,
`method_sha256`, `method_commit`, `read`, `cells`), and the report rows and pairs of §6.
**xs and s: five seeds each, required** (a rung is reported with five seeds or not at all, and
the arm enters the submission only with both); **m where it lands by Monday 10 November**
(the Phase A gate), the paper saying so where it does not; l and xl are not run under this
plan. Staging rule as the floors' (`smallest-rung` at xs; above xs the arms' previous-rung
ledger entry of the same grain). Every read records `ordering = end`, the sample, the
settings, the method-args, `method_sha256`, `method_commit`, the interpreter, the method's
`pip freeze` (`run_meta.json["method_env"]`) and the wall clock of every fit. Corpora from the
published release only (`chadyuk/trace-bench`, `v0.3.0`). **Local CPU** (the method is numpy
on one core; no GPU anywhere in it): xs and s run locally; m locally or on CPU hosts (one
replica per seed, built by `scripts/envs/thp-env.sh`), sized from the measured s wall clock.

**Timing gate and the cost note (stated in advance).** The xs seed-0 validation fit at the
middle setting was timed on 2026-11-04 before this plan was committed (§9): the shipped
search costs one full EM (over every type) per ordered pair of types per sweep whatever the
prior forbids (§2), ≈ 10 min per sweep at xs, and would need hundreds of sweeps to converge;
the patched search (§9, D-TH-7) costs a handful of per-type fits per candidate and, across
sweeps, refits only the candidates the last move touched. s and m run on CPU replicas built by
`scripts/envs/thp-env.sh`, sized from the measured xs per-sweep time and the extrapolation in
`algos/thps-cai-2024/RUN.md` §5; the s seed-0 validation fit is timed first. If the three s
fits of one seed cannot land in a working day on the available hosts, the arm is reported at xs
only, as an addendum here and in the paper's reproducibility appendix.

## 5. Caveats carried beside every number

- **Hard mask.** The method can only return pairs the topology names, so its directed recall at
  either grain is at most `floor/topology`'s (request: 1.0; session: the topology's coverage of
  the journey and retry edges, 0.60 at xs … 0.94 at m), and every gain over the floor is a
  precision gain. The session-grain journey and retry edges are unreachable by construction.
- **A DAG against a truth that need not be one.** TTPM refuses every cyclic candidate (§2), so
  it can return at most one direction of any pair of mutually-caused types and no cycle of
  types, while the request-grain target over `<op>:<outcome>` tokens can hold both directions
  of an op pair (an op's outcomes caused by, and causing, a neighbour's) — those edges are
  split or lost, and the loss is reported as recall, not excused.
- **A hard output.** TTPM returns a graph, not a ranking: its directed AUROC and average
  precision are those of a binary predictor and are reported, never compared with a ranked
  arm's.
- **Time at millisecond resolution.** Events at the same millisecond never excite each other
  (§2); a callee that ends in the same millisecond as its caller contributes nothing to that
  edge. Identical `(type, millisecond, op)` rows are merged into a count. Timestamps are span
  ends; the direction the target encodes (callee completes before its caller) makes the
  truth-direction delay non-negative, as for every arm here.
- **A topology that is the mask's span, not the deployment graph.** `op_topology` joins exactly
  the ops the allowed pairs join (the deployment topology's call edges over ops present in the
  sample); ops with no allowed pair are isolated nodes and their types can gain no parent. The
  paper's result that a *missing* topological edge costs precision, not recall (§9), is the
  reason the full span is given rather than a sparser one.
- **Causal sufficiency and excitation only.** THP assumes no hidden confounding among event
  types and its EM yields non-negative strengths (excitation only); the latent corpora violate
  the first by design, and inhibition is invisible to it.
- **An environment newer than the method.** gCastle 1.0.4 predates numpy 2 and pandas 3; the
  pinned commit runs under them (§9's smoke test and probe), but no claim is made that it
  equals the authors' environment bit for bit.
- **Five pairs: low power** (`report`'s own note).

## 6. Report rows and pairs

The arm's cells are **not** in the required set. Pairs added to the report of every rung the
arm ran on (frozen cut, both grains; a pair shares corpus and seed, not a model — D-SB-15):
`thp/topology/none/frozen : floor/topology/none/frozen`, `: floor/bigram/none/frozen`,
`: trace/core/shipped/frozen`, `: baseline/granger/shipped/frozen`; and, where both landed on
the rung, `thp/topology/none/frozen : cascade/topology/none/frozen` (two hard-mask methods on
the same prior: the comparison the family exists for).

## 7. Hypotheses (scored pass or fail per rung in `findings/thp-latent.md`)

- **H-thp-topology-request.** At the frozen cut and the default floor, `thp/topology`'s directed
  F1 at the request grain is **at least 0.02 above `floor/topology`'s** on every rung (five-seed
  mean of the paired difference, same corpus and seed, p < 0.05). *Fail:* any rung where the
  mean gain is below 0.02 or p ≥ 0.05.
- **H-thp-topology-session.** The same at the session grain. *Fail:* as above (the recall
  ceiling of §5 makes this the harder test).
- **H-thp-vs-core.** `thp/topology`'s directed F1 is **above `trace/core/shipped/frozen`'s** at
  both grains on every rung (paired, p < 0.05). *Fail:* any rung and grain where it is not.

## 8. Deviations (`D-TH-n`; rows in `algos/thps-cai-2024/RUN.md`, numbered before the first read)

- **D-TH-1** — node = op, and the topology is the undirected span of the topology floor's pair
  set over the ops present in the sample (not a network-element table): the deployment call
  graph stands in for the paper's physical topology.
- **D-TH-2** — the pair mask enters as `PrioriKnowledge` forbidden edges (every off-diagonal
  pair outside the mask); the paper has no mask, its only prior is the topology.
- **D-TH-3** — δ is swept as the `N` slot over {100, 1 000, 10 000} ms and `max_hop` is fixed
  at 1 (the paper hand-sets δ and chooses K by prior knowledge or BIC).
- **D-TH-4** — the runner keeps the prior's `forbidden_edges` as a set (gCastle keeps a list;
  `is_forbidden` is a membership test either way, so the output is identical and each test is
  O(1) instead of O(forbidden)).
- **D-TH-5** — the runner strips the diagonal (TTPM's always-on self-excitation) at read-out
  and asserts the output is a DAG over types; the dense type id is the event name.
- **D-TH-6** — the method environment is python 3.11 / numpy 2.4.6 / pandas 3.1.0 with torch
  CPU installed only to satisfy `castle.algorithms`' import chain; gCastle 1.0.4 is installed
  from the pinned git commit with `--no-build-isolation` (its setup.py imports `castle`).
- **D-TH-7** — the lab's patch `0001-decomposable-score` on `ttpm.py` (owner decision
  2026-11-04): the hill-climb computes the same per-type likelihoods on the same arrays, sums
  them in the same order, applies the same penalty expression and accepts the same move per
  sweep as the shipped search, but caches per-type values by parent set, fits at most the two
  columns a candidate changes, re-evaluates across sweeps only the candidates the last move
  touched, and never fits a pair the prior fixes or a cyclic candidate; the final graph is
  refitted by the shipped `_em` and must score identically (the search raises otherwise).
  Output-identical by construction and verified (§9, RUN.md D-TH-7): bit-identical sweep
  scores, moves and final graphs against the unpatched clone. Registered as `hash_extras`;
  `method_sha256` hashes the patched bytes.
- **D-TH-8** — `max_iter = 5000` as a safety cap in place of gCastle's default 20, with
  `n_sweeps_run` and `converged` recorded: the search runs to convergence as the paper's
  Algorithm 1 does; the shipped default would bound the arm at 20 edges (§9).

## 9. Provenance (not results)

- **The paper's partial-topology curve** (§V-A-4, Fig. 5, p.486–487 of `thps-cai-2024`):
  deleting topological edges at random costs precision almost linearly while recall stays at
  1.0 until about 80 % of the topology is gone — "missing topological edges would introduce
  unobservable confounders leading to redundant causal edges, which will decrease the
  precision". The hard mask of §2 removes the redundant-edge failure mode on the method's side
  (a spurious pair cannot be added at all), so the arm measures the method's precision *inside*
  the mask; that is the reading of H-thp-topology-request.
- **The paper's real-data result** (Table III, p.488): 18 alarm types, F1 0.510 / precision
  0.725 / recall 0.39, with the ablation that removes the topology trading precision for recall
  — topology as the precision mechanism, which is what the mask already supplies here.
- **Smoke test, 2026-11-04** (`algos/thps-cai-2024/RUN.md`): the runner on a synthetic 4-op
  chain (8 types, 3 901 events, 9 allowed pairs, `decay_ms` 100) under the `thp` environment:
  9 edges, acyclic, 1.8 s, the sidecar complete.
- **Timing probe, 2026-11-04** (`algos/thps-cai-2024/RUN.md` §5, not a landed read): the
  validation read on the real xs seed-0 corpus, request grain, `decay_ms` 1 000 (77 types,
  5 699 events, 551 allowed pairs). Shipped code: ≈ 10 min per hill-climb sweep (5 929 EM fits),
  stopped at the 30-minute cap inside sweep 3; runner peak footprint 330 MB. gCastle's default
  `max_iter = 20` on a skip-only scratch patch: 20 edges, precision 0.80, recall 0.035, F1
  0.068 — the cap, not the method. With `0001-decomposable-score` and `max_iter = 5000` the
  three settings converge by themselves in 46–143 sweeps and 3–6 s of fit (runner peak
  footprint 312 MB): `decay_ms` 100 → 142 edges, P 0.775 / R 0.243 / F1 0.370; 1 000 → 106
  edges, P 0.783 / R 0.183 / F1 0.297; 10 000 → 45 edges, P 0.800 / R 0.080 / F1 0.145 (the
  topology floor ≈ 0.82). Provenance for the grid (the faster kernel keeps more pairs), not a
  result: the validation read picks among them blind.
- **The cost patch (`0001-decomposable-score`, applied; D-TH-7).** Its first part, `ttpm.py`
  l.394–398, `_one_step_change_iterator`:
  yield only pairs that are off the diagonal and neither forbidden nor required, so the
  hill-climb refits the EM once per *open* pair per sweep instead of once per ordered pair
  (l.234–236 refit the unchanged matrix for every fixed pair; the EM is deterministic, so that
  candidate never satisfies `new_l > l_ret` and the output is unchanged by construction). A
  second, larger patch would make the score decomposable (cache each type's `li` and refit only
  the column a move touches — `_em` already fits each type independently, l.335–380 — and
  re-evaluate across sweeps only the candidates the accepted move touched), cutting the
  per-candidate cost from N type-EMs to one or two and a sweep to ≈ 2N candidates; RUN.md §6
  sketches it with the equivalence argument and holds the verification (bit-identical sweep
  scores, moves and final graphs against the unpatched clone on the xs events and on a
  `THPSimulation` sample; `scripts/envs/thp/verify-equivalence.py` is the check to repeat on a
  replica before its first landed read).
