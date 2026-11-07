# External-arm plan: CausalNET with the deployment topology (pre-registered 2026-11-04, before any read)

Committed under the external family's addendum of 2026-11-04 (`plans/cascade.md`, D-SB-29) before
any read of the method. The member is **CausalNET** (Zhu, Huang, Yin, Fan, Jin, Liu, IJCAI 2024;
corpus slug `causalnet-zhu-2024`; public code `github.com/CGCL-codes/CausalNET`, commit `aaccce6`),
a Transformer next-event predictor whose attention is gated by a trainable edge-probability matrix
over the event types and by a per-hop **causal decay matrix** over the topological network the
events live on — the topology enters as a *soft* prior (a hop gate in the attention and, here, the
edge-logit initialisation), never as a hard mask on the output. Owner decisions (publication PRD
Decision Log, 2026-11-04): the arm enters the CLeaR 2027 submission with **xs and s at five seeds
each**; **m is reported where it lands by the Phase A gate (Monday 10 November)** and the paper
says so where it does not; no unmasked configuration is run (`cascade/none` at xs stands as the
family's unmasked precedent). Every number the paper quotes for this arm comes from a read
governed by this file; the lab's timing probes of 2026-11-04 (§9) are provenance, not results.
Changes are dated addenda at the end, never edits in place. Deviation ids: the harness's D-SB-n
(RUN.md here) and the method's D-CN-n (`algos/causalnet-zhu-2024/RUN.md`).

## 1. Arms

| arm | class | reads (declared inputs) | statistic | paths | cuts | sample |
|---|---|---|---|---|---|---|
| `causalnet/topology` | `external` | **method side:** the views only, through `Corpus` (`views/end-<grain>/sequences/split=<val\|test>`, the event columns of `plans/cascade.md` §2); **score side:** the topology floor's `(callee token, caller token)` pair set through `floorprior.topology_read` — the same reader as `floor/topology` and `cascade/topology` — handed over as dense type ids together with the undirected op-level adjacency those pairs span (`op_topology`; node = op). A declared **topology-informed** arm. The prior is **soft**: the op adjacency is the method's topological network `G_N` (its hop gate), and the pair set initialises the edge logits (§2); the method may return any pair. | `causalnet_edge_score`: the method's unpruned read-out `final_prob[i, j] ∈ [0, 1]` (row = cause; §2) | `none` | `frozen` | the arms' (`plans/caps.md`: whole split at xs; 10k / 5k validation, 20k / 10k test from s; `head`; `--max-len 64`, D-SB-7) |

No backbone, no probe pass, no `c` or `g` axis (both recorded `0`); the `N` slot carries the one
swept setting, ξ = `window_ms` (§3, D-SB-28). No bidirected edge is ever predicted
(`structural_limitation`). Model hash `none` (D-SB-16); the method's identity is `method_sha256`
over the clone's `model/*.py`, `model/*/*.py`, `model/configs/*.yaml`, the pinned commit, the patch
names `0001-collapse-decay` and `0002-drivable-entry`, and the runner's bytes. **`mask_mode =
count`**: an edge outside the topology's pair set is kept, scored and counted
(`n_edges_outside_mask`); `hard_output = false`: τ is a real selection on `floor_taus`.

## 2. The method and its inputs

- **§2 the code.** `CGCL-codes/CausalNET` at `aaccce6dbd25eb4baa3711c250405aa1d8d6e7b1` (MIT)
  with the lab's two patches (`scripts/envs/causalnet/`, applied uncommitted to the clone's working
  tree as CASCADE's patch is; the full list in `algos/causalnet-zhu-2024/RUN.md`):
  - `0001-collapse-decay` (D-CN-1). The shipped causal decay matrix is
    `θ ∈ R^{(K+1) × |N| × |N| × |V| × |V|}` (hop × node pair × type pair). Here every event type
    lives on exactly one node (type = `<op>:<outcome>`, node = op), so `θ[k, u, v, i, j]` is only
    ever read, and only ever receives gradient, at `(u, v) = (node(i), node(j))`; the patch stores
    `θ̃ ∈ R^{(K+1) × |V| × |V|}` and the type-level hop adjacency `Ã_k[i, j] = A_k[node(i),
    node(j)]`, and the forward pass, the losses, the gradients and the read-out are those of the
    full tensor. **Equality check** (3 ops, |N| = |V| = 3, one type per node, topology 0–1–2,
    seed 42, CPU): over 40 stage-1 and stage-2 batches of two epochs the maximum absolute
    difference of every loss term between the full and the collapsed tensor is **0.0**, the
    read-out matrices differ by **0.0**, the trained slots of the full tensor equal the collapsed
    tensor (max |Δ| 0.0) and its untrained slots stay at 0 (on MPS: loss 1.2e-4, read-out 0.0,
    kernel rounding). At m this is the difference between 2 × 1 498² ≈ 4.5 M decay parameters
    and 2 × 541² × 1 498² ≈ 1.3 × 10¹² (unallocatable).
  - `0002-drivable-entry` (D-CN-2 … D-CN-8): the project and dataset directories from the config
    or flags instead of `/home/user_name/…`; the temporal-encoding vector as a module buffer
    (shipped on cuda unconditionally); tensorboard imported on demand (absent from the env);
    batches generated per epoch instead of 200 epochs of batches cached to disk (the same
    generator and the same numpy stream; ~70 GB at s otherwise); the time scale as a parameter
    (§2 time unit); a `graph_init: mask` option (§2 init); the shipped per-sequence `.item()`
    loops and the per-type `type_mask` loop vectorised (checked equal); the two DAG prunes computed
    only when displayed or scored (the saved read-out is unpruned either way); the window shift
    done in integers before the float32 cast (millisecond timestamps exceed float32's exact
    range); a stable sort on tied timestamps. `gcastle` (in the clone's `requirements.txt`) is
    imported by no module under `model/` and is not installed (D-CN-6: torch 2.x instead of the
    pinned 1.13.1).
  - The runner (`seq2causebench.causalnet_runner`, under the `causalnet` interpreter) writes the
    clone's dataset format from the harness's events (`alarm.csv`: `alarm_id` = type, `device_id`
    = node, `start_timestamp` = `end_timestamp` = the millisecond end time; `topology.npy` = the
    op adjacency; `causal_prior.npy` = the pair set), loads it through the clone's own
    `get_formated_dataset`, trains in-process and returns `final_prob`.
- **§2 events.** The sampled sequences of the split as `plans/cascade.md` §2: one event per
  token, type `<op>:<outcome>`, time = the span's **end** in integer milliseconds relative to the
  earliest sampled event, sorted (ties in row order), types dense over the sample. The method sees
  **one sequence** — the whole sampled timeline — as its own alarm data does; each training target
  is one event, its history the events within ξ before it (at most `max_seq_len = 45`, the shipped
  cap; events at the *same* millisecond as the target are excluded from its history by the shipped
  generator). Node = op (`events.npz["node"]`), `G_N` = `op_topology` (symmetric, int8).
- **§2 the time unit (D-CN-3).** The shipped data is in seconds and `max_time_lag = 120`; the
  generator then divides the window-shifted times by 60 before they reach the temporal encoding,
  the time-prediction loss and the intensity's `Δt / (t + 1)` term, so the shipped 120 s window
  spans 2 encoded units. Here `max_time_lag = ξ` in **milliseconds** (the setting) and the divisor
  is `time_scale = ξ / 2`, so a ξ-window spans the same 2 encoded units at every setting; the
  window test itself runs on exact integer milliseconds.
- **§2 the prior.** `causal_prior.npy[i, j] = 1` for every allowed `(cause, effect)` pair of the
  topology floor (dense ids over the sample; diagonal 0), 0 elsewhere. **Edge-logit init
  (D-CN-4):** `ϑ = 0` on the allowed pairs (the paper's zero init, ZI, its best) and `ϑ = −1` on
  every other pair (the lower end of its negative init, NI, its other best) — a soft edge init,
  never clamped afterwards (the shipped read-out's prior clamp is commented out in the code and
  stays off). `G_N` enters the attention as shipped: `max_hop = 1`, so the decay gate admits a
  history event only from the target's own op or an adjacent op (`A_1` = the adjacency with its
  diagonal forced to 1), and `k = 0` admits the same op only.
- **§2 read-out (D-CN-5).** The runner returns exactly what the shipped training loop saves as
  `dags/final_prob/<dataset>.npy` at the last epoch: `final_prob[i, j] = p_ij · max_{k, u, v :
  A_k[u, v] ≠ 0} σ(θ[k, u, v, i, j])` with `p = σ(ϑ)` (the paper's Eq. 16 before the threshold ε;
  unpruned — the paper's headline `CausalNET` variant; **no rollback prune**, the pruned
  `CausalNET*` is worse on every reported cell). Both factors lie in (0, 1), so the matrix is in
  [0, 1] without normalisation; the diagonal is zeroed. One property of that quantity is recorded
  here because it shapes the τ sweep: the maximum runs over *every* `(u, v)` slot with
  `A_k[u, v] ≠ 0`, and the slots other than `(node(i), node(j))` are never trained (they stay at
  their init 0, `σ(0) = 0.5`), so `final_prob[i, j] ≥ 0.5 · p_ij` for every pair, including pairs
  whose ops are not adjacent — the topology gates the attention, not the read-out. A `readout =
  gated` method-arg (the maximum over the trained slots only, `p_ij · max_k Ã_k[i, j] σ(θ̃_k[i,
  j])`) exists in the runner and is **not** used under this plan.
- **§2 fixed settings.** `config_m24.yaml` as shipped except the items above: `d_model 512`,
  `d_inner 1024`, 4 layers, 4 heads, `d_k = d_v = 512`, dropout 0.1, label smoothing 0.1, Adam
  `1e-4 → 1e-5` (predictor), `4e-2 → 4e-3` (graph and decay), weight decay 1e-3, Gumbel τ `1 →
  0.1`, loss weights `λ_l 10, λ_a1 1, λ_a2 1, λ_s 0.1, λ_d 0.1`, `batch_size 512`, `max_seq_len
  45`, `max_hop 1`, seed 42 (the shipped config; the harness `--seed` is the sample's). The
  schedules anneal over the registered epoch count, so a shorter run completes its annealing.
- **§2 device.** `device = auto`: cuda, else mps, else cpu (D-CN-7); the device is recorded in
  every fit's facts. The arithmetic differs across backends at kernel-rounding level (§2 equality
  check); seeds pin the sample, the init and the Gumbel/Bernoulli draws, not the backend.

## 3. Grids (validation sweep, blind)

- **§3 settings.** ξ = `window_ms` ∈ {1 000, 10 000, 60 000} (1 s, 10 s, 60 s), carried as `N`;
  one fit per setting, grain and read. The paper's own ξ ablation (30 s … 300 s on alarm data,
  120 s best) is the precedent for sweeping the window; the trace-bench floors' windows of
  `plans/cascade.md` §3 are reused so the three soft-prior and hard-prior methods share one grid.
- **§3 method-args (fixed, recorded verbatim in every record).** `device=auto seed=42 max_hop=1
  readout=shipped` and the epoch cap **`epochs=200` at xs** (the shipped `total_epoch`; 11 batches
  per epoch on the xs validation sample, 2 200 alternating stage pairs) and **`epochs=50` at s and
  m** (93–160 batches per epoch there, 4 650–8 000 stage pairs — more optimiser steps than the xs
  run, within the s/m wall-clock budget of §4; the paper's own regime is 126 × 200 ≈ 25 000).
  `time_scale` and `graph_init` keep their defaults (`auto` = ξ / 2; `mask`); `batch_size 512`.
- **§3 τ.** The floors' grid `floor_taus` (`plans/floors.md` §3, 20 values `0 … 0.99`), strict
  (`score > τ`), a real selection: the read-out is graded, and `≥ 0.5 · p` everywhere (§2), so
  low τ keeps nearly every pair and the sweep's argmax is where the arm's precision is bought.
- **§3 freeze rule.** Per cell `causalnet/topology/none/frozen/<grain>`: the argmax of validation
  directed F1 at the default floor over `(N, τ)`; ties → smaller `N`, then larger τ (the harness's
  rule, `freeze.select_cells`). A frozen `N` at either end of the grid is recorded, not refused.
- **§3 floor.** The scorer's default floor (0.05); the coverage rule is reported, never gated.

## 4. Protocol

Per corpus and grain, the family's protocol (`plans/cascade.md` §4 and its addendum): `externalarm
--method causalnet --arms causalnet/topology --split val --settings 1000 10000 60000 --method-args
…` (three fits per grain) → `scoresweep --arms causalnet/topology --floor-taus …` → `freeze
--pretrain-results none` (`freezes/<date>-<rung>-<variant>-s<k>-external-causalnet.json`) → one
committed freeze → `externalarm --split test --freeze … --cells <one cell>=<τ> --settings <the frozen
N>` (one arm, one fit) → `annotate --pretrain-results none --per-lag-files 0` → landing under
`results/<rung>/latent/seed=<k>/causalnet/topology/none/frozen/<grain>/` with the read's records
under `test-read/causalnet-topology-<grain>/` and `causalnet-val-sweep/`, ledger stage `causalnet`
(`results_sha256`, `model_sha256 = "none"`, `method_sha256`, `method_commit`, `read`, `cells`),
and the `report` rows and pairs of §6. **xs and s at five seeds each, required; m where it lands
by Monday 10 November** (owner decision 2026-11-04). Staging as the floors' (`smallest-rung` at
xs; above xs the arms' previous-rung ledger entry of the same grain). Every read records
`ordering = end`, the sample, the settings, the method-args, `method_sha256`, `method_commit`,
the interpreter, `method_env`, the device and the wall clock of every fit and every epoch.
Corpora from the published release only (`chadyuk/trace-bench`, `v0.3.0`); the score tier
carries the topology.

**Hosts.** xs runs locally (MPS, ~33 s per epoch on the xs validation sample: 200 epochs ≈ 1.9 h
per fit, six fits per seed for the validation reads plus two test fits; §9). **s and m run on a
`g5.2xlarge` replica (A10G 24 GB, owner-applied; `scripts/envs/causalnet-env.sh` builds the
method environment on the box), one replica per seed; CPU hosts if the GPU path fails** (the
method has a CPU path, §9 gives its cost). The s seed-0 validation read is timed first (both
grains, three settings) and the per-epoch wall clock sets whether m is attempted.

**Timing gate (owner's rule, stated in advance).** If the s reads cannot land five seeds by the
Phase A gate, the arm is absent from the submission and the fact is recorded here as an addendum
and in the paper's reproducibility appendix; m lands or is reported absent; nothing landed moves.

## 5. Caveats carried beside every number

- **Soft prior, counted.** The topology never masks the output: every pair is scored (`≥ 0.5 ·
  p`, §2) and the session-grain journey and retry edges, outside the topology, are reachable —
  whether the method finds them is measured (`n_edges_outside_mask` in every record, H-causalnet-
  outside-mask in §7). The arm's recall is not bounded by the floor's as the hard-mask arms' is.
- **|V|² edge logits and the NOTEARS penalty.** `ϑ` and `θ̃` are dense over the type pairs
  (s: 282–308 types → 80–95 k logits; m: 1 498–1 702 realised types in the sample → 2.2–2.9 M)
  and the DAG-ness term `tr e^{P∘P} − |V|` costs a |V|×|V| matrix exponential per stage-2 step,
  O(|V|³): negligible at s, ~4 × 10⁹ multiply-adds per exponential at m — within the per-step
  budget on an A10G, dominant on a CPU host. Memory per step at m: the one-hot `(bs, |V|, L)`
  tensors of the attention mask and the likelihood, ~ 512 × 1 702 × 46 floats ≈ 160 MB each.
- **The DAG penalty against a cyclic truth.** The method's `L_graph` pushes `σ(ϑ)` toward a DAG
  while the trace-bench truth graphs are cyclic by design (retries, mutual calls); the unpruned
  read-out can still carry cycles (the paper's own headline variant does), but the penalty
  weight `λ_d = 0.1` is a pressure against them.
- **No robustness study to a wrong topology.** The paper assumes a known, correct `G_N` and
  reports none; the trace-bench topology is the deployment's call graph, complete by
  construction at request grain and incomplete for the session-grain journey edges.
- **Scale of the published runs.** The paper trains for 2–6 h on a V100 for 64 k alarm events
  (200 epochs, batch 512); the s and m samples here hold 48 k–82 k events and the epoch cap of
  §3 is a budget decision, recorded as such.
- **Sample-bound, sample-paired.** As `plans/cascade.md` §5: paired with the arms and the
  bigram floor, not with the topology floor; `plans/caps.md`'s sample note travels with every s
  and m number.
- **Causal sufficiency / excitation only.** Granger-style predictive causality, no hidden
  confounding, no inhibition; bidirected truth edges are never predicted.
- **Five pairs: low power** (`report`'s own note).

## 6. Report rows and pairs

The external cells are **not** in the required set. Pairs added to the report of every rung the
arm ran on (frozen cut, both grains; a pair shares corpus and seed, not a model — D-SB-15):
`causalnet/topology/none/frozen : floor/topology/none/frozen`, `: floor/bigram/none/frozen`,
`: trace/core/shipped/frozen`, `: baseline/granger/shipped/frozen`; and, where the sibling arms
landed on the rung, `: cascade/topology/none/frozen`, `: thp/topology/none/frozen`,
`: lmt/topology/none/frozen` (reported, no hypothesis).

## 7. Hypotheses (scored pass or fail per rung in `findings/causalnet-latent.md`)

- **H-causalnet-topology-request.** At the frozen cut and the default floor,
  `causalnet/topology`'s directed F1 at the request grain is **at least 0.02 above
  `floor/topology`'s** on every rung (five-seed mean of the paired difference, same corpus and
  seed, p < 0.05). *Fail:* any rung where the mean gain is below 0.02 or p ≥ 0.05.
- **H-causalnet-topology-session.** The same at the session grain. *Fail:* as above.
- **H-causalnet-vs-core.** `causalnet/topology`'s directed F1 is **above
  `trace/core/shipped/frozen`'s** at both grains on every rung (paired, p < 0.05). *Fail:* any
  rung and grain where it is not.
- **H-causalnet-outside-mask (reported, no threshold).** At the frozen cut, the number and the
  precision of the arm's predicted edges outside the topology's pair set per grain and rung
  (`n_edges_outside_mask` over the predicted set, and their share of the session-grain journey and
  retry truth edges): does the soft prior let the method find what the hard-mask arms cannot?

## 8. Deviations (RUN.md rows, numbered before the first run)

- **D-SB-29** (the addendum's row, shared by the family) — one freeze per external method, the
  per-method registry, `mask_mode`, `method_commit`, `method_env`.
- **D-CN-1 … D-CN-8** — the method's own rows (`algos/causalnet-zhu-2024/RUN.md`): the collapsed
  decay tensor and its equality check (1); the drivable entry point (2); the millisecond time
  unit and `time_scale = ξ / 2` (3); the mask edge init (4); the unpruned `final_prob` read-out
  and its `0.5 · p` floor (5); torch 2.x in place of the pinned 1.13.1, no gcastle (6); the device
  rule (7); the per-epoch batch streaming, the vectorised loops, the integer time shift, the stable
  sort and the lazy tensorboard import (8).

## 9. Provenance (not results)

- The paper's initialisation ablation (§6, "Influence of Causal Graph Initialization"): zero and
  negative inits are best, random worse, and the **positive init "notably inferior … an
  excessively strong prior: an excessive number of edges … initialized as true, whereas
  real-world causal graphs are usually sparse"** — the reason the mask enters as `0 / −1`
  (allowed / other) and not as `+1` on the allowed pairs.
- The paper's ξ ablation: 120 s best on both alarm datasets, 30 s worst; its K ablation: `k = 0`
  (no topology) worst on both — the topology carries signal in its domain.
- **Timing probe, 2026-11-04 (local, not a landed read; `out/causalnet-probe-xs-s0-26-11-04`,
  gitignored).** The validation read on the real xs seed-0 corpus, request grain, ξ = 10 000,
  `epochs=5`, device `mps` (Apple M3 Pro, 18 GB): 5 699 events, 77 types, 20 ops, 551 allowed
  pairs, 11 batches per epoch; **per-epoch wall 29.8–37.4 s (mean 33.4 s)**, fit wall 172 s; peak
  memory footprint of the harness process 169 MB (the method process is measured separately in
  `algos/causalnet-zhu-2024/RUN.md`). Scoresweep of that 5-epoch fit (request grain, `floor_taus
  0 0.1 0.3 0.5 0.7 0.9`): directed P / R / F1 = 0.557 / 0.881 / **0.683 at τ = 0.1** (0.753 /
  0.128 / 0.219 at τ = 0.3; 0.122 / 0.881 / 0.214 at τ = 0); 5 301 of the 5 852 returned
  positive entries lie outside the mask (the `0.5 · p` floor of §2 at `p = σ(−1) = 0.27`). The
  topology floor's xs request F1 is ≈ 0.82. A 200-epoch xs fit projects to ~1.9 h on MPS and was
  not run under the ~40 min probe budget. A **60-epoch probe** of the same read (64 min on a
  shared host; the method's RUN.md): stage-1 type-prediction accuracy rose to 0.86 while the
  graph read-out did not improve — directed P / R / F1 0.616 / 0.611 / 0.614 at τ = 0.1 (best;
  0.671 / 0.113 / 0.193 at τ = 0.9), the DAG-ness term 41.6 → 146.4 and the sparsity term 0.204 →
  0.135. One setting, one seed, one grain: a warning that the shipped loss weights may not move
  the edge logits toward the trace-bench truth, not a result; the plan's grid and fixed
  settings stand as pre-registered, and the CPU path measured ~70 s per stage step at xs (RUN.md).
- **Extrapolation (from the xs probe and the code's complexity; not a measurement).** The cost per
  epoch is ~ (batches per epoch) × 2 stages × (one 512 × 46-token pass through the 21 M-parameter
  predictor, ~3 × 10¹² flop forward + backward) plus, in stage 2, one |V|×|V| matrix exponential.
  MPS runs ~1.5 s per stage step; an A10G at ~0.25 s per step (fp32, ~12 TFLOPS sustained)
  gives ≈ 6 s per epoch at xs (200 epochs ≈ 20 min), ≈ 47 s (request, 93 batches) and ≈ 80 s
  (session, 160 batches) per epoch at s (50 epochs ≈ 40 min and 67 min per fit; 2.6 h and 4.4 h
  at 200), and ≈ 50 s / 85 s per epoch at m (1 498 / 1 702 realised types; the matrix
  exponential and the one-hot tensors add < 10 %). Per seed at 50 epochs: s ≈ 3 × (40 + 67) min
  validation + 2 test fits ≈ 7 h of A10G; m ≈ 7.5 h. CPU hosts: §9's CPU measurement in the
  method's RUN.md.

## Addendum 2026-11-06 — `batch_size = 64` at m (D-CN-9)

Dated before any m read of the method lands; xs and s stand as read at the registered `batch_size
512` (§3). Owner decision (2026-11-06): **at m the fixed method-args of §3 are `epochs=50 device=auto
seed=42 max_hop=1 readout=shipped batch_size=64`**; every other knob keeps §3's value. Motivation
(RUN.md rows of 2026-11-05 and 2026-11-06): the shipped stage-2 sampler (`sigmoid_gumbel_sample` →
`gumbel_softmax`) materialises the sampled graph as a `batch × |V| × |V| × 2` float32 tensor and holds
about four and a half such copies at its peak, while the predictor reads one column of it per batch
element. At m the validation samples realise 1 498–1 597 types (request) and 1 702–1 803 (session), so
one copy is 9.45 GiB at 512 × 1 574² and the peak exceeds 52 GiB: the first m fit ran out of GPU
memory on the 24 GB class (stage 1, first backward) and on the 48 GB class (stage 2, first batch),
and no single-GPU EC2 class is larger. Memory scales as `batch × |V|²`; at 64 the projected peak is
≈ 8.6 GiB at 1 803 types, 2.6× under the 24 GB class (`plans/caps.md`'s headroom rule), so **m runs
on the `g5.2xlarge` class of §4**, cap 72 h.

What changes and what does not. The batch size enters the data generator (events per batch) and
the sampler only; the losses, the read-out, the τ grid, the `N` grid, the freeze rule and the
sample are §3's. The optimiser takes 8× the steps per epoch (m request ≈ 730–830 stage-2 steps per
epoch, session ≈ 1 240–1 440, against 91–103 and 155–179 at 512) under the same `epochs=50`, so the m
fits see ≈ 36 000–72 000 stage pairs where s saw 4 650–8 000 — a disclosed departure from the
xs/s regime the paper reports beside every m number (§5), not re-tuned. The m test reads use the
same args (the freeze binds them). The wall clock is unmeasured at 64; §9's extrapolation does not
cover it, and the first synced m fit's per-epoch wall clock sizes the m test caps. D-CN-9 in the
method's RUN.md records the deviation.

## Addendum 2026-11-07 — the improved configuration: `init_off` (D-CN-10) and `readout=gated` (D-CN-11); one verification run first

Dated before any m read of the method lands (nothing is frozen at m). Owner decision (2026-11-07):
register two improvements to the arm's stability and its target F1, verify them on ONE replica and
ONE seed (m, seed 0, the validation job) before any further seed, and decide the arm's final
configuration from that run. Every new knob defaults to the behaviour xs and s were read with, so
the landed records stay reproducible from their recorded `method_args`; the patch and runner edits
change `method_sha256` for every later read (the xs/s records keep the old hash — expected, said so
in their rows).

- **D-CN-10 — `init_off`, the off-mask logit at the mask init (D-CN-4), as a method-arg; default
  `−1` = the landed value; `−4` in the improved configuration.** Mechanism: the acyclicity penalty
  `h = tr(exp(A∘A)) − d` (float32) at init has largest eigenvalue ≈ σ(init_off)² · |V| plus a small
  on-mask part; the b64 runs of 2026-11-06 died on it (RUN.md row). Pre-flight on the ten m
  validation masks (every seed, both grains; the masks the runner writes, in float64): at `−1`
  λ_max = 109.8–132.1 (float32 overflows past 88.7), at `−2` 23.1–28.0, **at `−4` 1.62–2.11**
  — no overflow and a penalty that no longer dwarfs the data loss at the start. The knob is
  `graph_init_off` in the clone's config (patch `0003-init-off`, applied by `causalnet-env.sh`
  after 0002), `init_off` in the runner (`DEFAULTS`, recorded in `method_args` and in every fit's
  facts). A disclosed tuning step of the same kind as LMT's `a_init` (`plans/lmt.md` addendum
  2026-11-03, D-LM-5): the start is sparser than the paper's init; the paper reports it beside
  every number read under it.
- **D-CN-11 — `readout=gated` in the improved configuration (the knob of §2, there "not used under
  this plan"; this addendum lifts that for the improved configuration only).** `p_ij · max_k Ã_k[i, j]
  σ(θ̃_k[i, j])`, the paper's Eq. 16 over the trained slots: pairs whose ops are not adjacent in the
  undirected topology score exactly 0, the `≥ 0.5 · p` floor of §2 disappears and the in-mask
  scores become a graded product rather than a staircase. Motivation (s, landed): the arm's test F1
  0.765 / 0.538 against the floor's 0.839 / 0.740 is lost in precision (0.65 vs 0.72 at request); at
  s seed 2 request 89 701 of the 92 720 scored pairs lie outside the topology's 3 019; the 20-τ sweep
  yields 7 distinct F1 values and the frozen τ 0.15 sits on a step that three seeds' session test
  fits land on the other side of (std 0.16). Under `gated` `n_edges_outside_mask` is 0 by
  construction and H-causalnet-outside-mask (§7) is moot for that configuration; `mask_mode` stays
  `count` (no code change); recall becomes bounded by the undirected adjacency span, as the
  hard-mask arms'. **Open for the owner after the verification run:** whether the improved
  configuration is m-only (the paper then reports two configurations of one arm) or whether xs and s
  are re-read under the same args for one comparable method (recommended if adopted), and whether
  `gated` replaces `shipped` in the §6 pairs or sits beside it.
- **Not registered:** the penalty in float64 (a numerics change to the method, unnecessary once
  λ_max is ≈ 2), `nan_to_num`, gradient clipping (the loss itself is infinite), or another
  acyclicity characterisation (method redesigns, for the authors).
- **The verification run.** m, seed 0, validation job, `g5.2xlarge`, cap 72 h (measured chain at
  batch 64 ≈ 23 h → 3×), args **`epochs=50 device=auto seed=42 max_hop=1 readout=gated batch_size=64
  init_off=-4`** (§3 + addendum 2026-11-06 + this one); run name `…-m-causalnet-v2-s0-val`.
  Acceptance read on sync: (a) every epoch summary finite on all six fits; (b) per-epoch wall clock
  and batches per epoch (the caps of the remaining seeds and of the test job); (c) argmax validation
  directed F1 at the default floor per grain against the m seed-0 topology floor's validation F1;
  (d) the number of distinct F1 values over the 20 τ (s had 7); (e) `n_edges_outside_mask` (0 under
  `gated`), `n_nonzero`, `n_edges_returned`. **No freeze from the verification run** until the owner
  declares the configuration final (a committed freeze binds the test read and the family's
  one-freeze-per-method rule, D-SB-29).
