# External-arm plan: LMT with the deployment topology as a soft prior (pre-registered 2026-11-03, before the first read)

The third member of the `external` arm family of `plans/cascade.md` (its addendum of 2026-11-04,
D-SB-29): **LMT** (Xiao, Chen, Sun, Shi, Yue, *LMT: A Bayesian Framework for Causal Discovery from
Textual Alarm Records in Manufacturing Systems*, arXiv:2606.09892, 2026; corpus slug
`lmt-xiao-2026`; public code `github.com/xx987/LMT`, commit `6f4143dc`, no licence file), the
MAP learner that fits a multivariate exponential-kernel Hawkes process to the timestamps under a
Bernoulli log-prior over the adjacency and a NOTEARS acyclicity penalty — run here with the
shipped deployment topology entering as that **soft prior** (a q-matrix), never as a hard mask.
Owner decisions (publication PRD Decision Log, 2026-11-04): a method of the addendum enters the
CLeaR 2027 submission with xs and s at five seeds each; m is reported where it lands by the Phase A
gate (Monday 10 November) and the paper says so where it does not; no unmasked configuration.
Every number the paper quotes for this arm comes from a read governed by this file; the lab's
timing probes of 2026-11-03 (§9) are provenance, not results. Changes are dated addenda at the
end, never edits in place. Deviation ids: the harness's D-SB-n and the method's D-LM-n
(`algos/lmt-xiao-2026/RUN.md`).

## 1. Arms

| arm | class | reads (declared inputs) | statistic | paths | cuts | sample |
|---|---|---|---|---|---|---|
| `lmt/topology` | `external` | **method side:** the views only, through `Corpus` (`views/end-<grain>/sequences/split=<val\|test>`, columns `trace_id, ops, outcomes, n_spans, start_ns, offsets, durations`); **score side:** the topology floor's `(callee token, caller token)` pair set through `floorprior.topology_read` — the same reader and pairs as `floor/topology` and `cascade/topology` — handed to the method as the **q-matrix of its Bernoulli prior** (§2), not as a constraint. A declared **topology-informed** arm. | `lmt_edge_prob`: the fitted relaxed adjacency `A_ij = σ(A_raw)_ij` of the cause → effect pair, in (0, 1), every off-diagonal pair scored | `none` | `frozen` | the arms' (`plans/caps.md`: whole split at xs; 10k / 5k validation, 20k / 10k test from s; `head`; `--max-len 64`, D-SB-7) |

No backbone, no probe pass, no `c` or `g` axis (records carry `0`). The `N` slot carries the one
swept setting, `prior_weight_milli` = 1000 × the weight of the log-prior term (§3, D-SB-28). The
mask mode is **`count`**: an edge outside the topology's pairs is kept, scored and counted
(`n_edges_outside_mask`), never refused; because every off-diagonal entry of a sigmoid is
positive, that count at score > 0 is `K(K − 1) − |mask|` less the within-op pairs by construction,
and the informative count is the one at the frozen τ (§7, H-lmt-outside-mask). The output is
**soft** (`hard_output = False`): the τ sweep is a real selection. No bidirected edge is ever
predicted (`structural_limitation`, as every arm). `model_sha256 = "none"`; the method's identity
is `method_sha256` (sha256 over `chemical_1000_event_repro/temporal_hawkes_pure_map.py`,
`chemical_1000_event_repro/text_causal/*.py`, the commit and the runner's bytes) with
`method_commit` beside it.

## 2. The method and its inputs

- **§2 the code.** LMT at commit `6f4143dc0668187f69505bc206e87ae2084d47d3` (HEAD of `main`,
  cloned 2026-11-03 into `algos/lmt-xiao-2026/LMT`). The trainer is
  `chemical_1000_event_repro/temporal_hawkes_pure_map.py` (`PureTemporalMultivariateHawkes`):
  `A = σ(A_raw) ⊙ (1 − I)`, `α = softplus(·) ⊙ (1 − I) + 1e-8`, `β = softplus(·) ⊙ (1 − I) +
  1e-4`, `μ = softplus(·) + 1e-5`; objective `J = L_time + w · log p(A | q) − β_dag · h(A) −
  λ_e Σ_{i≠j} A_ij`, `loss = −J`, Adam; `log μ` initialised from `N / (K · T)` then Gaussian noise
  of `init_std` on every raw parameter (A_raw, log α, log β, log μ, in that order, under the seed);
  prior and penalties from `chemical_1000_event_repro/text_causal/graph_prior.py`
  (`semantic_log_bernoulli` = Σ_{i≠j} A_ij log q_ij + (1 − A_ij) log(1 − q_ij); `dag_penalty_h` =
  tr exp(A ⊙ A) − K; `sparsity_sum`). The repository carries **no licence**: the clone is imported
  under `--method-src` by the runner (`seq2causebench.lmt_runner`, driven by the `lmt` conda
  environment's interpreter: python 3.11, torch 2.15.0, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1;
  recipe `scripts/envs/lmt-env.sh`) and is never vendored into the harness.
- **§2 the likelihood is reimplemented (D-LM-1).** The shipped `forward` is a pure-Python double
  loop over events — O(N²) scalar torch operations — that cannot run on the harness's samples
  (5.7k events at xs, up to 640k at s). Under the charter's missing / broken-code policy and with
  the owner's approval, the runner evaluates the **same objective with the same parametrisation**
  vectorised: per cause type a, `S_{a, z_n}(t_n) = Σ_{m: z_m = a, t_m < t_n} exp(−β_{a, z_n}(t_n −
  t_m))` through `logcumsumexp` over the type-a times (one column per effect type), read at the
  number of type-a events strictly before `t_n` (ties contribute nothing, as the shipped `dt <= 0:
  continue`); the compensator in the shipped closed form. **Equality check (required, recorded in
  `algos/lmt-xiao-2026/RUN.md` D-LM-1):** on the clone's 50-event k = 5 sample at a random
  parameter setting (seed 0), per-event log-intensities and compensator agree with the shipped
  loop to **3.2e-14 relative in float64** (the algebraic identity), **3.1e-6 / 2.2e-6 against the
  shipped float32 loop**, and the chunked evaluation is identical. The prior, the penalty and the
  optimiser are the clone's and the shipped trainer's lines.
- **§2 time (D-LM-2).** The harness's integer-millisecond end times are converted to **seconds**
  (float64 → the compute dtype) before fitting; `T` is the last event's time, as shipped.
- **§2 the q-matrix (D-LM-3).** Clusters = event types (K = `n_types`, no node information). `q_ij
  = q_in` (0.5) on every allowed (cause, effect) pair of the topology mask, `q_ij = q_out` (0.01)
  on every other off-diagonal pair, 0.5 on the diagonal (the shipped loader's rule, masked out of
  the prior anyway); `q_default` (0.5) everywhere were there no mask (not run). Both strictly inside
  (0, 1): the prior is soft — a pair outside the topology costs `w · log(0.99 / 0.01)` ≈ 4.6 w per
  unit of A, nothing forbids it.
- **§2 precision (D-LM-4).** `h(A) = tr exp(A ⊙ A) − K` at the shipped initialisation (A ≈ 0.5
  everywhere) is of order `exp(0.25 (K − 1)) / K` per entry of its gradient; its square overflows
  Adam's second moment in float32 from K ≈ 200 (measured at K = 323: `h` stays bit-identical over
  the epochs, the adjacency never moves) and in float64 from K ≈ 1 450. The raw parameters, prior
  and penalty therefore run in **float64** (`param_dtype=float64`, the same model in double
  precision; the kernel sums in float64 for their own cancellation reason, D-LM-1). At xs (K = 77)
  float32 and float64 are both finite; at s (K = 323) only float64 moves; at m (K ≈ 2 650) neither
  does — see §4 and §5.
- **§2 events.** As `plans/cascade.md` §2: the arms' sample, the first 64 tokens of each sequence,
  type `<op>:<outcome>`, time = the span's end in milliseconds, sorted, dense type ids.
- **§2 read-out.** The runner returns the final `A` (float32, diagonal 0, row = cause); every
  off-diagonal entry is an edge at its score; within-op pairs are dropped at projection (D-SB-10);
  edges outside the mask are counted (`count`). One sweep-shaped file per setting,
  `scores-lmt-topology-none-c0-N<prior_weight_milli>-<grain>.npz`, `c = g = 0`, model hash `none`.
  The sidecar carries `final_loglike`, `final_h`, the epoch count, `n_above_half` and
  `n_above_half_outside_mask` (off-diagonal entries above 0.5, inside and outside the mask).

## 3. Grids (validation sweep, blind)

- **§3 settings.** `prior_weight_milli` ∈ {1 000, 10 000, 100 000} — the log-prior's weight w ∈
  {1, 10, 100} (the paper's λ; the shipped 1000-event trainer fixes w = 1, the 50-event one
  exposes `--lambda-prior`, run at 1.5 there), one fit each on the validation sample, carried as
  `N`.
- **§3 fixed method-args** (recorded in every record as `method_args`): the clone's
  `chemical_1000_event_repro/run_k5.sh` values — `epochs=50 lr=0.05 beta_dag=1.0 lambda_e=0.1
  init_std=0.35` — plus `seed=0 device=cpu q_in=0.5 q_out=0.01` and the precision knobs of
  D-LM-1 / D-LM-4, `dtype=float64 param_dtype=float64 chunk_events=0` (`chunk_events` may be set
  per host to bound memory; it changes nothing but floating-point summation order). The sigmoid
  temperature is the shipped 1.0. Nothing else is swept.
- **§3 τ.** The floors' grid `floor_taus` (`plans/floors.md` §3), strict (`score > τ`); a real
  selection on this soft output.
- **§3 freeze rule.** Per cell `lmt/topology/none/frozen/<grain>`: the argmax of validation
  directed F1 at the default floor over `(N, τ)`; ties → smaller `N` (the weaker prior), then
  larger τ — the harness's own rule. A frozen `N` at either end of the grid is recorded, not
  refused.
- **§3 floor.** The scorer's default floor (0.05); the coverage rule is reported, never gated.

## 4. Protocol

Per corpus and grain, the family's protocol (`plans/cascade.md` §4 and addendum) with stage
`lmt`: `externalarm --method lmt --arms lmt/topology --split val --settings 1000 10000 100000
--method-args <§3>` → `scoresweep --arms lmt/topology --floor-taus` → `freeze --pretrain-results
none` (its own dated file `freezes/<date>-<rung>-<variant>-s<k>-external-lmt.json`) → one committed
freeze → `externalarm --split test --freeze … --cells <one cell>=<τ> --settings <the frozen N>`
(one arm, one fit) → `annotate --pretrain-results none --per-lag-files 0` → landing under
`results/<rung>/latent/seed=<k>/lmt/topology/none/frozen/<grain>/` with the read's records under
`test-read/lmt-topology-<grain>/` and `lmt-val-sweep/`, ledger entries of stage `lmt`. Staging as
the floors' (`smallest-rung` at xs; above xs the arms' previous-rung ledger entry of the same
grain). Every read records `ordering = end`, the sample, the settings, `method_sha256`,
`method_commit`, the interpreter, `method_env` and the wall clock of every fit.

**Rungs (owner decision 2026-11-04).** **xs and s with five seeds each are required** for the arm
to enter the submission; **m where it lands by Monday 10 November**, the paper saying so where it
does not; l and xl are not run. xs and s run locally on CPU (xs: 10 s per read, 0.3 GB; s: §9's
extrapolation) or on a CPU host sized from the measured s seed-0 validation fit, which is timed
first. **m under this plan is expected not to land** (D-LM-4: at K ≈ 2 650 the penalty's gradient
overflows the optimiser's state in float64 as well, and the adjacency does not move); if the
owner approves a change to the shipped objective (a log-domain or spectral acyclicity term, a
smaller initial A), it is a dated addendum here and a D-LM-n row before any m read.

## 5. Caveats carried beside every number

- **A soft prior, not a mask.** Recall is not capped by the topology (unlike `cascade/topology`,
  `thp/topology`); the prior's pull is `w · 4.6` per unit of A on a pair outside the topology,
  against a likelihood that grows with the sample. What the arm finds outside the topology is
  measured (H-lmt-outside-mask), not assumed.
- **K² parameters and an O(K³) penalty.** Three K × K matrices (A_raw, log α, log β): 18k at xs,
  104k at s, ≈ 7M at m; `h(A)` is a matrix exponential per step (cheap to s, seconds per step at
  m). The fit is **far from a DAG at the shipped epoch count** (xs seed 0, w = 10: `h` from 2.8e8
  to 1.3e5 in 50 epochs); the thresholded graph carries no acyclicity guarantee.
- **The regime.** The paper's experiments have K ≤ 16 clusters and N ≤ 1 000 events; its
  `epochs=50, lr=0.05` are tuned for that. At K = 77 / N = 5.7k (xs) 50 Adam steps move each raw
  parameter by at most 2.5 (§9: the ranking's directed AUROC is 0.515 on the probe). The arm is
  run as shipped; the epoch count is a fixed setting of §3, not swept, and its adequacy is part of
  what the arm reports.
- **The likelihood is the lab's reimplementation** (D-LM-1; equality 3e-14 / 3e-6 on the shipped
  sample); the shipped loop itself is never run on a harness sample.
- **Precision is float64** (D-LM-4), not the shipped float32; at xs both are finite.
- **Sample-bound, sample-paired** with the arms and the bigram floor, not with the topology floor
  (`plans/cascade.md` §5); the validation / test sample-size note of `plans/caps.md` travels with
  every s and m number.
- **Causal sufficiency.** A Hawkes process assumes no hidden confounding; the latent corpora
  violate it by design.
- **Five pairs: low power** (`report`'s own note).

## 6. Report rows and pairs

Not in the required set. Pairs added to the report of every rung the arm ran on (frozen cut, both
grains; a pair shares corpus and seed, not a model — D-SB-15):
`lmt/topology/none/frozen : floor/topology/none/frozen`, `: floor/bigram/none/frozen`,
`: trace/core/shipped/frozen`, `: baseline/granger/shipped/frozen`.

## 7. Hypotheses (scored pass or fail per rung in `findings/lmt-latent.md`)

- **H-lmt-topology-request.** At the frozen cut and the default floor, `lmt/topology`'s directed
  F1 at the request grain is **at least 0.02 above `floor/topology`'s** on every rung (five-seed
  mean of the paired difference, same corpus and seed, p < 0.05). *Fail:* any rung where the mean
  gain is below 0.02 or p ≥ 0.05.
- **H-lmt-topology-session.** The same at the session grain. *Fail:* as above.
- **H-lmt-vs-core.** `lmt/topology`'s directed F1 is **above `trace/core/shipped/frozen`'s** at
  both grains on every rung (paired, p < 0.05). *Fail:* any rung and grain where it is not.
- **H-lmt-outside-mask** (reported, no pass / fail). At the session grain and the frozen τ: the
  **number of predicted directed edges outside the topology's pair set** and **their precision**
  against the session truth (the journey and retry edges lie outside the topology; a soft prior
  may reach them), per seed and rung, beside the request-grain count.

## 8. Deviations (RUN.md rows, numbered before the first run)

- **D-SB-29** (the family's, `plans/cascade.md` addendum) — the per-method registry, `mask_mode =
  count`, the method's own freeze file, `method_commit`, `method_env`.
- **D-LM-1** — the Hawkes log-likelihood and compensator are evaluated by the lab's vectorised
  forward (`lmt_runner.hawkes_chunk`), not the shipped O(N²) Python loop; same parametrisation,
  init, prior, penalty and optimiser; equality on the shipped 50-event sample recorded with its
  numbers; `chunk_events` bounds memory at equal gradient.
- **D-LM-2** — times in seconds (integer milliseconds / 1000) rather than the clone's CSV seconds
  column; `T` = the last event's time.
- **D-LM-3** — the q-matrix from the topology mask (`q_in` / `q_out`), in place of the paper's
  LLM-distilled cluster prior; the arm's `P_SME` is the deployment topology.
- **D-LM-4** — parameters, prior and penalty in float64 (the shipped model in double precision);
  the kernel sums in float64; the reason and the overflow thresholds of `h(A)` under the shipped
  initialisation.

## 9. Provenance (not results)

- The paper (review `reviews/lmt-xiao-2026.md`): on simulated chemical-plant alarms LMT reports a
  lower SHD than CASCADE and the temporal-only, text-only and direct-prompting baselines, with the
  prior's gain concentrated on the **FDR side** in the small-sample regime (it suppresses false
  discoveries when the Hawkes likelihood alone cannot identify the graph). Its largest graph has
  16 nodes.
- 2026-11-03, the lab's timing probe (xs seed 0, request grain, validation sample, w = 10, the
  shipped float32 parameters, not landed): K = 77 types, N = 5 699 events, 551 allowed pairs
  (625 prior pairs, 74 outside the sample); read 9.75 s wall, the fit 5.5 s (50 epochs, 0.093 s
  per epoch), runner peak memory footprint 281 MB; `h` 2.8e8 → 1.3e5, loglike −1.6e5 → −3.1e4; 464
  off-diagonal entries above 0.5, 417 of them outside the mask. Directed read-out on
  `floor_taus` ⊂ {0, 0.1, 0.3, 0.5, 0.7, 0.9}: F1 0.214 / 0.214 / 0.208 / 0.102 / 0 / 0
  (precision 0.122 and recall 0.881 at τ ≤ 0.1 = every cross-op pair scored), directed AUROC
  0.515, average precision 0.124 — against the topology floor's request F1 ≈ 0.82. The ranking of
  the shipped 50-epoch fit is at chance on this sample.
- 2026-11-03, two variants of the same probe (not landed): **float64 parameters, 50 epochs** —
  `h` 2.8e8 → 1.3e5, loglike −3.1e4, the same F1 row (0.214 / 0.214 / 0.207 / 0.109 / 0 / 0),
  AUROC 0.532, 480 entries above 0.5 (429 outside the mask); **float64, 500 epochs** (129 s) — `h`
  4.6e4, loglike −8.7e3, F1 0.214 / 0.214 / 0.209 / 0.061 / 0 / 0, AUROC 0.534, 283 above 0.5
  (260 outside). Ten times the shipped epoch count does not move the ranking off chance at K = 77;
  the mask prior (w = 10) does not dominate the entries outside it.
- 2026-11-03, the synthetic K = 323 check (20k uniform events over an hour, 2 000 random allowed
  pairs, 4 epochs): float32 parameters — `h` = 8.709e35 bit-identical across epochs, A unchanged
  from its initialisation (the stall of D-LM-4); float64 parameters — `h` 9.1e35 → 2.0e32, 1.03 s
  per epoch, 790 MB peak.
- 2026-11-03, the synthetic s-sized timing (K = 323, 640k uniform events over a day, 4 000
  allowed pairs, float64, 2 epochs; `algos/lmt-xiao-2026/RUN.md`): unchunked 169 s per epoch at a
  20.5 GB peak (swapping on an 18 GB box); `chunk_events=160000` 86 s per epoch at 7.9 GB, loss
  bit-identical. Per-epoch cost is linear in the event count. If s sequences average ~5 events as
  xs does, an s fit is minutes and a seed under an hour locally; at the 64-token worst case
  (640k validation events) a fit is 72 min and a seed ≈ 9 h — a CPU host per seed.

## Addendum 2026-11-03 — the fixed method-args re-registered from a validation-selected grid (D-LM-5)

Dated before the first governed read. Owner decision (2026-11-03): the fixed arguments of §3 are
chosen by a **small, pre-stated grid on the xs seed-0 validation sample** — a disclosed tuning
step the paper must report as a departure from the shipped defaults — and the arm then runs as
registered on every rung and seed. Motivation (§9): at the shipped start `A_raw = 0` (σ(0) = 0.5
on every pair) the acyclicity penalty is `h(A) ≈ exp(0.25 (K − 1))` and dominates the fit; the
ranking of the shipped 50-epoch fit was at chance on xs (AUROC 0.515–0.534, F1 0.214 = every
pair), and at K ≥ 200 (float32) / K ≥ 1 450 (float64) the squared gradient of `h` overflows
Adam's second moment so the adjacency never moves. A new method-arg **`a_init`** (the offset of
`A_raw` at initialisation, `A_raw = a_init + noise`; the shipped value is 0) starts the relaxed
adjacency sparse.

**The grid (stated before the runs; 15 fits at the request grain, then the chosen setting once at
the session grain).** `lr = 0.05`, `lambda_e = 0.1`, `seed = 0`, `prior_weight_milli = 10 000` (the
`N` grid's middle value), `q_in = 0.5`, `q_out = 0.01`, float64, the arms' validation sample
(1 154 sequences, 5 699 events, K = 77). Block A: `a_init ∈ {0, −2, −4} × beta_dag ∈ {1.0, 0.1,
0.01}` at `init_std = 0.35`, `epochs = 50`. Block B: `a_init ∈ {0, −2, −4}` at `beta_dag = 0.1`
(the middle value), `init_std = 0.0`, `epochs = 50`. Block C: `a_init ∈ {0, −2, −4}` at `beta_dag
= 0.1`, `init_std = 0.35`, `epochs = 200`. Scored by `scoresweep` on `floor_taus` (20 values).
**Selection rule:** the best validation directed F1 at the request grain over τ, ties by directed
AUROC; one setting for both grains.

| block | `a_init` | `beta_dag` | `init_std` | `epochs` | best τ | F1 | prec | rec | edges | AUROC | AP | > 0.5 (outside) | `h` | loglike | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (shipped) | 0 | 1.0 | 0.35 | 50 | 0.20 | 0.2144 | 0.122 | 0.879 | 5 574 | 0.532 | 0.128 | 480 (429) | 1.3e5 | −31 355 | 9.2 |
| A | 0 | 0.1 | 0.35 | 50 | 0.20 | 0.2144 | 0.122 | 0.879 | 5 574 | 0.532 | 0.128 | 475 (424) | 1.3e5 | −31 319 | 5.0 |
| A | 0 | 0.01 | 0.35 | 50 | 0.25 | 0.2151 | 0.123 | 0.850 | 5 374 | 0.538 | 0.131 | 433 (389) | 1.0e5 | −30 968 | 4.4 |
| A | −2 | 1.0 | 0.35 | 50 | 0.05 | 0.4332 | 0.712 | 0.311 | 213 | 0.740 | 0.475 | 1 (0) | 0.010 | −15 033 | 4.6 |
| A | −2 | 0.1 | 0.35 | 50 | 0.05 | 0.4332 | 0.712 | 0.311 | 213 | 0.740 | 0.475 | 1 (0) | 0.010 | −15 033 | 5.1 |
| A | −2 | 0.01 | 0.35 | 50 | 0.05 | 0.4332 | 0.712 | 0.311 | 213 | 0.740 | 0.475 | 1 (0) | 0.010 | −15 033 | 4.9 |
| A | −4 | 1.0 | 0.35 | 50 | 0.01 | 0.4275 | 0.493 | 0.377 | 384 | 0.737 | 0.449 | 0 (0) | 0.0018 | −13 734 | 5.0 |
| A | −4 | 0.1 | 0.35 | 50 | 0.01 | 0.4275 | 0.493 | 0.377 | 384 | 0.737 | 0.449 | 0 (0) | 0.0018 | −13 734 | 5.4 |
| A | −4 | 0.01 | 0.35 | 50 | 0.01 | 0.4275 | 0.493 | 0.377 | 384 | 0.737 | 0.449 | 0 (0) | 0.0018 | −13 734 | 5.5 |
| B | 0 | 0.1 | 0.0 | 50 | 0.30 | 0.2143 | 0.122 | 0.881 | 5 600 | 0.796 | 0.666 | 0 (0) | 7.5e4 | −30 883 | 6.0 |
| B | −2 | 0.1 | 0.0 | 50 | 0.03 | 0.4994 | 0.590 | 0.433 | 347 | 0.843 | 0.599 | 1 (0) | 0.0096 | −14 897 | 9.9 |
| B | −4 | 0.1 | 0.0 | 50 | 0.02 | 0.4318 | 0.585 | 0.342 | 292 | 0.807 | 0.505 | 0 (0) | 0.0016 | −13 687 | 7.9 |
| C | 0 | 0.1 | 0.35 | 200 | 0.25 | 0.2156 | 0.123 | 0.850 | 5 355 | 0.538 | 0.131 | 377 (340) | 7.9e4 | −11 973 | 22.2 |
| **C** | **−2** | **0.1** | **0.35** | **200** | **0.01** | **0.5467** | 0.455 | 0.684 | 760 | **0.831** | 0.583 | 3 (0) | 0.0011 | −10 233 | 20.1 |
| C | −4 | 0.1 | 0.35 | 200 | 0.01 | 0.4182 | 0.532 | 0.344 | 329 | 0.807 | 0.498 | 2 (0) | 0.00035 | −10 019 | 19.1 |

(`edges` = scored pairs above the best τ; `> 0.5 (outside)` = off-diagonal entries above 0.5 and
those among them outside the mask; `fit s` = the runner's wall clock, 0.08–0.18 s per epoch at
K = 77, N = 5.7k; the harness read adds ≈ 4 s.) Reading: `beta_dag` is immaterial once the start is
sparse (`h` ≤ 0.01 throughout, identical rows); the sparse start is what moves the ranking off
chance; `−2` beats `−4` at every epoch count; the init noise helps at 200 epochs (C) and hurts at 50
(B); the chosen row's F1 curve over τ is 0.214 / 0.372 / **0.547** / 0.479 / 0.457 / 0.430 / 0.407 /
0.359 / 0.260 / 0.180 / 0.135 / 0.060 / 0.009 … at τ = 0 / 0.005 / 0.01 / 0.02 / 0.03 / 0.05 / 0.07 /
0.1 / 0.15 / 0.2 / 0.25 / 0.3 / 0.4 … (the sigmoid output sits below ≈ 0.3; the selected τ is the
grid's second-lowest value and is recorded as a grid-edge cell where the freeze lands there).

**Chosen (registered from here on; §3's fixed method-args are superseded):**

    epochs=200 lr=0.05 beta_dag=0.1 lambda_e=0.1 init_std=0.35 a_init=-2 seed=0 device=cpu q_in=0.5 q_out=0.01 dtype=float64 param_dtype=float64 chunk_events=0

Against the shipped `epochs=50 beta_dag=1.0 a_init=0`: xs seed-0 validation request F1 0.547
vs 0.214, AUROC 0.831 vs 0.532 — still below `floor/topology`'s ≈ 0.82. The same setting at the
session grain: F1 **0.481** at τ = 0.01 (precision 0.462, recall 0.503, 787 scored pairs above τ), AUROC 0.753, AP 0.481, `h` 0.0011, loglike −10 907, 3 entries above 0.5 (none outside the mask), 23.3 s (0.113 s per epoch) — the same τ as the request grain; F1 over τ 0.215 / 0.322 / 0.481 / 0.396 / 0.353 / 0.314 … at τ = 0 / 0.005 / 0.01 / 0.02 / 0.03 / 0.05 …. The runner's defaults carry the chosen values; the shipped ones stay
selectable (`epochs=50 beta_dag=1.0 a_init=0`).

**m under the chosen start.** `h(A)` at initialisation in float64 (seed 0, `init_std` 0.35): K = 323
— 9.1e35 at `a_init = 0`, **220 at −2**, 0.009 at −4; K = 2 650 — 9.3e295 at 0 (its gradient 6e292,
squared beyond float64: the stall), **2.4e19 at −2 (gradient 9e15, squared finite even in float32)**,
0.87 at −4. So under the registered `a_init = −2` the m fit is **numerically computable** in float64
(and the penalty is inactive from the start at s); its cost is the §9 extrapolation plus ≈ 3 s per
2 650 × 2 650 matrix exponential per step (forward and backward), i.e. m remains a CPU-host read,
not a local one, and §4's "m does not land" is withdrawn in favour of "m where it lands by the
Phase A gate" (the family's rule). CPU-heavy reads (s, m) run on EC2 replicas, not locally.

- **D-LM-5** — the fixed method-args `epochs=200 beta_dag=0.1 a_init=-2` are a validation-selected
  departure from the clone's `run_k5.sh` defaults (`50 / 1.0 / 0`), chosen on the xs seed-0
  validation split by the rule above before any governed read, applied unchanged to every rung and
  seed, and disclosed in the paper; `a_init` is a lab addition to the shipped initialisation (an
  offset on `A_raw`, with the shipped noise on top). Equal-budget note: 200 epochs is 4 × the shipped
  count (20 s at xs).
