# Declared caps and budgets (pre-registered 2026-09-24, before any run)

PRD carried decisions 1 and 2, and scenario 18: every stage computes its
memory estimate from its recorded inputs before allocating, writes the
estimate and the cap into the run record, and refuses when the estimate
exceeds the cap. Wall-clock caps are the provisioning replica's
`max_runtime_hours`. Changes are dated addenda at the end of this file, never
edits in place.

## Model and training budget (every rung)

Llama `6` layers / `d_model 256` / `8` heads / `ff_mult 2` (SwiGLU hidden 512)
/ tied embedding / positions `max_len + 2` — the sibling's `xs` anchor
(≈ 4.2 M non-embedding parameters; the vocabulary embedding is the only size
that varies with the rung, ≈ 5 M extra at `xl`). `12,000` steps × `256`
sequences, `lr 3e-4`, warmup `500`, cosine to zero, weight decay `0.1`, betas
`0.9 / 0.95`, grad clip `1.0`, bf16 autocast with the loss in float32,
validation every `1,000` steps over `50` batches, checkpoint every `1,000`
steps, entropy order `2`, `--max-len 64`, seed = the corpus seed. Trained on
the `end-session` view (D-SB-7). Uniform across rungs so cells stay
comparable.

**Checkpoint rule (carried decision 3, owner 2026-09-24):** the frozen model is
the **last** checkpoint. Trigger: if `val_loss_final / val_loss_min` exceeds
`1.01` (`CHECKPOINT_TRIGGER_RATIO`), the validation sweep also probes the
argmin-val checkpoint with `trace/core` at the request grid's `c*` and
`N = 8`, and the freeze records which checkpoint won on validation directed
F1 at the default floor; every arm then shares that pick. Every checkpoint
stays in object storage.

## Memory estimate (D-SB-12)

For a particle probe on one sequence of length `L` with context `c`
(`Lc = L − c` rows), `N` particles and vocabulary `V`:

    shipped_estimate  = cli.estimate_tensor_bytes(N, Lc, L, V)   (the logits tensor, float32)
    harness_estimate  = 2 · shipped_estimate                       (logits + softmax / log_softmax)
                      + N · Lc · L · d_model · n_layers · 4 · 4     (activations, an upper-bound factor)
                      + model parameters · 4

The stage refuses when `harness_estimate > --memory-cap-gb · 2^30` and
records `shipped_estimate`, `harness_estimate`, the cap and the measured
`peak_device_bytes`. Saliency and Shapley have no particle tensor; their
estimate is the per-row activation term.

## Caps per rung

`xs` is the calibration run: every later replica is re-sized from `xs`'s
recorded `tok_pos_per_s` and `peak_device_bytes` by a dated addendum below.

| rung | V (realised) | val sweep sample (request / session) | test read | Shapley sample | N grid | memory cap (GiB) | Job V (h) | Job T (h) | instance class |
|---|---|---|---|---|---|---|---|---|---|
| xs | 76–79 | whole split | whole | whole | 2, 8, 32 | 20 | 4 | 3 | one A10G, 16 GB host |
| s | 322 | 10k / 5k | 20k / 10k | 2k / 500 | 2, 8, 32 | 20 | 12 | 8 | one A10G, 16 GB host |
| m | 2,644–2,654 | 10k / 5k | 20k / 10k | 2k / 500 | 2, 8, 32 | 20 | 16 | 10 | one A10G, 16 GB host |
| l | 7,430–7,533 | 10k / 5k | 20k / 10k | 2k / 500 | 2, 8, 32 | 20 | 24 | 14 | one A10G, 32 GB host (the 1.3 GB session target) |
| xl | 19,677–19,873 | 10k / 5k | 20k / 10k | 2k / 500 | 2, 8, 16 | 20 | 36 | 24 | one A10G, 64 GB host (the 6–7 GB session target) |

Sizing basis: at the session grain (`L = 64`, `Lc = 63`) the logits tensor is
`N · 63 · 64 · V · 4` bytes — 0.08 GB at `xs`, 0.33 GB at `s`, 2.7 GB at `m`,
7.8 GB at `l` and 10.2 GB at `xl` for `N = 32`; doubled with the softmax, so
`N ≤ 16` at `xl`. Request sequences (`L ≈ 13`) are negligible everywhere.
The shipped engine's throughput is unmeasured; the sibling's anchor on the
same GPU class is 6.9e5 token-positions per second in float32, and the
shipped full-vocabulary softmax will be slower. Shapley costs about
`10 · L · (L − c)` row-forwards per sequence, hence its own sample.

## Addenda

### 2026-09-25 — xs calibration run landed (Job V, `xs/latent/seed=0`, g5.xlarge A10G)

**Measured anchors** (the run record; sweep cells are the whole validation split,
1,154 request / 437 session sequences; owner-reviewed 2026-09-25):

| stage | measured | note |
|---|---|---|
| pretrain 12k × 256 bf16 | 305 s; 1.42e5 tok-pos/s; peak device 0.88 GB | 8.8× below the sibling's anchor (HF Llama, no fused kernels) |
| particle probe cell (core / cli-full / cli-atomic) | request 8–16 s, session 3–18 s | **flat in N** (2 → 32 changes the wall clock by < 10 %): the per-sequence Python loop, not the GPU, is the cost; ≈ 13 ms / request sequence, ≈ 15–40 ms / session sequence |
| saliency | 0.09 s / request sequence; 0.21 s / session sequence | — |
| Shapley | 0.90 s / request sequence; 4.6 s / session sequence | 74 % of the request sweep, 89 % of the session sweep |
| sweep wall clock | request 59 min; session 95 min | chain 2 h 41 min end to end |
| peak device memory | request sweep 0.20 GB; session sweep 3.07 GB | the session figure equals the activation term of the D-SB-12 formula at `N = 32, L = 64`; the logits term is 0.08 GB at `V = 79` |

**Projection for `s`** (10k / 5k validation sample, 2k / 500 Shapley sample, from the
per-sequence costs above): particle cells ≈ 2.0 h (request) + 1.5 h (session), Shapley
≈ 1.5 h + 1.9 h, saliency ≈ 0.25 h, pretrain ≈ 0.1 h → **≈ 7.2 h**, inside the 12 h cap at a
1.7× margin. The full-vocabulary softmax's growth with `V` is unmeasured at `V = 79` (the
loop dominates); `s` is the first rung that measures it, so its cap stays at 12 h and is not
tightened.

**Checkpoint rule (replaces carried decision 3 and the trigger paragraph above; owner
decision 2026-09-25).** The frozen model is the **argmin-validation checkpoint**: `pretrain`
validates and checkpoints every **250** steps (`--val-every 250 --checkpoint-every 250`, 48
candidates plus the final step), writes the argmin checkpoint as `model/`, and records
`model_choice` (`argmin-val`, the chosen step, its validation loss, the final loss and
`val_final_over_min`). The oracle score ε̂ is taken at the chosen checkpoint and reported;
the final checkpoint's ε̂ is recorded beside it. Ties → the smaller step. The `1.01×` trigger,
`--alt-val-tables` and `--checkpoint-choice` are withdrawn. *Why:* at xs the last checkpoint's
validation loss was 1.498× the curve minimum (2.20 at step 1000 → 3.30 at step 12,000, monotone;
train loss 0.62), the argmin was the *first* checkpoint, so the trigger's own action (sweep the
argmin) could not resolve the minimum; the harness trains the backbone, so an overfit model is a
harness defect, not a TRACE result. The step budget stays 12,000 × 256 at every rung (cells
comparable); the selected step is the effective budget and is recorded.

**Threshold classes (owner-agreed 2026-09-25).** A pre-registered threshold is one of:
(1) a *selection rule* — hard, blocks (the freeze); (2) an *instrument-soundness check* —
reported beside every headline, gating only where the failure is something the lab can act on;
(3) a *hypothesis* — a commitment about the claim, never a gate. The coverage rule and the
oracle regime are class 2 (see the arm plans' addenda of the same date).


### 2026-09-27 — s seed-0 Job V landed (`s-latent-s0-val-26-09-26`, g5.xlarge A10G; owner-reviewed on landing)

**Measured** (10,000 request / 5,000 session head sample; Shapley 2,000 / 500; V = 323):

| stage | measured | note |
|---|---|---|
| pretrain 12k × 256 bf16 | 5.8 min; 1.63e5 tok-pos/s; peak 1.69 GB | ≈ 9 epochs of the 347k session rows; argmin-val at step 6750 (xs: 500), last / min 1.015 |
| particle probe cells | 13–14 ms / request sequence; 13–50 ms / session sequence | equal to xs: the per-sequence loop, not the vocabulary, sets the cost; the full-vocabulary softmax's growth with V is invisible at V = 323 |
| saliency | 0.07–0.09 s / request; 0.18–0.26 s / session | the 2026-09-25 projection counted 0.25 h; measured 1.5 h (3 cells × 10k + 3 × 5k) |
| Shapley | 0.67–0.89 s / request; 4.8–6.2 s / session | 1.2 h + 2.2 h on its own sample |
| sweep wall clock | request 213 min; session 269 min (8.0 h) | projection 7.2 h |
| scoresweep | request 25 min; **session 271 min** | the benchmark scorer's mixed-SHD loop rebuilt `set(pd)`/`set(pb)` per universe pair → O(pairs × predicted edges); at 85,580 ordered pairs a 20k-edge prediction cost 10.5 s per call, 16 τ + ranking per column. Fixed upstream (`trace-bench` `fix/score-pair-state-sets`, outputs byte-identical, 0.11 s per call); until the harness pins the fix, budget 4.5 h for it |
| peak device | 0.20 GB (request), 1.98 GB (session) | the D-SB-12 formula's activation term at N = 32; logits 0.17 GB |
| chain | 13 h 06 min | the 12 h cap would have terminated the instance **without an output sync** (the cap is `sleep; shutdown -h now`); the owner cancelled the timer by SSM at 09:46 UTC |

**Cap amendments for `s` (owner decision on landing).** Job V cap **18 h** while the harness pins
trace-bench v0.3.0 (13.1 h measured), **12 h** once the scorer fix is pinned (≈ 8.7 h projected); Job T cap
stays **8 h** (projection ≈ 5 h: discover reads on the 20k / 10k test sample ≈ 3 h with Shapley on 2k / 500,
annotate ≈ 1 h at the slow scorer, seqscore minutes). Seeds 1–4 are copies of the seed-0 replica. A cap
that fires must never destroy the run again: until the template syncs on the cap, the replica's cap carries
a 1.4× margin over the measured chain.

### 2026-09-27 — s seed-0 Job T landed (`s-latent-s0-test-26-09-27`, g5.xlarge A10G; owner-reviewed on landing)

**Measured** (20,000 request / 10,000 session head sample; Shapley 2,000 / 500; the harness pinned
to the fixed scorer, `a1f3a89`):

| stage | measured | note |
|---|---|---|
| particle discover reads | 2.6–4.7 min each (13 ms / request sequence; 21–27 ms / session) | 13 reads; peak 2.89 GB at session N = 32 |
| saliency reads | 30 min (request) + 44 min (session) | 0.09 / 0.26 s per sequence, as in Job V |
| Shapley reads | 29 min (2,000 request) + 49 min (496 session) | 0.87 / 5.9 s per sequence |
| annotate × 38 | 6.8 min in total (3–22 s per cell) | the fixed scorer: Job V's session scoresweep had cost 271 min on the same universe |
| seqscore × 38 | 9.3 min | — |
| chain | 3 h 47 min | projection ≈ 5 h; cap 8 h holds |

**Registered note — validation sample vs test read (owner decision 2026-09-27).** The caps table
gives every rung from `s` a 10k / 5k validation sample and a 20k / 10k test read because the sweep
re-probes its sample on 60 cells per grain and the read once per frozen cell. The two jobs therefore
see different coverage — 0.388 → 0.474 (request) and 0.560 → 0.690 (session) on `s/latent/seed=0` —
and the test F1 of a frozen cell lands above its validation value (+0.005 … +0.047 here) as a
coverage effect, not a split effect; at `xs`, where both jobs probed the whole split, the check was
like for like. The sizes stay as registered (a 20k / 10k sweep would double the 8 h Job V sweeps);
the note is carried in every `s` … `xl` findings file, and the ceilings of both jobs are on record
(freeze `diagnostics.coverage`, `annotate.coverage`). The Shapley sample (2k / 500) bounds that
baseline's ceiling the same way (0.245 / 0.298 on this corpus). Seeds 1–4 Job V run at the **12 h**
cap (the pinned scorer; ≈ 8.7 h projected).

**2026-09-28 — seeds 1–4 Job V measured (pinned scorer).** Chains 8 h 19 min – 10 h 28 min at the
12 h cap (sweeps 7.3–10.1 h — the corpus, not the vocabulary, sets the spread; scoresweeps 2–7 min,
confirming the scorer fix); argmin steps 4000–8000, ε̂ −0.060 … −0.112, all in regime. The seed-4
chain leaves a 1.15× cap margin, inside the 1.4× rule only because nothing fired — the `m` Job V
replicas must re-project from the seed-4 anchor, not the seed mean.

### 2026-09-29 — m seed-0 Job V landed (`m-latent-s0-val-26-09-28`, g5.xlarge A10G; owner-reviewed on landing)

**Measured** (10,000 / 5,000 head sample; Shapley 2,000 / 500; V ≈ 2,650; the owner raised the applied
cap to 24 h in the replica before `terraform apply`):

| stage | measured | note |
|---|---|---|
| pretrain 12k × 256 bf16 | 6 min; argmin-val at step 7500 | ε̂ = −0.021, in regime; last / min 1.024 |
| sweeps | request 205 min; session 263 min (7.8 h) | per-sequence probe costs equal xs and s on all five probes — **the full-vocabulary softmax's growth with V is still invisible at V ≈ 2,650**; `l` / `xl` sweeps project from corpus size alone |
| scoresweeps | request 103 min; session 229 min (5.5 h) | the fixed scorer is linear per call, but the m universe is ≈ 40× s — **from m the scoring rivals the sweeps**; per-file cost ∝ universe |
| peak device | 2.13 GB | logits term still small at N = 32 |
| chain | 13 h 28 min | 1.78× margin at the applied 24 h cap |

**Cap decisions for `m` (owner-applied + this addendum).** Job V cap **24 h** (as applied; 13.5 h
measured × 1.4 ≈ 19 h — the table's 16 h row is superseded); seeds 1–4 replicas inherit it. Job T cap
**12 h** (superseding the table's 10 h): discover ≈ 3.5 h (probe costs flat), **annotate ≈ 4 h** — the
38 annotate calls run the same universe-bound scorer that put the scoresweeps at 5.5 h — seqscore
minutes; ≈ 8 h projected. A trace-bench improvement is on record as a suggestion, not scheduled: the
16 τ of one scoresweep column could share a single ranking pass.

**H-hazard observation:** 0 saturated and 0 collapsed cells on every path at m — the first rung where
the hypothesis is live produced no corrupted cells on the 10k / 5k validation sample; the test read
and the `l` rung keep it under watch.


### 2026-09-30 — m seed-0 Job T landed (`m-latent-s0-test-26-09-29`, g5.xlarge A10G) + seeds 1–4 Job V measured

**Job T measured** (20,000 / 10,000 head sample; Shapley 2,000 / 500):

| stage | measured | note |
|---|---|---|
| discover (16 reads) | 205 min | probe costs flat: particle reads 2.5–5.0 min, saliency 30 + 46 min, Shapley 28 + 48 min — as projected |
| annotate (38 cells) | **593 min (9.9 h)** | the projection said ≈ 4 h; each annotate runs the scorer's six-value floor sweep where a scoresweep file runs 16 τ over the same universe, so the per-cell cost tracks the 3.8-min scoresweep file, ≈ 33 min on the session cells — the sweep-to-annotate ratio was mis-carried from `s`, where the whole stage cost 6.8 min |
| seqscore (38 cells) | 19 min | — |
| chain | 13 h 46 min | the applied replica carried `max_runtime_hours = 18` against this addendum's 12 — the discrepancy saved the run (a 12 h cap fires mid-annotate, and the box syncs only after the chain exits) |

**Cap decision for the `m` Job T** (superseding this addendum's 2026-09-29 value): **18 h**
(13.8 h measured × 1.3; the replicas for seeds 1–4 carry it explicitly, ending the
comment-vs-value drift that has now recurred at `s` and `m`). The annotate projection for `l`
must scale from the 593 min measurement by universe size, not from the discover stage.

**Seeds 1–4 Job V measured (2026-09-29 runs):** chains 14 h 25 min – 15 h 24 min at the 24 h cap
(1.56× margin at the slowest); argmin steps 4000 / 4250 / 6500 / 7750, val losses 1.98–2.13
(ε̂ at each freeze), last / min ≤ 1.026; 0 corrupted cells on every seed; sweeps 3.4–3.7 h
(request) + 4.7–5.3 h (session); scoresweep pace ≈ 2 min (request) / ≈ 4–5 min (session) per
file — the m universe cost, consistent with seed 0. The 24 h cap holds for `m`; the `l` Job V
re-projects from the slowest seed (15.4 h) and the scoresweep's universe growth, not the mean.

**Cap decision for the `l` Job V (2026-09-30, authored with the seed-0 replica).** **30 h**,
superseding the table's 24, on a **g5.2xlarge** (the table's "32 GB host" instance class):
sweeps ≈ 8 h (probe costs flat in V through m, fixed 10k / 5k sample), scoresweeps = m's 5.5 h
× the universe growth (s→m the universe grew ≈ 3.2× while V grew 8.2×; l's V ratio is 2.8× →
≈ 2–2.5×, so ≈ 11–14 h) → ≈ 20–23 h projected, and 24 h leaves under 1.2× margin against a
hard kill without sync. First rung near the memory cap: the session logits + softmax term is
≈ 15.6 GB at N = 32 against the 20 GiB knob — the run's memory events are a landing check.

**2026-10-01 — m Job T, seeds 1–4 measured.** Chains 15 h 49 min – 16 h 50 min at the 18 h cap —
the slowest (seed 4, annotate 745 min) left a **1.07× margin**, the tightest of the programme;
the cap held only because the addendum's 18 h was applied as written. Annotate 680–745 min on
every seed confirms the stage as the m-rung cost driver. The `l` Job T must re-project from the
seed-4 anchor (16.8 h) times the scoresweep-measured universe growth — with the l Job V's
observed scoresweep costs (request 11.6 h vs m's 1.7 h, ≈ 6.8×) an unbatched l Job T projects
far beyond a practical cap, so the trace-bench τ-batching improvement (one ranking pass shared
across the 16 τ of a scoresweep column, suggestion on record 2026-09-29) graduates from
suggestion to prerequisite for the `l` test reads.

**2026-10-01 — scorer re-pinned to the sparse context (`8e91aa2`); the prerequisite above is
discharged, by a different route.** Profiling the pinned scorer showed the cost was not the τ
loop: `score_at_floor` walked the whole universe on every call whatever the prediction's size
(≈ 4 s per call at the m session universe), and annotate — which has no τ loop — pays the same
per call, seven floors and up to eight lags per cell. Sharing a ranking pass across τ would have
left annotate, the Job T cost driver, where it was. The scorer now takes a `ScoreContext` (the
sorted universe and, per floor, the truth side) and a call costs in proportion to the truth and
predicted edges; `scoresweep` builds one context per grain and `annotate` one per cell. Output is
byte-identical — re-scoring the landed sweeps reproduces every landed val table exactly:

| landed table | landed scorer | landed wall clock | re-scored (laptop, same sweep files) |
|---|---|---|---|
| s seed 0, request | v0.3.0 | 25 min | 18 s |
| s seed 0, session | v0.3.0 (quadratic) | 271 min | 66 s |
| m seed 0, request | `a1f3a89` | 103 min | 158 s |
| m seed 0, session | `a1f3a89` | 229 min | 449 s |

One m session annotate cell (`trace/core/fixed/frozen/session`, 1,979 s as landed) re-scored its
`score.json` and `score-ranking.json` byte-identical with all scoring done inside 29 s. What
remains per file is reading the sweep table and building the prediction documents, so the
scoring stages no longer scale with the universe. Consequences: the `l` seed-0 Job V (landed on
the old pin, ≈ 38 h of scoresweeps) is the last run that pays the universe cost — its tables are
re-scored on the new pin at landing as the l-rung identity check; the `l` Job T and the `l`
seeds 1–4 caps are re-projected from the sweep and discover stages plus these anchors, not from
the m annotate or the l scoresweep measurements.

**2026-10-02 — `l` Job V, seed 0 measured; the l-rung identity check passes.** Chain 46 h 23 min
on a g5.2xlarge at **cap 40 h as applied** (the owner raised the authored 30 before apply; the
cap timer was cancelled on the box on 2026-10-01, on the owner's word, once the scoresweep pace
showed even 40 h would fire before the sync — the same intervention as at `s`).

| stage | measured | note |
|---|---|---|
| pretrain | 6 min | argmin-val step 1250 of 12,000, ε̂ = +0.036 (in regime), last / min 1.21 |
| sweep, request (60 cells) | 191 min | probe costs flat in V through `l` (m: 205 min) |
| sweep, session (60 cells) | 228 min | m: 263 min; peak device memory 6.37 GB against the 20 GiB knob — the ≈ 15.6 GB logits + softmax projection did not materialise; two allocator out-of-memory retries, recovered |
| scoresweep, request | **11.9 h** | universe 11.3 M ordered pairs (m ≈ 1.7 M): the universe-bound scorer `a1f3a89` |
| scoresweep, session | **27.4 h** | universe 25.6 M ordered pairs; 17–45 min per file by probe and noise |

The 2026-09-30 projection (scoresweeps ≈ 11–14 h) carried the s→m universe growth forward; the
universe grows roughly with V², so m→l it grew ≈ 6.6–6.9× (1.7 M → 11.3 M request, 3.7 M →
25.6 M session), not 2–2.5×. All of the overrun was the scorer.

**Identity check at `l` (the open item of the 2026-10-01 entry).** Both landed val tables were
re-scored from the synced sweep files on the pinned sparse-context scorer (`8e91aa2`) and are
equal on every key (1,596 rows each):

| landed table | landed scorer | landed wall clock | re-scored (laptop, both grains concurrently) |
|---|---|---|---|
| l seed 0, request | `a1f3a89` | 11.9 h | 445 s |
| l seed 0, session | `a1f3a89` | 27.4 h | 1,201 s |

**Coverage at `l`.** The 10k / 5k validation sample co-observes 0.28 % / 0.58 % of the ordered
pairs and reaches 6.6 % (request) / 10.3 % (session) of the truth edges, so validation F1 is
ceiling-bound (0.09 / 0.11) and the arms sit within 0.003–0.008 of each other; the two Granger
request cells freeze at the bottom of the Granger grid (recorded, `at_grid_edge`). The registered
sample-size note applies with more force than at `m`: the test read's 20k / 10k sample is not
like-for-like with validation.

**Cap decisions for `l`.** Job V, seeds 1–4 (on the re-pinned scorer): sweeps ≈ 7 h + scoresweeps
≈ 0.5 h → ≈ 8 h projected; applied at **48 h** (owner's value). Job T: discover ≈ 3.5–4 h (the
m read cost 205 min at equal probe costs), annotate + seqscore on the re-pinned scorer — first
measured at `l` by the seed-0 read — projected ≈ 1–2 h; **cap 24 h**. The cap remains a hard
shutdown without a sync: never tighten it below a measured chain × 1.3.

**2026-10-03 — `l` Job V, seeds 1–4 measured (the first runs on the re-pinned scorer).** Chains
8 h 24 min – 10 h 06 min at the 48 h cap the owner applied (the authored 24 would have held with
a 2.4× margin at the slowest): sweeps 204–229 min (request) + 253–323 min (session), probe costs
flat in V as at seed 0; **scoresweeps 11–12 min (request) + 31–37 min (session)** against seed 0's
11.9 h + 27.4 h on the universe-bound scorer — the sparse context removes the universe term from
the validation side, as the identity check projected. Peak device memory 4.46–6.70 GB. Argmin
steps 1750 / 750 / 1500 / 3000, ε̂ +0.042 … +0.063 (all in regime), last / min 1.15–1.23 — the
uniform 12k budget overfits every `l` corpus as it did at xs and m; the 250-step grid holds.
0 corrupted cells on every seed (H-hazard: zero incidence on all five `l` corpora, val side). Seed
2 freezes five request cells at a grid bottom (three trace arms at 1e-7, both Granger at 1e-5);
seed 0 froze the two Granger request cells there — the coverage-bound validation sample makes the
lowest τ the argmax on the sparse request grain. The `l` Job V projection for `xl` is sweeps
≈ 7–9 h (fixed sample; flat probe costs) + scoresweeps ≈ 1 h; the xl Job V cap re-projects from
the slowest `l` chain (10.1 h) × 1.3 plus the memory step to N = 16.

**Cap for the `l` Job T, seeds 1–4: 24 h**, as for seed 0 (its read is in flight; the first
measurement of annotate at `l` on the re-pinned scorer lands with it — re-anchor if the seed-0
chain exceeds 12 h).

**2026-10-03 — `l` Job T: discover measured, annotate stopped and fixed.** All five test reads
finished discover in 3.1–3.7 h of probe wall clock (16 / 16 / 18 / 17 / 18 reads, peak device
memory 6.4–6.7 GB, 0 corrupted cells) — the projection held. Annotate did not: a frozen request
cell took 147–246 min and a shipped-cut cell 2.5–5 min, so the 24 h cap would have fired with
eleven or twelve request cells still to go on every box and no session cell started. The owner
stopped annotate on all five boxes; the chains exited 143 and synced.

The cost was in the harness, not the scorer: `annotate.confounded_pair_count` rebuilt the set of
bidirected truth pairs for every predicted edge. The `l` truth holds 50,291 (request) and 179,049
(session) such pairs and a frozen prediction 50k–150k edges; measured on the seed-0 truth, 500
predictions cost 29 s (request) and 179 s (session). The set is now built once per call. Run end
to end on the synced seed-0 reads (laptop), the fixed stage reproduces the landed cell
`trace/cli-atomic/fixed-kl/frozen/request` byte for byte — `annotate.json`, `score.json`,
`score-ranking.json` — in 68 s against 155 min:

| command, one cell | request | session | peak memory |
|---|---|---|---|
| annotate (frozen cell) | 68 s | 151 s | 6.3 GB |
| seqscore | 42 s | 175 s | 5.8 GB |

**What the 2026-10-01 and 2026-10-02 identity checks did and did not cover.** They re-ran the
scorer (val tables; one m annotate cell's two score files), not the annotate command — the m
check could not, its discover scores were not local — so the `l` Job T cap was projected from a
function-level measurement. From here a cap is projected only from the stage's own command run
end to end on real inputs of the rung, one cell at least per grain.

**Scoring half of the `l` test reads.** The discover reads are kept (verified against the
manifests); annotate + seqscore run again per seed from the synced reads, fetched as an input
artifact (`scripts/jobs/seq2cause-bench-26-10-03-l-latent-s{0..4}-annotate.sh`: pull method, pull
score, 38 annotate, 38 seqscore). Projection ≈ 2.3 h of laptop wall clock for the 76 commands,
≈ 7 h on the box's CPU; no GPU work, so the smallest instance of the image's GPU families with a
16 GB host; **cap 24 h**.

**2026-10-04 — scoring half of the `l` test reads measured; the 16 GB line above was wrong.**
Attempt 1 ran on a 16 GB host and was killed for memory (exit 137) on the first session annotate
cell on all five seeds, after the 19 request cells (44–58 min); the output synced. The 6.3 GB
figure in the table above was the laptop's maximum resident set size, which leaves out compressed
memory; the true laptop peak footprint is 15.5–15.6 GB for every session annotate and seqscore
cell (set by the session universe, not by the scores file) and 7.0 GB at request. Attempt 2, on a
32 GB CPU host (the default image boots on a CPU instance; the runner only warns that no GPU is
visible), completed on every seed:

| seed | chain | annotate | seqscore | peak resident memory |
|---|---|---|---|---|
| 0 | 2 h 38 min | 82 min | 72 min | 16.0 GB |
| 1 | 3 h 12 min | 102 min | 86 min | 17.8 GB |
| 2 | 3 h 15 min | 105 min | 87 min | 17.9 GB |
| 3 | 3 h 15 min | 100 min | 91 min | 18.9 GB |
| 4 | 3 h 08 min | 98 min | 86 min | 17.4 GB |

So the box ran the 76 commands in ≈ 1.3× the laptop's wall clock (not 3×) and needed 1.0–1.2×
the laptop's peak footprint. **Rules from here:** a host is sized from the peak memory footprint
of the largest grain's command (never the resident set size) with at least 2× headroom, and the
scoring half of a test read needs no GPU — at `l` it is ≈ 3 h on a 32 GB CPU host against the
3.1–3.7 h of discover on the GPU host. For `xl` the session universe grows again (≈ 7× by V²):
the scoring half is sized from a measured `xl` cell before its replica is authored, and running
it as its own CPU job after discover syncs is the default shape.
