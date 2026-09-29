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

