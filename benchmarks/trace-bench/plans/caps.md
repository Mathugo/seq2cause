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

(none)
