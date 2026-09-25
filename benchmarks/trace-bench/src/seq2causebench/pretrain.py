# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Learning the density: train one backbone on a corpus's training split by
next-token prediction (PRD Behavior, "Learning the density"; D-SB-13).

    python -m seq2causebench.pretrain --corpus <dir> --ordering end --grain session --max-len 64 \
        --prepare-record <prepare run-dir> --n-layers 6 --d-model 256 --n-heads 8 --ff-mult 2 \
        --dropout 0.0 --rope-theta 10000 --lr 3e-4 --adam-beta1 0.9 --adam-beta2 0.95 \
        --warmup-steps 500 --lr-schedule cosine --weight-decay 0.1 --batch-size 256 --steps 12000 \
        --grad-clip 1.0 --amp bf16 --val-every 250 --val-batches 50 --checkpoint-every 250 \
        --model-choice argmin-val \
        --entropy-order 2 --seed 0 --device cuda --replica <tfvars name> --aws-profile-name default \
        --output-folder <run-dir>

Nothing causal-specific: plain maximum likelihood. One model per corpus,
trained on the `end-session` view and shared by every arm scored on that
corpus (D-SB-7). The backbone is a Hugging Face `LlamaForCausalLM` saved in
the library's directory format so the shipped tool loads it unchanged
(`backbone.py`). A checkpoint is written before every validation; `model/`
is the checkpoint `--model-choice` names — `argmin-val` (the validated step
with the smallest validation loss, ties to the smaller step; the plans'
2026-09-25 addendum) or `last` — and `results.json.model_sha256` (the sha256
of its saved weights) is the hash every arm and comparison record carries;
`model_choice` records the chosen step, its loss, the final loss and their
ratio. The oracle score is reported at the chosen checkpoint (and at the
last, `oracle_last`) against an independent order-k entropy floor and never
gated. A training run names a completed `prepare` record for the same
corpus and view and refuses to start without one (PRD scenario 47); the
provisioning replica and the AWS profile it was told are recorded
(scenarios 17, 25, 30).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .backbone import (
    TRAINING_JSON,
    architecture,
    architecture_sha256,
    build_model,
    loss,
    lr_at,
    model_sha256,
    n_params,
    save_model,
)
from .constants import (
    AMP_MODES,
    CHECKPOINT_DIR_FMT,
    DEVICES,
    GRAINS,
    LR_SCHEDULES,
    MODEL_CHOICES,
    MODEL_DIR,
    MODEL_WEIGHTS,
    ORACLE_IN_REGIME,
    ORDERINGS,
    PREPARE_JSON,
    RESULTS_JSON,
    RUN_DIR,
)
from .corpus import Corpus
from .data import SequenceStore, endless_batches
from .entropy import oracle_score, order_k_floor
from .log import log
from .record import RunRecord, read_json, sha256_file
from .vocab import Vocab


class PrepareRefusal(RuntimeError):
    pass


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_prepare_record(prepare_dir, corpus):
    """A completed `prepare` run for this corpus directory and view (scenario 47)."""
    prepare_dir = Path(prepare_dir)
    results = prepare_dir / RUN_DIR / RESULTS_JSON
    report = prepare_dir / PREPARE_JSON
    if not results.exists() or not report.exists():
        raise PrepareRefusal(
            f"{prepare_dir} holds no completed prepare record ({RUN_DIR}/{RESULTS_JSON} + {PREPARE_JSON})"
        )
    r = read_json(results)
    if r.get("status") != "ok":
        raise PrepareRefusal(f"prepare record {results} has status {r.get('status')!r}")
    p = read_json(report)
    if p.get("corpus_dir_name") != corpus.dir.name or p.get("view") != corpus.view:
        raise PrepareRefusal(
            f"prepare record is for {p.get('corpus_dir_name')!r} / {p.get('view')!r}, "
            f"this run is {corpus.dir.name!r} / {corpus.view!r}"
        )
    return {
        "path": str(prepare_dir),
        "prepare_json_sha256": sha256_file(report),
        "vocab_size": p["vocab"]["size"],
    }


@torch.no_grad()
def validate(model, store, batch_size, val_batches, device, amp):
    """Mean NLL per non-PAD target over the first `val_batches` length-sorted batches."""
    model.eval()
    total, count = 0.0, 0
    for k, idx in enumerate(store.batches(batch_size)):
        if k >= val_batches:
            break
        ids, pad = store.collate(idx, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(amp == "bf16")):
            vloss, n = loss(model, ids, pad)
        total += float(vloss) * n
        count += n
    model.train()
    return total / max(count, 1), count


def pretrain(args, out_dir):
    if args.model_choice == "argmin-val" and args.checkpoint_every != args.val_every:
        raise ModelChoiceRefusal(
            "--model-choice argmin-val needs --checkpoint-every == --val-every (every validated step must have a checkpoint)"
        )
    seed_all(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    if args.dropout != 0.0:
        raise ValueError(
            "--dropout must be 0.0: the library model has attention dropout only, and the "
            "pre-registered budget trains without dropout"
        )
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    prepare_record = require_prepare_record(args.prepare_record, corpus)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    if prepare_record["vocab_size"] != vocab.size:
        raise PrepareRefusal(
            f"prepare record vocab_size {prepare_record['vocab_size']} != view vocab_size {vocab.size}"
        )
    train = SequenceStore.from_corpus(corpus, "train", vocab, args.max_len)
    val = SequenceStore.from_corpus(corpus, "val", vocab, args.max_len)
    floor = order_k_floor(train, args.entropy_order, vocab.size)
    log_alphabet = math.log(len(vocab.predictable_ids))
    log(
        {
            "event": "data",
            "train_sequences": len(train),
            "train_tokens": train.n_tokens,
            "val_sequences": len(val),
            "truncated_train": train.n_truncated,
            "folded_train": train.n_folded,
            "vocab_size": vocab.size,
            "entropy_floor": floor["entropy_floor"],
            "entropy_order": args.entropy_order,
            "log_alphabet": log_alphabet,
        }
    )

    model = build_model(
        vocab.size,
        args.n_layers,
        args.d_model,
        args.n_heads,
        args.ff_mult,
        args.dropout,
        args.rope_theta,
        args.max_len,
    ).to(device)
    arch = architecture(model)
    log({"event": "model", "params": n_params(model), **arch})
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    stream = endless_batches(train, args.batch_size, args.seed)
    val_curve = []
    checkpoints = []
    t0 = time.monotonic()
    tokens_seen = 0
    running, running_n = 0.0, 0
    model.train()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup_steps, args.steps, args.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        if step % args.checkpoint_every == 0 and step > 0:
            path = out_dir / CHECKPOINT_DIR_FMT.format(step=step)
            sha = save_model(model, path, {"step": step, "vocab_size": vocab.size})
            checkpoints.append({"step": step, "dir": path.name, "sha256": sha})
        if step % args.val_every == 0 and step > 0:
            vloss, vn = validate(model, val, args.batch_size, args.val_batches, device, args.amp)
            val_curve.append({"step": step, "loss": vloss, "n_targets": vn})
            log(
                {
                    "event": "val",
                    "step": step,
                    "val_loss": vloss,
                    "train_loss": running / max(running_n, 1),
                    "lr": lr,
                    "tok_pos_per_s": tokens_seen / max(time.monotonic() - t0, 1e-9),
                }
            )
            running, running_n = 0.0, 0
        ids, pad = train.collate(next(stream), device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(args.amp == "bf16")):
            tloss, n = loss(model, ids, pad)
        opt.zero_grad(set_to_none=True)
        tloss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        tokens_seen += n
        running += float(tloss.detach())
        running_n += 1
    train_wall = time.monotonic() - t0
    vloss, vn = validate(model, val, args.batch_size, args.val_batches, device, args.amp)
    val_curve.append({"step": args.steps, "loss": vloss, "n_targets": vn})
    model_dir = out_dir / MODEL_DIR
    last_sha = save_model(model, model_dir, {"step": args.steps, "vocab_size": vocab.size})
    min_loss = min(v["loss"] for v in val_curve)
    choice = choose_model(args.model_choice, val_curve, checkpoints, args.steps, last_sha)
    if choice["step"] != args.steps:
        # the argmin checkpoint becomes model/ (same weights file, same hash as its checkpoint)
        shutil.rmtree(model_dir)
        shutil.copytree(out_dir / choice["checkpoint_dir"], model_dir)
        (model_dir / TRAINING_JSON).write_text(
            json.dumps({"step": choice["step"], "vocab_size": vocab.size}, sort_keys=True, indent=1)
        )
    model_sha = model_sha256(model_dir)
    if model_sha != choice["sha256"]:
        raise RuntimeError(
            f"model/ hashes to {model_sha} but the chosen checkpoint (step {choice['step']}) is {choice['sha256']}"
        )
    oracle = oracle_score(
        choice["val_loss"], floor["entropy_floor"], log_alphabet, ORACLE_IN_REGIME
    )
    oracle["entropy_order"] = args.entropy_order
    oracle["step"] = choice["step"]
    oracle_last = oracle_score(vloss, floor["entropy_floor"], log_alphabet, ORACLE_IN_REGIME)
    oracle_last["entropy_order"] = args.entropy_order
    oracle_last["step"] = args.steps
    results = {
        "model_dir": MODEL_DIR,
        "model_weights": f"{MODEL_DIR}/{MODEL_WEIGHTS}",
        "model_sha256": model_sha,
        "model_choice": {
            "choice": args.model_choice,
            "step": choice["step"],
            "val_loss": choice["val_loss"],
            "checkpoint_dir": choice["checkpoint_dir"],
            "last_step": args.steps,
            "last_val_loss": vloss,
            "last_sha256": last_sha,
            "val_loss_min": min_loss,
            "val_final_over_min": (vloss / min_loss) if min_loss > 0 else None,
        },
        "oracle_last": oracle_last,
        "params": n_params(model),
        "architecture": arch,
        "architecture_sha256": architecture_sha256(model),
        "vocab_size": vocab.size,
        "train_sequences": len(train),
        "train_tokens": train.n_tokens,
        "truncated_train": train.n_truncated,
        "folded_train": train.n_folded,
        "train_loss_last": running / max(running_n, 1),
        "val_curve": val_curve,
        "checkpoints": checkpoints,
        "val_loss_final": vloss,
        "val_loss_min": min_loss,
        "val_final_over_min": (vloss / min_loss) if min_loss > 0 else None,
        "oracle": oracle,
        "entropy": floor,
        "prepare_record": prepare_record,
        "budget": {
            "configs_tried_on_val": 1,
            "total_steps": args.steps,
            "wall_clock_s": round(train_wall, 3),
        },
        "tok_pos_per_s": tokens_seen / max(train_wall, 1e-9),
        "tokens_seen": tokens_seen,
    }
    log(
        {
            "event": "pretrain_done",
            "model_sha256": model_sha,
            "val_loss": vloss,
            "eps_hat": oracle["eps_hat"],
            "in_regime": oracle["in_regime"],
            "tok_pos_per_s": results["tok_pos_per_s"],
        }
    )
    return results


class ModelChoiceRefusal(ValueError):
    pass


def choose_model(model_choice, val_curve, checkpoints, total_steps, last_sha):
    """Which saved model becomes `model/` (plans/caps.md addendum 2026-09-25).

    `last`: the final step. `argmin-val`: the validated step with the smallest validation loss
    (ties -> the smaller step); every validated step before the last must have a checkpoint, so
    `--checkpoint-every` must equal `--val-every` under this choice."""
    if model_choice not in MODEL_CHOICES:
        raise ModelChoiceRefusal(f"--model-choice {model_choice!r} not in {MODEL_CHOICES}")
    last = {
        "step": total_steps,
        "val_loss": val_curve[-1]["loss"],
        "checkpoint_dir": MODEL_DIR,
        "sha256": last_sha,
    }
    if model_choice == "last":
        return last
    by_step = {c["step"]: c for c in checkpoints}
    best = min(val_curve, key=lambda v: (v["loss"], v["step"]))
    if best["step"] == total_steps:
        return last
    if best["step"] not in by_step:
        raise ModelChoiceRefusal(
            f"argmin-val step {best['step']} has no checkpoint; --checkpoint-every must equal --val-every"
        )
    return {
        "step": best["step"],
        "val_loss": best["loss"],
        "checkpoint_dir": by_step[best["step"]]["dir"],
        "sha256": by_step[best["step"]]["sha256"],
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--max-len", required=True, type=int)
    p.add_argument(
        "--prepare-record",
        required=True,
        help="a completed prepare run directory for this corpus and view (scenario 47)",
    )
    p.add_argument("--n-layers", required=True, type=int)
    p.add_argument("--d-model", required=True, type=int)
    p.add_argument("--n-heads", required=True, type=int)
    p.add_argument(
        "--ff-mult",
        required=True,
        type=float,
        help="SwiGLU hidden = ff_mult * d_model (2 gives the 4.2M anchor at |X| = 1000)",
    )
    p.add_argument("--dropout", required=True, type=float, help="attention dropout; must be 0.0")
    p.add_argument("--rope-theta", required=True, type=float)
    p.add_argument("--lr", required=True, type=float)
    p.add_argument("--adam-beta1", required=True, type=float)
    p.add_argument("--adam-beta2", required=True, type=float)
    p.add_argument("--warmup-steps", required=True, type=int)
    p.add_argument("--lr-schedule", required=True, choices=LR_SCHEDULES)
    p.add_argument("--weight-decay", required=True, type=float)
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--grad-clip", required=True, type=float, help="0 disables clipping")
    p.add_argument("--amp", required=True, choices=AMP_MODES)
    p.add_argument("--val-every", required=True, type=int)
    p.add_argument("--val-batches", required=True, type=int)
    p.add_argument("--checkpoint-every", required=True, type=int)
    p.add_argument(
        "--model-choice",
        required=True,
        choices=MODEL_CHOICES,
        help="which checkpoint becomes model/: 'argmin-val' needs --checkpoint-every == --val-every (plans/caps.md addendum 2026-09-25)",
    )
    p.add_argument(
        "--entropy-order",
        required=True,
        type=int,
        help="order k of the n-gram entropy floor (D-SB-13)",
    )
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, choices=DEVICES)
    p.add_argument(
        "--replica",
        required=True,
        help="the dated provisioning replica this run executes under, by file name (scenario 25); 'local-test' for a local test",
    )
    p.add_argument(
        "--aws-profile-name",
        required=True,
        help="the AWS profile name the run executes under (scenario 30); 'none' for a local test",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "pretrain", vars(args)) as rec:
        rec.note(replica=args.replica, aws_profile=args.aws_profile_name)
        results = pretrain(args, rec.out_dir)
        rec.note(tok_pos_per_s=results["tok_pos_per_s"])
        rec.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
