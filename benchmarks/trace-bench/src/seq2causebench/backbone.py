# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The backbone the shipped tool loads: a Hugging Face `LlamaForCausalLM` built
from a `LlamaConfig`, saved per checkpoint in the Hugging Face directory
format (D-SB-13).

The author's own research script trains exactly this class and saves it with
`save_pretrained`; the CLI loads it with `AutoModelForCausalLM.from_pretrained`;
the saliency read-out needs the embedding layer, an embeddings-input forward,
the device and the hidden size, and the Shapley read-out the padding id — all
native on the library model. The sibling's decoder is Llama-shaped (pre-norm
RMSNorm, RoPE, SwiGLU, tied embedding), so its sizing anchors transfer and
only the class changes.

The model hash is the sha256 of the saved weights file (`model.safetensors`),
never of `config.json`, which embeds a library version. The architecture is
hashed separately from the fields that define it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from .constants import BOS, EOS, MODEL_WEIGHTS, PAD

ARCHITECTURE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_parameters",  # transformers >= 5 nests rope_theta here
    "rms_norm_eps",
    "tie_word_embeddings",
    "attention_dropout",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
)
TRAINING_JSON = "training.json"


def build_config(vocab_size, n_layers, d_model, n_heads, ff_mult, dropout, rope_theta, max_len):
    """`max_len` real tokens plus BOS and EOS bound the positions."""
    return LlamaConfig(
        vocab_size=int(vocab_size),
        hidden_size=int(d_model),
        intermediate_size=int(round(ff_mult * d_model)),
        num_hidden_layers=int(n_layers),
        num_attention_heads=int(n_heads),
        num_key_value_heads=int(n_heads),
        max_position_embeddings=int(max_len) + 2,
        rope_theta=float(rope_theta),
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        tie_word_embeddings=True,
        attention_dropout=float(dropout),
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )


def build_model(vocab_size, n_layers, d_model, n_heads, ff_mult, dropout, rope_theta, max_len):
    """A fresh `LlamaForCausalLM`; seed the torch RNG before calling for a reproducible init."""
    return LlamaForCausalLM(
        build_config(vocab_size, n_layers, d_model, n_heads, ff_mult, dropout, rope_theta, max_len)
    )


def architecture(model):
    cfg = model.config.to_dict()
    return {k: cfg[k] for k in ARCHITECTURE_FIELDS}


def architecture_sha256(model):
    return hashlib.sha256(json.dumps(architecture(model), sort_keys=True).encode()).hexdigest()


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def logits_fp32(model, ids, pad_mask):
    """`[B, L, V]` float32 logits with PAD masked from attention (bf16 autocast leaves
    the library's logits in bf16; every consumer here wants float32)."""
    out = model(input_ids=ids, attention_mask=(~pad_mask).long())
    return out.logits.float()


def loss(model, ids, pad_mask):
    """Mean next-token NLL in nats over non-PAD targets (positions 1..L-1), `(loss, n)`."""
    logits = logits_fp32(model, ids, pad_mask)[:, :-1]
    target = ids[:, 1:]
    keep = ~pad_mask[:, 1:]
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    )
    keep = keep.reshape(-1)
    return (nll * keep).sum() / keep.sum(), int(keep.sum())


@torch.no_grad()
def log_probs_at(model, ids, pad_mask):
    """`[B, L]` float32: entry `t >= 1` is `log p(ids[t] | ids[<t])`; entry 0 is 0."""
    lp = torch.log_softmax(logits_fp32(model, ids, pad_mask), dim=-1)
    got = lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    out = torch.zeros(ids.shape, dtype=torch.float32, device=ids.device)
    out[:, 1:] = got
    return out


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_model(model, directory, extra=None):
    """`save_pretrained` into `directory` (+ `training.json` with `extra`); returns the
    sha256 of the saved weights file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    if extra is not None:
        (directory / TRAINING_JSON).write_text(json.dumps(extra, sort_keys=True, indent=1))
    return model_sha256(directory)


def model_sha256(directory):
    weights = Path(directory) / MODEL_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found: not a saved backbone directory")
    return sha256_of(weights)


def load_model(directory, device="cpu"):
    """`(model.eval() on device, extra)`; the raw library model, which every shipped
    read-out and the shipped adapter accept."""
    directory = Path(directory)
    model = LlamaForCausalLM.from_pretrained(directory).to(device).eval()
    extra_path = directory / TRAINING_JSON
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else {}
    return model, extra


def lr_at(step, lr, warmup_steps, total_steps, schedule):
    """Linear warmup, then cosine decay to zero at `total_steps` or a constant."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return lr
    if schedule == "cosine":
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return 0.5 * lr * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}")


__all__ = [
    "build_config",
    "build_model",
    "architecture",
    "architecture_sha256",
    "n_params",
    "logits_fp32",
    "loss",
    "log_probs_at",
    "sha256_of",
    "save_model",
    "model_sha256",
    "load_model",
    "lr_at",
]
