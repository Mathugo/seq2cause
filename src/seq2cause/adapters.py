"""Adapters that let real, trained (or pretrained) HuggingFace causal LMs be
used with `seq2cause.diagnostics`/`seq2cause.sampling` wherever the codebase
otherwise expects `seq2cause.scm.NonlinearSCM` (the synthetic oracle model
used for testing/validation) -- namely, a `.vocab_size` attribute and a
HF-style `forward(input_ids=...)` call.
"""

from __future__ import annotations

import torch

__all__ = ["HFModelAdapter"]


class HFModelAdapter:
    """Wraps any HuggingFace causal LM (`GPT2LMHeadModel`, `LlamaForCausalLM`,
    your own fine-tuned checkpoint, ...) so it satisfies the small surface
    `compare_intervention_strategies`/`do_interventions` expect.

    This is the piece that lets you run seq2cause against a real model on
    your own event sequences, with no synthetic generator and no known
    ground-truth causal graph required -- see the README Quick Start.
    """

    def __init__(self, model, vocab_size: int | None = None):
        self._model = model
        self.vocab_size = vocab_size if vocab_size is not None else model.config.vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs) -> dict:
        # `do_interventions`/diagnostics feed tensors with arbitrary leading
        # (particle, row, ...) dims; a real HF model's forward only accepts
        # 2D [batch, seq_len] input_ids, so flatten/restore around the call.
        *lead_dims, seq_len = input_ids.shape
        flat_input_ids = input_ids.reshape(-1, seq_len)
        with torch.no_grad():
            out = self._model(input_ids=flat_input_ids)
        logits = out.logits.reshape(*lead_dims, seq_len, -1)
        return {"logits": logits}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
