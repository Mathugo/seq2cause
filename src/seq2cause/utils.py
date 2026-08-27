from __future__ import annotations

import math

import torch


def estimate_tensor_bytes(*dims: int, dtype_bytes: int = 4) -> int:
    """Estimates the memory footprint (in bytes) of a dense tensor of shape
    `dims`, assuming `dtype_bytes` bytes per element (default 4, i.e.
    float32/int32 -- halve it for float16/bfloat16).

    This is a rough, allocation-count estimate (the single largest tensor
    materialized during a computation, e.g. the do-intervention logits of
    shape `[batch_size, n_particles, L-context, L, vocab_size]`) -- it does
    NOT account for autograd's saved-activation memory, optimizer states, or
    framework/CUDA-allocator overhead, so treat it as a lower bound / rough
    order-of-magnitude guide for "will this batch fit in VRAM/RAM", not an
    exact figure.
    """
    numel = 1
    for d in dims:
        numel *= max(int(d), 0)
    return numel * dtype_bytes


def format_bytes(n_bytes: int) -> str:
    """Formats a byte count as a human-readable string (e.g. `"512.0 MB"`)."""
    if n_bytes <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    exponent = min(int(math.log(n_bytes, 1024)), len(units) - 1)
    value = n_bytes / (1024**exponent)
    return f"{value:.1f} {units[exponent]}"


def next_token_collate(batch, device: str | None = None):
    """
    Standard next token collate function to create
    batches of input_ids and attention_mask tensors for the model.

    Args:
        batch: A list of dictionaries, each containing 'input_ids' and 'attention_mask'.
        device: The target device to move the tensors to. Defaults to `"cuda"` if a GPU is
            available, otherwise `"cpu"` -- never hardcode `"cuda"`, since that raises on any
            CPU-only machine (this library is regularly used/tested on CPU).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_ids_batch = torch.tensor(input_ids).long().to(device)
    attention_mask_batch = torch.tensor(attention_mask).long().to(device)
    # Create the new batch dictionary
    new_batch = {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
    }
    return new_batch
