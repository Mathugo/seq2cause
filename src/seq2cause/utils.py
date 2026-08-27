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


def get_available_memory_bytes(device: torch.device | str) -> int | None:
    """Best-effort query of the currently AVAILABLE (free) memory on
    `device`, in bytes.

    Returns `None` if this can't be determined for the given device/backend,
    rather than 0 -- callers should treat `None` as "unknown, skip the
    check", not "no memory available":
      - CUDA: `torch.cuda.mem_get_info` (always available).
      - CPU: requires the optional `psutil` package; `None` if not
        installed (it isn't one of seq2cause's own dependencies).
      - MPS and any other backend: PyTorch has no public "free VRAM" query
        for MPS as of this writing (only allocated-memory counters), so
        this always returns `None` there.
    """
    device = torch.device(device)
    if device.type == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return free_bytes
    if device.type == "cpu":
        try:
            import psutil
        except ImportError:
            return None
        return psutil.virtual_memory().available
    return None


def check_memory_budget(
    estimated_bytes: int, device: torch.device | str, safety_margin: float = 0.8
) -> None:
    """Raises `MemoryError` BEFORE attempting an allocation likely to run
    out of memory, instead of letting a real allocation fail deep inside a
    forward pass with a much less actionable "CUDA out of memory" (or
    similar) error.

    Compares `estimated_bytes` (e.g. from `estimate_tensor_bytes`) against
    the currently available memory on `device` (`get_available_memory_bytes`).
    If that can't be determined for this device/backend (e.g. MPS, or CPU
    without `psutil` installed), this silently does nothing -- it can only
    catch what it can actually measure.

    Args:
        estimated_bytes: the estimated size of the tensor about to be
            allocated (see `estimate_tensor_bytes`).
        device: the device the allocation would happen on.
        safety_margin: fraction of *available* memory this allocation is
            allowed to use (default 0.8, i.e. refuse if the estimate would
            use more than 80% of what's currently free). Lower this if
            other processes/allocations also need headroom.
    """
    available = get_available_memory_bytes(device)
    if available is None:
        return
    budget = available * safety_margin
    if estimated_bytes > budget:
        raise MemoryError(
            f"Estimated allocation ({format_bytes(estimated_bytes)}) exceeds "
            f"{safety_margin:.0%} of the memory currently available on {device} "
            f"({format_bytes(available)} free). Reduce n_particles/batch_size, "
            "or use compute_cmi_matrix_sparse for long sequences with a "
            "known or assumed memory bound, before this actually allocates and OOMs."
        )


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
