import pytest
import torch

from seq2cause.utils import (
    check_memory_budget,
    estimate_tensor_bytes,
    format_bytes,
    get_available_memory_bytes,
    next_token_collate,
)


def test_next_token_collate_batches_input_ids_and_attention_mask():
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 0]},
    ]
    out = next_token_collate(batch)

    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    assert torch.equal(out["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(out["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))


def test_next_token_collate_defaults_to_cpu_when_no_cuda():
    batch = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]
    out = next_token_collate(batch)
    assert out["input_ids"].device.type == "cpu"


def test_next_token_collate_respects_explicit_device():
    batch = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]
    out = next_token_collate(batch, device="cpu")
    assert out["input_ids"].device.type == "cpu"
    assert out["input_ids"].dtype == torch.long


def test_estimate_tensor_bytes_matches_numel_times_dtype_size():
    assert estimate_tensor_bytes(2, 3, 4, dtype_bytes=4) == 2 * 3 * 4 * 4
    assert estimate_tensor_bytes(10, dtype_bytes=2) == 20


def test_estimate_tensor_bytes_handles_zero_or_negative_dims():
    assert estimate_tensor_bytes(0, 5, dtype_bytes=4) == 0
    assert estimate_tensor_bytes(-1, 5, dtype_bytes=4) == 0


def test_format_bytes_picks_sensible_units():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512.0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1024**2) == "1.0 MB"
    assert format_bytes(1024**3 * 2) == "2.0 GB"


def test_get_available_memory_bytes_uses_cuda_mem_get_info(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (123, 456))
    assert get_available_memory_bytes("cuda") == 123


def test_get_available_memory_bytes_uses_psutil_for_cpu(monkeypatch):
    import sys
    import types

    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=999)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert get_available_memory_bytes("cpu") == 999


def _deny_psutil_import(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_get_available_memory_bytes_returns_none_without_psutil(monkeypatch):
    _deny_psutil_import(monkeypatch)
    assert get_available_memory_bytes("cpu") is None


def test_check_memory_budget_raises_when_estimate_exceeds_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1000, 2000))

    check_memory_budget(700, "cuda", safety_margin=0.8)  # 700 < 0.8 * 1000, fine
    with pytest.raises(MemoryError, match="exceeds"):
        check_memory_budget(900, "cuda", safety_margin=0.8)  # 900 > 0.8 * 1000


def test_check_memory_budget_is_a_noop_when_available_memory_is_unknown(monkeypatch):
    _deny_psutil_import(monkeypatch)
    check_memory_budget(10**18, "cpu")  # would clearly overflow if it could check
