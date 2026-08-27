import torch

from seq2cause.utils import estimate_tensor_bytes, format_bytes, next_token_collate


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
