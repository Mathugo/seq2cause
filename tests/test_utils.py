import torch

from seq2cause.utils import next_token_collate


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
