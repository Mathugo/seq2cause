"""Tests for `seq2cause.cli`: the tokenized-dataset loader and a full
end-to-end run through `main()`."""

from __future__ import annotations

import torch

from seq2cause.cli import load_tokenized_dataset, main


def test_load_tokenized_dataset_from_text(tmp_path):
    path = tmp_path / "events.txt"
    path.write_text("1 2 3 4\n5, 6 ,7,8\n\n9 10\n")

    sequences = load_tokenized_dataset(path)

    assert len(sequences) == 3
    assert torch.equal(sequences[0], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(sequences[1], torch.tensor([5, 6, 7, 8]))
    assert torch.equal(sequences[2], torch.tensor([9, 10]))
    assert all(seq.dtype == torch.long for seq in sequences)


def test_load_tokenized_dataset_from_pt_tensor(tmp_path):
    path = tmp_path / "events.pt"
    torch.save(torch.randint(0, 50, (4, 10)), path)

    sequences = load_tokenized_dataset(path)

    assert len(sequences) == 4
    assert all(seq.shape == (10,) for seq in sequences)


def test_load_tokenized_dataset_from_pt_list_of_variable_length_sequences(tmp_path):
    path = tmp_path / "events.pt"
    raw = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    torch.save(raw, path)

    sequences = load_tokenized_dataset(path)

    assert len(sequences) == 2
    assert torch.equal(sequences[0], torch.tensor([1, 2, 3]))
    assert torch.equal(sequences[1], torch.tensor([4, 5]))


def test_load_tokenized_dataset_from_npy(tmp_path):
    import numpy as np

    path = tmp_path / "events.npy"
    np.save(path, np.random.randint(0, 50, size=(3, 6)))

    sequences = load_tokenized_dataset(path)

    assert len(sequences) == 3
    assert all(seq.shape == (6,) for seq in sequences)


def test_main_runs_end_to_end_with_default_model(tmp_path, capsys):
    torch.manual_seed(0)
    dataset_path = tmp_path / "events.txt"
    seqs = torch.randint(0, 30, (2, 16))
    dataset_path.write_text("\n".join(" ".join(str(t) for t in row.tolist()) for row in seqs))
    output_path = tmp_path / "graphs.pt"

    main(
        [
            "--dataset",
            str(dataset_path),
            "--vocab-size",
            "30",
            "--context-len",
            "3",
            "--n-particles",
            "8",
            "--seed",
            "0",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert "Done in" in captured.out
    assert "candidate causal edges" in captured.out

    graphs = torch.load(output_path, weights_only=True)
    assert len(graphs) == 2
    for graph in graphs:
        assert graph.shape == (13, 13)
        assert graph.dtype == torch.bool


def test_main_infers_vocab_size_from_dataset_when_model_and_vocab_size_omitted(tmp_path, capsys):
    dataset_path = tmp_path / "events.txt"
    dataset_path.write_text("1 2 3 4 5\n4 29 7 12 3\n")

    main(["--dataset", str(dataset_path), "--n-particles", "4", "--context-len", "2", "--seed", "0"])

    captured = capsys.readouterr()
    # Max token id across the dataset is 29 -> inferred vocab_size = 30.
    assert "vocab_size=30" in captured.out
    assert "Done in" in captured.out


class _FakeConfig:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class _FakeCausalLM(torch.nn.Module):
    """Minimal stand-in for a HuggingFace causal LM, avoiding a real Hub
    download in tests."""

    def __init__(self, vocab_size):
        super().__init__()
        self.config = _FakeConfig(vocab_size)
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids, **kwargs):
        logits = self.head(self.embed(input_ids))
        return type("FakeCausalLMOutput", (), {"logits": logits})()


def test_main_uses_models_own_vocab_size_and_warns_on_conflicting_flag(tmp_path, capsys, monkeypatch):
    torch.manual_seed(0)
    dataset_path = tmp_path / "events.txt"
    seqs = torch.randint(0, 50, (2, 12))
    dataset_path.write_text("\n".join(" ".join(str(t) for t in row.tolist()) for row in seqs))

    fake_model = _FakeCausalLM(vocab_size=50)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: fake_model
    )

    main(
        [
            "--dataset", str(dataset_path),
            "--model", "fake/tiny-model",
            "--vocab-size", "999",  # deliberately conflicting -- should be ignored.
            "--n-particles", "4",
            "--context-len", "2",
            "--seed", "0",
        ]
    )

    captured = capsys.readouterr()
    assert "--vocab-size 999 ignored" in captured.out
    assert "vocab_size (50)" in captured.out
    assert "Done in" in captured.out
