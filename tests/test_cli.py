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


def test_main_requires_vocab_size_without_model(tmp_path):
    dataset_path = tmp_path / "events.txt"
    dataset_path.write_text("1 2 3\n")

    try:
        main(["--dataset", str(dataset_path)])
    except SystemExit as exc:
        assert "vocab-size" in str(exc)
    else:
        raise AssertionError("expected SystemExit when --vocab-size and --model are both omitted")
