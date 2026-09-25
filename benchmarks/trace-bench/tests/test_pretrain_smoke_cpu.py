# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""`pretrain` end to end on the fixture, CPU, seconds: a Hugging Face model
directory whose weights-file sha256 the record carries, checkpoints before
validations, the oracle score against the independent floor, the budget
triple, the prepare-record requirement (PRD scenario 47) and the provisioning
facts (scenarios 17, 30)."""

import pytest
import torch
from fixture_corpus import fixture_corpus

from seq2causebench import prepare as prep
from seq2causebench import pretrain as pt
from seq2causebench.backbone import load_model, model_sha256
from seq2causebench.constants import RESULTS_JSON, RUN_DIR
from seq2causebench.record import read_json

ARGS = [
    "--ordering",
    "end",
    "--grain",
    "session",
    "--max-len",
    "32",
    "--n-layers",
    "2",
    "--d-model",
    "32",
    "--n-heads",
    "4",
    "--ff-mult",
    "2",
    "--dropout",
    "0.0",
    "--rope-theta",
    "10000",
    "--lr",
    "3e-3",
    "--adam-beta1",
    "0.9",
    "--adam-beta2",
    "0.95",
    "--warmup-steps",
    "5",
    "--lr-schedule",
    "cosine",
    "--weight-decay",
    "0.01",
    "--batch-size",
    "16",
    "--steps",
    "40",
    "--grad-clip",
    "1.0",
    "--amp",
    "none",
    "--val-every",
    "20",
    "--val-batches",
    "4",
    "--checkpoint-every",
    "20",
    "--model-choice",
    "last",
    "--entropy-order",
    "2",
    "--seed",
    "0",
    "--device",
    "cpu",
    "--replica",
    "local-test",
    "--aws-profile-name",
    "none",
]


def _prepare(tmp_path, grain="session"):
    out = tmp_path / f"prep-{grain}"
    assert (
        prep.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--ordering",
                "end",
                "--grain",
                grain,
                "--max-len",
                "32",
                "--entropy-order",
                "2",
                "--output-folder",
                str(out),
            ]
        )
        == 0
    )
    return out


def test_pretrain_smoke(tmp_path):
    prep_dir = _prepare(tmp_path)
    out = tmp_path / "pre"
    assert (
        pt.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--prepare-record",
                str(prep_dir),
                *ARGS,
                "--output-folder",
                str(out),
            ]
        )
        == 0
    )
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "ok" and (out / "model" / "model.safetensors").exists()
    assert (out / "model" / "config.json").exists()
    assert r["model_sha256"] == model_sha256(out / "model")
    assert [c["step"] for c in r["checkpoints"]] == [20] and (
        out / "checkpoint-0000020" / "model.safetensors"
    ).exists()
    assert [v["step"] for v in r["val_curve"]] == [20, 40]
    assert r["val_curve"][-1]["loss"] < r["val_curve"][0]["loss"] + 0.5  # not diverging
    o = r["oracle"]
    assert set(o) >= {
        "eps_hat",
        "val_loss",
        "entropy_floor",
        "log_alphabet",
        "in_regime",
        "entropy_order",
    }
    assert o["entropy_floor"] == r["entropy"]["entropy_floor"] and o["entropy_order"] == 2
    assert 0 < o["entropy_floor"] < o["log_alphabet"]
    assert r["budget"] == {
        "configs_tried_on_val": 1,
        "total_steps": 40,
        "wall_clock_s": r["budget"]["wall_clock_s"],
    }
    assert r["tok_pos_per_s"] > 0 and r["params"] > 0 and r["vocab_size"] == 14
    assert r["architecture"]["hidden_size"] == 32 and r["architecture"]["vocab_size"] == 14
    assert r["architecture"]["pad_token_id"] == 0 if "pad_token_id" in r["architecture"] else True
    assert r["val_final_over_min"] >= 1.0 and r["prepare_record"]["path"] == str(prep_dir)
    mc = r["model_choice"]
    assert mc["choice"] == "last" and mc["step"] == 40 and mc["last_step"] == 40
    assert (
        mc["last_sha256"] == r["model_sha256"]
        and mc["val_final_over_min"] == r["val_final_over_min"]
    )
    assert r["oracle"]["step"] == 40 and r["oracle_last"] == r["oracle"]
    m, extra = load_model(out / "model")
    assert extra["step"] == 40 and m.config.hidden_size == 32 and m.config.pad_token_id == 0
    assert (
        m.config.bos_token_id == 1 and m.config.eos_token_id == 2 and m.config.tie_word_embeddings
    )
    meta = read_json(out / RUN_DIR / "run_meta.json")
    assert meta["tok_pos_per_s"] == r["tok_pos_per_s"] and meta["status"] == "ok"
    assert meta["replica"] == "local-test" and meta["aws_profile"] == "none"
    assert meta["seq2cause"].startswith("0.1.")


def test_pretrain_refuses_without_a_matching_prepare_record(tmp_path):
    out = tmp_path / "pre"
    # no record at all
    with pytest.raises(pt.PrepareRefusal):
        pt.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--prepare-record",
                str(tmp_path / "nowhere"),
                *ARGS,
                "--output-folder",
                str(out),
            ]
        )
    assert read_json(out / RUN_DIR / RESULTS_JSON)["status"] == "failed"
    # a record for the other view
    prep_req = _prepare(tmp_path, "request")
    with pytest.raises(pt.PrepareRefusal):
        pt.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--prepare-record",
                str(prep_req),
                *ARGS,
                "--output-folder",
                str(tmp_path / "pre2"),
            ]
        )


def test_seeded_runs_are_reproducible_on_cpu(tmp_path):
    prep_dir = _prepare(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    args = [*ARGS]
    args[args.index("--steps") + 1] = "10"
    for out in (a, b):
        assert (
            pt.main(
                [
                    "--corpus",
                    str(fixture_corpus("latent")),
                    "--prepare-record",
                    str(prep_dir),
                    *args,
                    "--output-folder",
                    str(out),
                ]
            )
            == 0
        )
    ma, _ = load_model(a / "model")
    mb, _ = load_model(b / "model")
    for pa, pb in zip(ma.parameters(), mb.parameters(), strict=True):
        assert torch.equal(pa, pb)
    assert model_sha256(a / "model") == model_sha256(b / "model")


def test_model_hash_is_the_weights_file_only(tmp_path):
    """The hash is stable across saves of the same weights and ignores config.json."""
    prep_dir = _prepare(tmp_path)
    out = tmp_path / "pre"
    args = [*ARGS]
    args[args.index("--steps") + 1] = "5"
    assert (
        pt.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--prepare-record",
                str(prep_dir),
                *args,
                "--output-folder",
                str(out),
            ]
        )
        == 0
    )
    m, _ = load_model(out / "model")
    from seq2causebench.backbone import save_model

    again = save_model(m, tmp_path / "again")
    assert again == model_sha256(out / "model")
    (tmp_path / "again" / "config.json").write_text("{}")
    assert model_sha256(tmp_path / "again") == again


def test_argmin_val_selection(tmp_path):
    """`--model-choice argmin-val` makes model/ the validated step with the smallest validation
    loss (plans/caps.md addendum 2026-09-25): its hash equals that checkpoint's, the oracle is
    taken there, and the last checkpoint's facts stay recorded beside it."""
    prep_dir = _prepare(tmp_path)
    out = tmp_path / "pre"
    args = [*ARGS]
    args[args.index("--model-choice") + 1] = "argmin-val"
    args[args.index("--steps") + 1] = "60"
    assert (
        pt.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--prepare-record",
                str(prep_dir),
                *args,
                "--output-folder",
                str(out),
            ]
        )
        == 0
    )
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    curve = {v["step"]: v["loss"] for v in r["val_curve"]}
    assert sorted(curve) == [20, 40, 60]
    best = min(curve, key=lambda k: (curve[k], k))
    mc = r["model_choice"]
    assert mc["choice"] == "argmin-val" and mc["step"] == best and mc["val_loss"] == curve[best]
    assert mc["last_step"] == 60 and mc["last_val_loss"] == curve[60]
    assert r["model_sha256"] == model_sha256(out / "model")
    if best != 60:
        ck = next(c for c in r["checkpoints"] if c["step"] == best)
        assert r["model_sha256"] == ck["sha256"] == model_sha256(out / ck["dir"])
        assert r["model_sha256"] != mc["last_sha256"]
    else:
        assert r["model_sha256"] == mc["last_sha256"]
    _, extra = load_model(out / "model")
    assert extra["step"] == best
    assert r["oracle"]["step"] == best and r["oracle"]["val_loss"] == curve[best]
    assert r["oracle_last"]["step"] == 60 and r["oracle_last"]["val_loss"] == curve[60]
    assert r["val_loss_min"] == curve[best] and mc["val_loss_min"] == r["val_loss_min"]


def test_argmin_val_needs_matching_checkpoint_and_val_periods(tmp_path):
    prep_dir = _prepare(tmp_path)
    args = [*ARGS]
    args[args.index("--model-choice") + 1] = "argmin-val"
    args[args.index("--checkpoint-every") + 1] = "10"
    with pytest.raises(pt.ModelChoiceRefusal, match="checkpoint-every"):
        pt.main(
            [
                "--corpus",
                str(fixture_corpus("latent")),
                "--prepare-record",
                str(prep_dir),
                *args,
                "--output-folder",
                str(tmp_path / "pre"),
            ]
        )
