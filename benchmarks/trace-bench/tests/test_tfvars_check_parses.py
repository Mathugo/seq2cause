# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""`scripts/tfvars_check.py` parses a replica's command with the real parsers,
refuses a missing knob or a results push from the public repository, and
compares a run record to the replica (the replica and the run agree by construction)."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tfvars_check", REPO / "scripts" / "tfvars_check.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

GOOD = (
    "python -m seq2causebench.prepare --corpus corpora/x --ordering end --grain session --max-len 64 "
    "--entropy-order 2 --output-folder out/prep"
)


def _tfvars(tmp_path, command, push="false", repo="Mathugo/seq2cause"):
    p = tmp_path / "r.tfvars"
    p.write_text(
        f'# header with "quotes" and a # sign\nrun_name = "x"\ngithub_repo = "{repo}"\n'
        f'output_local_dir = "out"\nrepo_subdir = "benchmarks/trace-bench"\npush_results = {push} # comment\nsetup_command = "pip install -e ."\n'
        f'train_command = "{command}"\nmax_runtime_hours = 8\n'
    )
    return p


def test_good_replica_parses(tmp_path):
    score = "python -m tracebench.score --corpus c --prediction p --grain request --out o"
    tf, parsed, problems, skipped = tc.check(_tfvars(tmp_path, GOOD + " && " + score))
    assert problems == [] and skipped == [
        score
    ]  # the benchmark's own command is chained, not checked
    assert [m for m, _ in parsed] == ["seq2causebench.prepare"]
    assert parsed[0][1]["max_len"] == 64 and parsed[0][1]["grain"] == "session"
    assert tf["push_results"] is False and tf["max_runtime_hours"] == 8
    assert tc.main([str(_tfvars(tmp_path, GOOD))]) == 0


def test_missing_knob_and_public_push_are_refused(tmp_path):
    _, _, problems, _ = tc.check(_tfvars(tmp_path, GOOD.replace(" --entropy-order 2", "")))
    assert any(p.startswith("argparse rejected") for p in problems)
    _, _, problems, _ = tc.check(_tfvars(tmp_path, GOOD, push="true"))
    assert "push_results must be false for the public repository" in problems
    assert tc.main([str(_tfvars(tmp_path, GOOD, push="true"))]) == 1


def test_args_diff_against_a_record(tmp_path):
    tf, parsed, _, _ = tc.check(_tfvars(tmp_path, GOOD))
    run = tmp_path / "run-dir"
    (run / "run").mkdir(parents=True)
    rec = {"command": "prepare", "arguments": dict(parsed[0][1]), "argv": []}
    (run / "run" / "arguments.json").write_text(json.dumps(rec))
    assert tc.args_diff(parsed, run) == {}
    rec["arguments"]["max_len"] = 32
    (run / "run" / "arguments.json").write_text(json.dumps(rec))
    assert tc.args_diff(parsed, run) == {"max_len": (64, 32)}
    assert tc.main([str(_tfvars(tmp_path, GOOD)), "--args-diff", str(run)]) == 1
