"""The pre-registrations exist, are dated, and agree with the registry: every arm is named in
`plans/rca-arms.md`, the bin-width rule's N and widths match `constants.BIN_WIDTH_S`, both code
pins appear, and RUN.md carries the three deviations the plan numbers."""

import re
from pathlib import Path

from tracebenchrca.constants import (
    ARMS,
    BARO_COMMIT,
    BIN_WIDTH_S,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    CAUSALRCA_COMMIT,
    RCAEVAL_COMMIT,
)

BENCH = Path(__file__).resolve().parents[1]
DATE_RE = re.compile(r"pre-registered (20\d\d-\d\d-\d\d)")


def test_rca_arms_plan_is_dated_and_names_every_registered_fact():
    text = (BENCH / "plans" / "rca-arms.md").read_text(encoding="utf-8")
    assert DATE_RE.search(text), "the plan must say when it was pre-registered"
    for arm in ARMS:
        assert f"`{arm}`" in text, arm
    assert "N = 10" in text
    for rung, w in BIN_WIDTH_S.items():
        assert f"{rung} {w} s" in text, (rung, w)
    for sha in (BARO_COMMIT, RCAEVAL_COMMIT, CAUSALRCA_COMMIT):
        assert sha in text, sha
    assert f"B = {BOOTSTRAP_B}" in text and f"seed {BOOTSTRAP_SEED}" in text
    for dev in ("D-TBR-1", "D-TBR-2", "D-TBR-3"):
        assert dev in text
    assert "no change point" in text  # the BOCPD empty-anomalies rule is registered, not improvised
    assert "`time`" in text  # the suite's identity preprocess leaves the time column; its removal is registered


def test_controls_plan_is_dated_and_marks_the_open_rule():
    text = (BENCH / "plans" / "controls.md").read_text(encoding="utf-8")
    assert DATE_RE.search(text)
    assert "`control/random`" in text and "`control/rule`" in text
    assert "min(k, n) / n" in text
    assert "OPEN" in text  # the rule control awaits the owner (publication plan Q8)


def test_run_md_numbers_the_deviations_before_the_first_run():
    text = (BENCH / "RUN.md").read_text(encoding="utf-8")
    for dev in ("D-TBR-1", "D-TBR-2", "D-TBR-3"):
        assert f"| {dev} |" in text, dev
    assert "## Run registry" in text
