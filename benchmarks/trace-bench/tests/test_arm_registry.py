"""PRD Arm Register (amended 2026-09-24) and scenario 39: the registry, its
paths and cuts, and the four-slot cell key."""

import pytest

from seq2causebench import arms
from seq2causebench.constants import (
    AGGREGATIONS,
    ARM_CLI,
    ARM_CLI_ATOMIC,
    ARM_CORE,
    ARM_GRANGER,
    ARM_SALIENCY,
    ARM_SHAPLEY,
    ARMS,
    BASELINE_ARMS,
    CUT_FROZEN,
    CUT_SHIPPED,
    GRAINS,
    PATH_FIXED,
    PATH_FIXED_KL,
    PATH_NONE,
    PATH_SHIPPED,
    PATH_SPEC,
    PATHS,
    REFERENCE_ARMS,
)


def test_six_arms_three_reference_three_baseline():
    assert set(ARMS) == set(REFERENCE_ARMS) | set(BASELINE_ARMS)
    assert len(ARMS) == 6
    assert all(ARMS[a]["class"] == "reference" for a in REFERENCE_ARMS)
    assert all(ARMS[a]["class"] == "baseline" for a in BASELINE_ARMS)
    assert AGGREGATIONS == ("max",)


def test_reference_arms_run_every_path_and_cli_arms_both_cuts():
    for arm in REFERENCE_ARMS:
        assert ARMS[arm]["paths"] == (PATH_SHIPPED, PATH_FIXED_KL, PATH_FIXED)
    assert ARMS[ARM_CORE]["cuts"] == (CUT_FROZEN,)
    assert ARMS[ARM_CLI]["cuts"] == ARMS[ARM_CLI_ATOMIC]["cuts"] == (CUT_FROZEN, CUT_SHIPPED)
    assert ARMS[ARM_GRANGER]["paths"] == (PATH_SHIPPED, PATH_FIXED)
    assert ARMS[ARM_SALIENCY]["paths"] == ARMS[ARM_SHAPLEY]["paths"] == (PATH_NONE,)


def test_granger_shares_the_core_probe():
    assert arms.probe_of(ARM_GRANGER) == arms.probe_of(ARM_CORE) == "core"
    assert arms.pass_of(ARM_GRANGER, PATH_SHIPPED) == arms.pass_of(ARM_CORE, PATH_SHIPPED)
    assert arms.pass_of(ARM_CORE, PATH_FIXED_KL) == arms.pass_of(
        ARM_CORE, PATH_SHIPPED
    )  # same forward, same draws
    assert arms.pass_of(ARM_CORE, PATH_FIXED) != arms.pass_of(
        ARM_CORE, PATH_SHIPPED
    )  # a separate forward


def test_path_spec_is_the_bug_policy_dimension():
    assert set(PATH_SPEC) == set(PATHS)
    assert PATH_SPEC[PATH_SHIPPED] == {"noise": "all", "kl": "clamp"}
    assert PATH_SPEC[PATH_FIXED_KL] == {"noise": "all", "kl": "logspace"}
    assert PATH_SPEC[PATH_FIXED] == {"noise": "real", "kl": "logspace"}


def test_cell_key_roundtrip_and_refusals():
    key = arms.cell_key(ARM_CLI_ATOMIC, PATH_SHIPPED, CUT_SHIPPED, "request")
    assert key == "trace/cli-atomic/shipped/shipped/request"
    assert arms.parse_cell_key(key) == (ARM_CLI_ATOMIC, PATH_SHIPPED, CUT_SHIPPED, "request")
    with pytest.raises(ValueError):
        arms.cell_key(ARM_CORE, PATH_SHIPPED, CUT_SHIPPED, "request")  # core has no shipped cut
    with pytest.raises(ValueError):
        arms.cell_key(
            ARM_SALIENCY, PATH_SHIPPED, CUT_FROZEN, "request"
        )  # saliency has no shipped path
    with pytest.raises(ValueError):
        arms.cell_key(ARM_CORE, PATH_SHIPPED, CUT_FROZEN, "hourly")
    with pytest.raises(ValueError):
        arms.cell_key("trace/unknown", PATH_SHIPPED, CUT_FROZEN, "request")
    with pytest.raises(ValueError):
        arms.parse_cell_key("trace/core/max/request")  # the sibling's three-slot key


def test_allowed_cells_and_columns():
    cells = arms.allowed_cells(ARM_CLI)
    assert len(cells) == 3 * 2 * len(GRAINS)
    assert len(arms.allowed_cells(ARM_SHAPLEY)) == len(GRAINS)
    assert arms.column_of(ARM_CORE, PATH_FIXED_KL) == "trace-core__fixed-kl"
    assert arms.arm_slug(ARM_CLI_ATOMIC) == "trace-cli-atomic"
