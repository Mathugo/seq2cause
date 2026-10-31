"""PRD Arm Register (amended 2026-09-24; the floor family of plans/floors.md, 2026-10-09) and
scenario 39: the registry, its paths and cuts, and the four-slot cell key."""

import pytest

from seq2causebench import arms
from seq2causebench.constants import (
    AGGREGATIONS,
    ARM_CLI,
    ARM_CLI_ATOMIC,
    ARM_CLI_ATOMIC_CMI_KEEP,
    ARM_CORE,
    ARM_CORE_CMI_KEEP,
    ARM_CORE_CMI_OVERWRITE,
    ARM_CORE_IG_KEEP,
    ARM_CORE_IG_OVERWRITE,
    ARM_GRANGER,
    ARM_SALIENCY,
    ARM_SHAPLEY,
    ARMS,
    BASELINE_ARMS,
    CONTEXT_ARMS,
    CONTEXT_FAMILY_GRID,
    CONTEXT_MODE_OF_PROBE,
    CUT_FROZEN,
    CUT_SHIPPED,
    FLOOR_ARM_OF_PROBE,
    FLOOR_ARMS,
    FLOOR_PROBES,
    GRAINS,
    PATH_FIXED,
    PATH_FIXED_KL,
    PATH_NONE,
    PATH_SHIPPED,
    PATH_SPEC,
    PATHS,
    PROBE_CLI_ATOMIC_CMI,
    PROBE_CORE,
    PROBE_CORE_KEEP,
    REFERENCE_ARMS,
    STAT_ATOMIC_CMI,
    STAT_LAG_CMI,
    STAT_LAG_INFO_GAIN,
    TAU_GRID_ARMS,
)


def test_thirteen_arms_three_reference_three_baseline_two_floor_five_context():
    assert set(ARMS) == (
        set(REFERENCE_ARMS) | set(BASELINE_ARMS) | set(FLOOR_ARMS) | set(CONTEXT_ARMS)
    )
    assert len(ARMS) == 13 and len(FLOOR_ARMS) == 2 and len(CONTEXT_ARMS) == 5
    assert REFERENCE_ARMS == (ARM_CORE, ARM_CLI, ARM_CLI_ATOMIC)  # the landed arms, never extended
    assert all(ARMS[a]["class"] == "reference" for a in REFERENCE_ARMS)
    assert all(ARMS[a]["class"] == "reference" for a in CONTEXT_ARMS)
    assert all(ARMS[a]["class"] == "baseline" for a in BASELINE_ARMS)
    assert all(ARMS[a]["class"] == "floor" for a in FLOOR_ARMS)
    assert AGGREGATIONS == ("max",)


def test_context_family_arms_paths_cuts_and_grid():
    """The context family of plans/reference-arms.md, addendum 2026-10-31 (D-SB-20 … 23)."""
    assert CONTEXT_ARMS == (
        ARM_CORE_IG_OVERWRITE,
        ARM_CORE_CMI_OVERWRITE,
        ARM_CORE_IG_KEEP,
        ARM_CORE_CMI_KEEP,
        ARM_CLI_ATOMIC_CMI_KEEP,
    )
    for arm in CONTEXT_ARMS[:4]:  # the core 2 x 2: the fixed-kl path, the frozen cut
        assert ARMS[arm]["paths"] == (PATH_FIXED_KL,) and ARMS[arm]["cuts"] == (CUT_FROZEN,)
    assert (
        ARMS[ARM_CORE_IG_OVERWRITE]["probe"] == ARMS[ARM_CORE_CMI_OVERWRITE]["probe"] == PROBE_CORE
    )
    assert ARMS[ARM_CORE_IG_KEEP]["probe"] == ARMS[ARM_CORE_CMI_KEEP]["probe"] == PROBE_CORE_KEEP
    assert ARMS[ARM_CORE_IG_OVERWRITE]["statistic"] == ARMS[ARM_CORE_IG_KEEP]["statistic"]
    assert (
        ARMS[ARM_CORE_IG_OVERWRITE]["statistic"]
        == ARMS[ARM_CORE]["statistic"]
        == STAT_LAG_INFO_GAIN
    )
    assert (
        ARMS[ARM_CORE_CMI_OVERWRITE]["statistic"]
        == ARMS[ARM_CORE_CMI_KEEP]["statistic"]
        == STAT_LAG_CMI
    )
    # the overwrite arms share trace/core's pass (one forward, two more columns)
    assert arms.pass_of(ARM_CORE_CMI_OVERWRITE, PATH_FIXED_KL) == arms.pass_of(
        ARM_CORE, PATH_FIXED_KL
    )
    assert arms.pass_of(ARM_CORE_IG_KEEP, PATH_FIXED_KL) != arms.pass_of(ARM_CORE, PATH_FIXED_KL)
    # the atomic CMI arm: both paths (two forwards), both cuts, its own probe
    atomic = ARMS[ARM_CLI_ATOMIC_CMI_KEEP]
    assert atomic["probe"] == PROBE_CLI_ATOMIC_CMI and atomic["statistic"] == STAT_ATOMIC_CMI
    assert atomic["paths"] == (PATH_FIXED_KL, PATH_FIXED) and atomic["cuts"] == (
        CUT_FROZEN,
        CUT_SHIPPED,
    )
    assert len(arms.allowed_cells(ARM_CLI_ATOMIC_CMI_KEEP)) == 2 * 2 * len(GRAINS)
    assert arms.column_of(ARM_CLI_ATOMIC_CMI_KEEP, PATH_FIXED) == "trace-cli-atomic-cmi-keep__fixed"
    with pytest.raises(ValueError):
        arms.cell_key(ARM_CLI_ATOMIC_CMI_KEEP, PATH_SHIPPED, CUT_FROZEN, "request")  # no clamp CMI
    with pytest.raises(ValueError):
        arms.cell_key(ARM_CORE_CMI_KEEP, PATH_FIXED, CUT_FROZEN, "request")  # fixed-kl only
    # the τ grid, the context modes and the registered family grid
    assert TAU_GRID_ARMS == REFERENCE_ARMS + CONTEXT_ARMS
    assert CONTEXT_MODE_OF_PROBE == {PROBE_CORE: "overwrite", PROBE_CORE_KEEP: "none"}
    assert CONTEXT_FAMILY_GRID["N"] == (8,)
    assert CONTEXT_FAMILY_GRID["c"][PROBE_CORE] == {"request": (1, 2, 3), "session": (2, 4, 8)}
    for probe in (PROBE_CORE_KEEP, PROBE_CLI_ATOMIC_CMI):
        assert CONTEXT_FAMILY_GRID["c"][probe] == {"request": (1,), "session": (1,)}


def test_floor_arms_are_model_free_single_cut():
    for arm in FLOOR_ARMS:
        assert ARMS[arm]["paths"] == (PATH_NONE,) and ARMS[arm]["cuts"] == (CUT_FROZEN,)
        assert arms.is_floor(arm) and ARMS[arm]["probe"] in FLOOR_PROBES
        assert len(arms.allowed_cells(arm)) == len(GRAINS)
    assert not arms.is_floor(ARM_CORE) and not arms.is_floor(ARM_SALIENCY)
    assert {FLOOR_ARM_OF_PROBE[p] for p in FLOOR_PROBES} == set(FLOOR_ARMS)
    assert arms.floor_columns("topology") == {"floor/topology": {"none": "floor-topology__none"}}
    assert arms.cell_key("floor/bigram", PATH_NONE, CUT_FROZEN, "session") == (
        "floor/bigram/none/frozen/session"
    )
    with pytest.raises(ValueError):
        arms.floor_columns("core")  # a model probe is not a floor table
    with pytest.raises(ValueError):
        arms.cell_key("floor/topology", PATH_SHIPPED, CUT_FROZEN, "request")


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
