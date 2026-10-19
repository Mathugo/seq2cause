"""The arm registry of plans/rca-arms.md and plans/controls.md: three declared classes, every
arm's inputs and axes declared, the code pins, and the registered bin widths equal to the
exporter's."""

import re

import pytest

from tracebenchrca import arms
from tracebenchrca.constants import (
    ARM_BARO,
    ARM_BARO_BOCPD,
    ARM_CAUSALRCA,
    ARM_CONTROL_RANDOM,
    ARM_CONTROL_RULE,
    ARM_FLOOR_TOPOLOGY,
    ARMS,
    AXES,
    AXIS_GRAPH,
    AXIS_RANKING,
    BIN_WIDTH_S,
    CLASS_DISCOVERY,
    CLASS_FLOOR_CONTROL,
    CLASS_ROOT_CAUSE,
    CLASSES,
    CONTROL_ARMS,
    DISCOVERY_ARMS,
    GRAPH_ARMS,
    INPUT_SERIES,
    INPUT_SERVICES,
    INPUT_TOPOLOGY,
    INPUT_WINDOW,
    INPUTS,
    MIN_POST_BINS,
    ROOT_CAUSE_ARMS,
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_three_classes_six_arms():
    assert CLASSES == (CLASS_DISCOVERY, CLASS_ROOT_CAUSE, CLASS_FLOOR_CONTROL)
    assert len(ARMS) == 6
    assert set(ROOT_CAUSE_ARMS) == {ARM_BARO, ARM_BARO_BOCPD, ARM_CAUSALRCA}
    assert set(CONTROL_ARMS) == {ARM_CONTROL_RANDOM, ARM_CONTROL_RULE, ARM_FLOOR_TOPOLOGY}
    assert DISCOVERY_ARMS == ()  # declared, no member yet
    for arm in ARMS:
        assert arms.arm_class(arm) in CLASSES
        assert set(arms.inputs_of(arm)) <= set(INPUTS) and set(arms.axes_of(arm)) <= set(AXES)


def test_declared_inputs_match_the_plans():
    assert arms.inputs_of(ARM_BARO) == (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES)
    assert INPUT_WINDOW not in arms.inputs_of(ARM_BARO_BOCPD)  # the detector locates the change point
    assert arms.inputs_of(ARM_CONTROL_RANDOM) == (INPUT_SERVICES,)
    assert arms.inputs_of(ARM_FLOOR_TOPOLOGY) == (INPUT_TOPOLOGY,)
    assert arms.check_input(ARM_BARO, INPUT_WINDOW) == INPUT_WINDOW
    with pytest.raises(PermissionError):
        arms.check_input(ARM_BARO_BOCPD, INPUT_WINDOW)
    with pytest.raises(PermissionError):
        arms.check_input(ARM_CONTROL_RANDOM, INPUT_SERIES)


def test_axes_and_cell_keys():
    assert set(GRAPH_ARMS) == {ARM_CAUSALRCA, ARM_FLOOR_TOPOLOGY}
    assert arms.axes_of(ARM_CAUSALRCA) == (AXIS_RANKING, AXIS_GRAPH)
    assert arms.cell_key(ARM_BARO_BOCPD, AXIS_RANKING) == "rca/baro-bocpd/ranking"
    assert arms.parse_cell_key("rca/causalrca/graph") == (ARM_CAUSALRCA, AXIS_GRAPH)
    assert arms.arm_slug(ARM_BARO_BOCPD) == "rca-baro-bocpd"
    with pytest.raises(ValueError):
        arms.cell_key(ARM_BARO, AXIS_GRAPH)  # BARO is not scored on any graph axis
    with pytest.raises(ValueError):
        arms.cell_key("rca/unknown", AXIS_RANKING)
    with pytest.raises(ValueError):
        arms.parse_cell_key("rca/baro/shipped/frozen/request")  # the reference harness's key


def test_code_pins_are_full_shas_and_controls_have_none():
    for arm in ROOT_CAUSE_ARMS:
        code = arms.code_of(arm)
        assert SHA_RE.match(code["commit"]), arm
        if code["runner"] is not None:
            assert SHA_RE.match(code["runner_commit"])
    assert arms.code_of(ARM_BARO)["commit"] == arms.code_of(ARM_BARO_BOCPD)["commit"]
    assert arms.code_of(ARM_BARO)["runner"] == "phamquiluan/RCAEval"
    assert arms.code_of(ARM_BARO_BOCPD)["runner"] is None  # the full pipeline is the method package's own functions
    assert "no licence file" in ARMS[ARM_CAUSALRCA]["licence"]
    for arm in CONTROL_ARMS:
        assert arms.code_of(arm) is None


def test_registered_widths_are_the_exporters_and_divide_the_shard():
    assert BIN_WIDTH_S == {"xs": 30, "s": 15, "m": 60}
    for w in BIN_WIDTH_S.values():
        assert 900 % w == 0 and 300 // w >= MIN_POST_BINS  # the shortest shipped fault (xs crash, 300 s) keeps ≥ 5 bins
    export_rca = pytest.importorskip("tracebench.export_rca")  # present once the benchmark pin carries the exporter
    assert dict(export_rca.BIN_WIDTH_S) == BIN_WIDTH_S
    assert tuple(export_rca.METRICS) == ("count", "error", "latency-50", "latency-95")
    assert tuple(export_rca.WINDOW_KEYS) == ("inject_time", "end_time")
