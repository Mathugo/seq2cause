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
    ARM_CIRCA,
    ARM_CIRCA_TOPOLOGY,
    ARM_CONTROL_RANDOM,
    ARM_CONTROL_RULE,
    ARM_CONTROL_TRACE_CALLEES,
    ARM_CONTROL_TRACE_EARLIEST,
    ARM_CONTROL_TRACE_RANDOM,
    ARM_FLOOR_TOPOLOGY,
    ARM_FLOOR_TRACE_TOPOLOGY,
    ARM_PC_PAGERANK,
    ARM_RCD,
    ARM_TRACE,
    ARM_TRACE_CORE,
    ARM_TRACERCA,
    ARMS,
    AXES,
    AXIS_GRAPH,
    AXIS_RANKING,
    AXIS_TRACE,
    BIN_WIDTH_S,
    CLASS_DISCOVERY,
    CLASS_FLOOR_CONTROL,
    CLASS_ROOT_CAUSE,
    CLASSES,
    CONTEXT_TRACE_ARMS,
    CONTROL_ARMS,
    DISCOVERY_ARMS,
    GRAPH_ARMS,
    INPUT_SERIES,
    INPUT_SERVICES,
    INPUT_SPANS,
    INPUT_TOPOLOGY,
    INPUT_VIEWS,
    INPUT_WINDOW,
    INPUTS,
    MIN_POST_BINS,
    ROOT_CAUSE_ARMS,
    TRACE_AXIS_ARMS,
    TRACE_BROADCAST_ARMS,
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_three_classes_twenty_two_arms():
    assert CLASSES == (CLASS_DISCOVERY, CLASS_ROOT_CAUSE, CLASS_FLOOR_CONTROL)
    assert len(ARMS) == 22  # 16 landed + the context family's five + the earliest-error control (2026-10-31)
    assert set(ROOT_CAUSE_ARMS) == {
        ARM_BARO,
        ARM_BARO_BOCPD,
        ARM_CAUSALRCA,
        ARM_CIRCA,
        ARM_CIRCA_TOPOLOGY,
        ARM_RCD,
        ARM_TRACERCA,
        ARM_PC_PAGERANK,
        ARM_TRACE,
        ARM_TRACE_CORE,
        *CONTEXT_TRACE_ARMS,
    }
    assert set(CONTROL_ARMS) == {
        ARM_CONTROL_RANDOM,
        ARM_CONTROL_RULE,
        ARM_FLOOR_TOPOLOGY,
        ARM_FLOOR_TRACE_TOPOLOGY,
        ARM_CONTROL_TRACE_RANDOM,
        ARM_CONTROL_TRACE_CALLEES,
        ARM_CONTROL_TRACE_EARLIEST,
    }
    assert DISCOVERY_ARMS == ()  # declared, no member yet
    for arm in ARMS:
        assert arms.arm_class(arm) in CLASSES
        assert set(arms.inputs_of(arm)) <= set(INPUTS) and set(arms.axes_of(arm)) <= set(AXES)


def test_declared_inputs_match_the_plans():
    assert arms.inputs_of(ARM_BARO) == (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES)
    assert INPUT_WINDOW not in arms.inputs_of(ARM_BARO_BOCPD)  # the detector locates the change point
    assert arms.inputs_of(ARM_CONTROL_RANDOM) == (INPUT_SERVICES,)
    assert arms.inputs_of(ARM_FLOOR_TOPOLOGY) == (INPUT_TOPOLOGY,)
    for arm in (ARM_FLOOR_TRACE_TOPOLOGY, ARM_CONTROL_TRACE_RANDOM, ARM_CONTROL_TRACE_CALLEES):
        assert arms.inputs_of(arm) == ()  # score-side floors of the per-trace axis: no method-side input at all
        with pytest.raises(PermissionError):
            arms.check_input(arm, INPUT_SERVICES)
    assert arms.inputs_of(ARM_CIRCA_TOPOLOGY) == (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES, INPUT_TOPOLOGY)
    assert INPUT_TOPOLOGY not in arms.inputs_of(ARM_CIRCA)  # the PC cell reads no topology
    assert arms.inputs_of(ARM_TRACERCA) == (INPUT_SPANS, INPUT_WINDOW, INPUT_SERVICES)  # the series are not its input
    assert arms.inputs_of(ARM_PC_PAGERANK) == (INPUT_SERIES, INPUT_SERVICES)  # the port never reads the injection time
    for arm in (
        ARM_TRACE,
        ARM_TRACE_CORE,
    ):  # the view through the benchmark's accessor, never the series (addendum 2026-10-30)
        assert arms.inputs_of(arm) == (INPUT_VIEWS, INPUT_WINDOW, INPUT_SERVICES)
    assert arms.check_input(ARM_BARO, INPUT_WINDOW) == INPUT_WINDOW
    with pytest.raises(PermissionError):
        arms.check_input(ARM_PC_PAGERANK, INPUT_WINDOW)
    with pytest.raises(PermissionError):
        arms.check_input(ARM_BARO_BOCPD, INPUT_WINDOW)
    with pytest.raises(PermissionError):
        arms.check_input(ARM_CONTROL_RANDOM, INPUT_SERIES)
    with pytest.raises(PermissionError):
        arms.check_input(ARM_TRACE, INPUT_SERIES)


def test_axes_and_cell_keys():
    assert AXES == (AXIS_RANKING, AXIS_GRAPH, AXIS_TRACE)
    assert set(GRAPH_ARMS) == {ARM_CAUSALRCA, ARM_FLOOR_TOPOLOGY}
    assert arms.axes_of(ARM_CAUSALRCA) == (AXIS_RANKING, AXIS_GRAPH)
    assert set(TRACE_AXIS_ARMS) == {
        ARM_TRACE,
        ARM_TRACE_CORE,
        *CONTEXT_TRACE_ARMS,
        ARM_FLOOR_TRACE_TOPOLOGY,
        ARM_CONTROL_TRACE_RANDOM,
        ARM_CONTROL_TRACE_CALLEES,
        ARM_CONTROL_TRACE_EARLIEST,
    }
    for arm in (ARM_TRACE, ARM_TRACE_CORE, *CONTEXT_TRACE_ARMS, ARM_CONTROL_TRACE_EARLIEST):
        assert arms.axes_of(arm) == (AXIS_RANKING, AXIS_TRACE)
    for arm in (ARM_FLOOR_TRACE_TOPOLOGY, ARM_CONTROL_TRACE_RANDOM, ARM_CONTROL_TRACE_CALLEES):
        assert arms.axes_of(arm) == (AXIS_TRACE,) and arms.arm_class(arm) == CLASS_FLOOR_CONTROL
    assert arms.cell_key(ARM_TRACE, AXIS_TRACE) == "rca/trace/trace"
    assert arms.parse_cell_key("control/trace-random/trace") == (ARM_CONTROL_TRACE_RANDOM, AXIS_TRACE)
    # the case-level arms broadcast to the per-trace axis are ranking arms without a trace stage of their own
    assert set(TRACE_BROADCAST_ARMS) <= set(ARMS) and not set(TRACE_BROADCAST_ARMS) & set(TRACE_AXIS_ARMS)
    assert all(AXIS_RANKING in arms.axes_of(a) for a in TRACE_BROADCAST_ARMS)
    with pytest.raises(ValueError):
        arms.cell_key(ARM_BARO, AXIS_TRACE)  # BARO reaches the per-trace axis only by broadcast
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
    assert (
        arms.code_of(ARM_CIRCA)["commit"]
        == arms.code_of(ARM_CIRCA_TOPOLOGY)["commit"]
        == arms.code_of(ARM_BARO)["runner_commit"]
    )
    assert arms.code_of(ARM_PC_PAGERANK)["commit"] == arms.code_of(ARM_CIRCA)["commit"]  # the same pinned suite
    assert arms.code_of(ARM_TRACE)["commit"] == arms.code_of(ARM_TRACE_CORE)["commit"]  # the shipped seq2cause commit
    assert arms.code_of(ARM_TRACE)["repo"] == "Mathugo/seq2cause"
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
