"""The case door (`tracebenchrca.cases`): a method reads the three case files and nothing else;
the window carries exactly its two fields; the table is the services × metrics grid with the
suite's fill rule applied and counted; the suite's split and service collapse."""

import pytest
from fixture_corpus import METRICS, SERVICES, fixture_export

from tracebenchrca import cases


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    return root, fixture_export(root)


def test_list_cases_in_position_order(export):
    root, dirs = export
    assert cases.list_cases(root) == dirs and [d.name for d in dirs] == ["case-00", "case-01"]
    with pytest.raises(FileNotFoundError):
        cases.list_cases(root / "nothing")


@pytest.mark.parametrize(
    "name",
    [
        "labels/cases.json",
        "../case-01/data.csv",
        "../../instantiation.json",
        "run/results.json",
        "graph.json",
        "COMPLETE",
    ],
)
def test_refuses_everything_but_the_three_case_files(export, name):
    _, dirs = export
    with pytest.raises(cases.NotCaseReadable):
        cases.open_case_file(dirs[0], name)


def test_window_carries_exactly_two_fields(export, tmp_path):
    _, dirs = export
    w = cases.read_window(dirs[0])
    assert tuple(w) == ("inject_time", "end_time") and w["inject_time"] < w["end_time"]
    bad = tmp_path / "case-09"
    bad.mkdir()
    (bad / "window.json").write_text('{"inject_time": 1, "end_time": 2, "root_cause_component": "x"}')
    with pytest.raises(ValueError, match="exactly"):
        cases.read_window(bad)


def test_table_is_the_grid_with_the_suite_fill_rule(export):
    _, dirs = export
    services = cases.read_services(dirs[1])
    assert services == SERVICES
    raw, _ = cases.read_table(dirs[1], fill=False)
    assert raw.isna().sum().sum() == 2  # the one empty bin's two latency cells
    df, n_filled = cases.read_table(dirs[1])
    assert n_filled == 2 and df.isna().sum().sum() == 0
    assert list(df.columns) == ["time"] + [f"{s}_{m}" for s, _ in SERVICES for m in METRICS]
    w = cases.read_window(dirs[1])
    pre, post = cases.split_at(df, w["inject_time"])
    assert (
        len(pre) == 30
        and len(post) == 10
        and (pre.time < w["inject_time"]).all()
        and (post.time >= w["inject_time"]).all()
    )
    # forward fill carried the previous bin's latency into the empty bin, as the suite's loader does
    empty_row = df[df["ext-gamma_count"] == 0].iloc[0]
    assert empty_row["ext-gamma_latency-50"] == 0.010 and empty_row["ext-gamma_latency-95"] == 0.050


def test_column_grid_mismatch_is_refused(export, tmp_path):
    _, dirs = export
    bad = tmp_path / "case-07"
    bad.mkdir()
    for f in ("window.json", "services.json"):
        (bad / f).write_text((dirs[0] / f).read_text())
    (bad / "data.csv").write_text("time,tb-edge_count\n1,2\n")
    with pytest.raises(ValueError, match="grid"):
        cases.read_table(bad)


def test_suite_collapse_keeps_first_occurrence_and_drops_time():
    ranking = ["time", "tb-beta_latency-95", "tb-edge_error", "tb-beta_count", "ext-gamma_count", "tb-edge_latency-50"]
    assert cases.collapse_to_services(ranking) == ["tb-beta", "tb-edge", "ext-gamma"]
    assert cases.service_of("ext-gamma_latency-50") == "ext-gamma"
