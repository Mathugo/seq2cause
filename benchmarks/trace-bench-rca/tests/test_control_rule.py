"""`control/rule`, the three-rule alert counter (`plans/controls.md` §2 and its addenda of 2026-10-22
and 2026-10-23), pinned rule by rule on hand-built cases: the three series (the median latency is
not a rule), the population mean / standard deviation baseline on the pre-fault rows, the strict
3-sd alert in either direction, a zero standard deviation that never alerts, the suite's split and
fill rule, the ranking order (alerting series, earliest alert, export order, every service), the
stage record, and H-rca-rule in the report."""

import csv
import json

from fixture_corpus import EPOCH, WIDTH, fixture_corpus
from test_rank_score import _export_record, _rank, _score, _stub_baro

from tracebenchrca import rank as rank_mod
from tracebenchrca import report as report_mod
from tracebenchrca.constants import METRICS, RESULTS_JSON, RUN_DIR
from tracebenchrca.record import read_json

N_ROWS, INJECT = 40, 30
# a noisy baseline per metric: alternating values, so every series has mean m and population sd s
BASE = {"count": (19.0, 21.0), "error": (0.0, 0.02), "latency-50": (0.010, 0.012), "latency-95": (0.05, 0.06)}


def _time(row):
    return EPOCH + row * WIDTH


def _case(root, services, changes=None, base=None):
    """One case directory in the exporter's shape. `changes[(service, metric)] = {row: value}` overrides the
    alternating baseline at those rows (a value of "" is an empty cell)."""
    changes = changes or {}
    base = {**BASE, **(base or {})}
    root.mkdir(parents=True)
    with open(root / "data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["time"] + [f"{s}_{m}" for s in services for m in METRICS])
        for r in range(N_ROWS):
            row = [_time(r)]
            for s in services:
                for m in METRICS:
                    row.append(changes.get((s, m), {}).get(r, base[m][r % 2]))
            w.writerow(row)
    (root / "window.json").write_text(json.dumps({"inject_time": _time(INJECT), "end_time": _time(N_ROWS)}))
    (root / "services.json").write_text(json.dumps({"services": [{"name": s, "kind": "service"} for s in services]}))
    return root


def test_three_series_either_direction_and_the_ranking_order(tmp_path):
    case = _case(
        tmp_path / "case-00",
        ["svc-a", "svc-b", "svc-c", "svc-d"],
        {
            ("svc-a", "latency-95"): {INJECT + 2: 5.0},  # one series, late
            ("svc-b", "latency-50"): {r: 5.0 for r in range(INJECT, N_ROWS)},  # the median is not a rule
            ("svc-c", "count"): {INJECT: 0.0},  # a fall alerts too, and earlier than svc-a
            ("svc-d", "error"): {INJECT + 5: 0.9},
            ("svc-d", "latency-95"): {INJECT + 6: 5.0},  # two series
        },
    )
    out = rank_mod.rank_rule(case)
    assert out["services"] == ["svc-d", "svc-c", "svc-a", "svc-b"]
    assert out["alerts"] == {
        "svc-d": {"error": _time(INJECT + 5), "latency-95": _time(INJECT + 6)},
        "svc-c": {"count": _time(INJECT)},
        "svc-a": {"latency-95": _time(INJECT + 2)},
    }
    assert out["n_alerting_services"] == 3 and out["n_pre_rows"] == INJECT and out["n_rows"] == N_ROWS


def test_zero_sd_never_alerts_and_the_threshold_is_strict(tmp_path):
    # latency-95 baseline 9 / 11: mean 10, population sd 1 → 13 is exactly 3 sd (no alert), 13.5 is past it
    case = _case(
        tmp_path / "case-00",
        ["svc-a", "svc-b", "svc-c"],
        {
            ("svc-a", "count"): {**{r: 20.0 for r in range(INJECT)}, INJECT: 1000.0},
            ("svc-b", "latency-95"): {**{r: (9.0, 11.0)[r % 2] for r in range(INJECT)}, INJECT: 13.0},
            ("svc-c", "latency-95"): {**{r: (9.0, 11.0)[r % 2] for r in range(INJECT)}, INJECT: 13.5},
        },
        base={"latency-95": (9.0, 11.0)},
    )
    out = rank_mod.rank_rule(case)
    assert out["alerts"] == {"svc-c": {"latency-95": _time(INJECT)}}
    assert out["services"] == ["svc-c", "svc-a", "svc-b"]  # non-alerting services in the export's order


def test_split_at_the_injection_and_the_fill_rule(tmp_path):
    case = _case(
        tmp_path / "case-00",
        ["svc-a", "svc-b", "svc-c"],
        {
            ("svc-a", "latency-95"): {INJECT - 1: 5.0},  # the last pre-fault bin is baseline, not an alert
            ("svc-b", "latency-95"): {INJECT: 5.0},  # the bin at inject_time is post-fault
            ("svc-c", "latency-95"): {INJECT + 1: ""},  # an empty cell is forward-filled, never an alert
        },
    )
    out = rank_mod.rank_rule(case)
    assert out["alerts"] == {"svc-b": {"latency-95": _time(INJECT)}}
    assert out["n_filled"] == 1


def test_no_pre_fault_row_means_no_alert(tmp_path):
    case = _case(tmp_path / "case-00", ["svc-a", "svc-b"], {("svc-b", "count"): {0: 900.0}})
    w = json.loads((case / "window.json").read_text())
    (case / "window.json").write_text(json.dumps({**w, "inject_time": _time(0)}))
    out = rank_mod.rank_rule(case)
    assert out["n_pre_rows"] == 0 and out["alerts"] == {} and out["services"] == ["svc-a", "svc-b"]


def test_the_rank_stage_records_every_service_and_no_third_party_code(tmp_path):
    rec, _ = _export_record(tmp_path)
    assert _rank("control/rule", rec, tmp_path / "rule") == 0
    res = read_json(tmp_path / "rule" / RUN_DIR / RESULTS_JSON)
    assert res["arm"] == "control/rule" and res["code"] is None and res["n_unranked_cases"] == 0
    positional = read_json(tmp_path / "rule" / rank_mod.POSITIONAL_JSON)["cases"]
    assert all(sorted(v) == sorted(["tb-edge", "tb-beta", "ext-gamma"]) for v in positional.values())
    assert all("n_alerting_services" in c and "alerts" in c for c in res["cases"])


def test_report_scores_h_rca_rule_from_both_arms(tmp_path, monkeypatch):
    _stub_baro(monkeypatch, {EPOCH + 40 * WIDTH: "tb-beta", EPOCH + 90 * WIDTH: "ext-gamma"})
    results = tmp_path / "results"
    for k in range(5):
        rec, _ = _export_record(tmp_path, seed=k)
        for arm, slug in (("rca/baro", "rca-baro"), ("control/rule", "control-rule")):
            cell = results / f"xs/latent/seed={k}/{slug}"
            _rank(arm, rec, cell / "rank")
            _score(fixture_corpus(seed=k), cell / "rank", cell)
    out = tmp_path / "tables"
    argv = ["--results-dir", str(results), "--rungs", "xs", "--seeds", "0", "1", "2", "3", "4"]
    argv += ["--arms", "rca/baro", "control/rule", "--bootstrap-b", "50", "--bootstrap-seed", "0"]
    assert report_mod.main(argv + ["--output-folder", str(out)]) == 0
    summary = read_json(out / "summary.json")
    rule = next(r for r in summary["rows"] if r["arm"] == "control/rule")
    assert rule["alerting_services_per_case"]["n_cases"] == 10
    h = summary["hypotheses"]["H-rca-rule"]
    assert h["verdict"] == "pass" and h["per_rung"] == {"xs": True}
    assert "alerting services per case" in (out / "rca-latent.md").read_text(encoding="utf-8")
    # the verdict turns when the rule control's Avg@5 exceeds BARO's
    by = {(r["rung"], r["arm"]): r for r in summary["rows"]}
    by[("xs", "control/rule")]["metrics"]["avg@5"] = by[("xs", "rca/baro")]["metrics"]["avg@5"] + 0.01
    assert report_mod.hypotheses(list(by.values()), ["xs"])["H-rca-rule"]["verdict"] == "fail"
