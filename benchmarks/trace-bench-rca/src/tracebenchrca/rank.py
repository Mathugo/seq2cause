"""Rank the deployed services of every case of one export with a root-cause arm (method side).

    python -m tracebenchrca.rank --arm rca/baro --export-record <export stage folder> --rung xs \
        --prior-rung-record smallest-rung --staging-ledger ../trace-bench/plans/staging-ledger.json \
        --output-folder <run-dir>

The arm reads each case only through the case door (`tracebenchrca.cases`) and only the inputs it
declares (`constants.ARMS`): `rca/baro-bocpd` never opens `window.json`. The table goes to the
pinned function unmodified, as registered in `plans/rca-arms.md` §2:

- `rca/baro` — RCAEval `e2e/baro.py::baro(data, inject_time, dataset=None)` on the whole case
  table after the suite's fill rule; the `time` column is a scored "metric" of that path and is
  removed from the returned ranking before the services collapse (D-TBR-2), and its rank is
  recorded.
- `rca/baro-bocpd` — `baro.anomaly_detection.bocpd(data)` on the same table, then
  `baro.root_cause_analysis.robust_scorer(data, anomalies=<change points>)`, which splits at the
  first detected row; a case without a change point has no ranking (a counted miss), never a
  fallback to the injection time.
- `rca/circa` — RCAEval `e2e/circa.py::circa(data, inject_time, dataset=None)`: the port builds a PC
  graph over every metric column and ranks the metrics with its regression-based hypothesis-testing
  head (D-TBR-8: the port's exception fallback, the unranked column order with `adj = []`, is a
  counted miss, never a ranking).
- `rca/circa-topology` — the same head (`graph_heads/rht.py::rht`) on the structural graph of CIRCA's
  Algorithm 1 built from the case's `topology.json` (D-TBR-9): Traffic → Latency, Traffic → Errors inside
  a service; caller Traffic → callee Traffic, callee Latency → caller Latency, callee Errors → caller
  Errors across each call edge; no PC.
- `rca/tracerca` — RCAEval `e2e/tracerca.py::tracerca` (the suite's reimplementation, D-TBR-10) on the
  case's `spans.csv` (the exporter's extra: the end-ordered request view's spans of every trace ending
  in the case window), columns renamed to the port's, the injection time in microseconds; a `KeyError`
  (an operation first seen after the injection) and an empty ranking (no abnormal span) are counted
  misses; operations collapse to services through the spans' own (service, operation) pairs. The
  synthetic client's root spans (service `client`: not a candidate, absent from `services.json` and
  `topology.json`; its span is the whole request) are dropped before the port and counted per case
  (`n_spans_client`; correction 2026-10-26).
- `control/rule` — the three-rule alert counter of `plans/controls.md` §2 (addendum 2026-10-23), no
  third-party code: per service, `latency-95`, `error` and `count` alert at the first post-fault bin
  more than 3 population standard deviations from their pre-fault mean; services ranked by the
  number of alerting series, then the earliest alert time, then the export's order.

Writes `ranking-positional.json` — `{"cases": {"case-00": [service, …]}}`, keyed by case
directory and never by case id (D-TBR-3); `tracebenchrca.score` restores the ids on the score
side. The record carries, per case, the rows, the cells the fill rule changed, the pre-fault rows
the scorer fitted on and the number of metrics it fitted on fewer than four rows, and for the
full pipeline the first change point's row and time (its offset from the labelled injection is
computed on the score side).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import math
from pathlib import Path

import numpy as np

from .arms import check_input, code_of
from .cases import (
    collapse_to_services,
    list_cases,
    read_services,
    read_spans,
    read_table,
    read_topology,
    read_window,
    split_at,
)
from .constants import (
    ARM_BARO,
    ARM_BARO_BOCPD,
    ARM_CIRCA,
    ARM_CIRCA_TOPOLOGY,
    ARM_CONTROL_RULE,
    ARM_TRACERCA,
    CIRCA_ACROSS,
    CIRCA_META,
    CIRCA_WITHIN,
    CLIENT_SERVICE,
    INPUT_SERIES,
    INPUT_SERVICES,
    INPUT_SPANS,
    INPUT_TOPOLOGY,
    INPUT_WINDOW,
    METRIC_SEP,
    MIN_PRE_BINS,
    RESULTS_JSON,
    RUN_DIR,
    RUNGS,
    TIME_COLUMN,
    TRACERCA_PORT_COLUMNS,
    TRACERCA_SIGMAS,
    US_PER_S,
)
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json
from .staging import StagingRefusal, require_prior_rung

STAGE = "rank"
RANK_ARMS = (
    ARM_BARO,
    ARM_BARO_BOCPD,
    ARM_CIRCA,
    ARM_CIRCA_TOPOLOGY,
    ARM_TRACERCA,
    ARM_CONTROL_RULE,
)  # the ranking arms with a method-side stage (control/random is analytic)
POSITIONAL_JSON = "ranking-positional.json"


def _baro():
    return importlib.import_module("RCAEval.e2e.baro").baro


def _bocpd():
    return importlib.import_module("baro.anomaly_detection").bocpd


def _robust_scorer():
    return importlib.import_module("baro.root_cause_analysis").robust_scorer


def _circa():
    return importlib.import_module("RCAEval.e2e.circa").circa


def _rht():
    return importlib.import_module("RCAEval.graph_heads.rht").rht


def _tracerca():
    return importlib.import_module("RCAEval.e2e.tracerca").tracerca


def _n_metrics(df):
    return len([c for c in df.columns if c != TIME_COLUMN])


def _table(arm, case_dir):
    check_input(arm, INPUT_SERIES)
    check_input(arm, INPUT_SERVICES)
    df, n_filled = read_table(case_dir, fill=True)
    return df, n_filled, [s for s, _ in read_services(case_dir)]


def _ranked(metric_ranks, df, n_pre, services):
    services_ranked = collapse_to_services(metric_ranks)
    return {
        "services": services_ranked,
        "n_rows": len(df),
        "n_pre_rows": n_pre,
        "n_metrics": _n_metrics(df),
        "n_metrics_lt_min_pre": _n_metrics(df) if n_pre < MIN_PRE_BINS else 0,
        "time_rank": metric_ranks.index(TIME_COLUMN) + 1 if TIME_COLUMN in metric_ranks else None,
        "n_services_unranked": len(set(services) - set(services_ranked)),
    }


def rank_baro(case_dir):
    df, n_filled, services = _table(ARM_BARO, case_dir)
    check_input(ARM_BARO, INPUT_WINDOW)
    inject_time = read_window(case_dir)["inject_time"]
    out = _baro()(df, inject_time=inject_time, dataset=None)
    n_pre = int((df[TIME_COLUMN] < inject_time).sum())
    return {"n_filled": n_filled, **_ranked([str(c) for c in out["ranks"]], df, n_pre, services)}


def rank_baro_bocpd(case_dir):
    df, n_filled, services = _table(ARM_BARO_BOCPD, case_dir)
    try:
        cps = [int(i) for i in _bocpd()(df)]
    except np.linalg.LinAlgError as e:
        # the detector's multivariate-t predictive calls scipy's eigh (default driver dsyevr), which can fail on
        # the shape matrix's clusters of repeated eigenvalues (addendum 2026-10-21): a counted miss, no fallback
        return {
            "n_filled": n_filled,
            "services": None,
            "miss": "detector raised LinAlgError",
            "detector_error": f"{type(e).__name__}: {e}",
            "n_rows": len(df),
            "n_metrics": _n_metrics(df),
        }
    if not cps:
        return {
            "n_filled": n_filled,
            "services": None,
            "miss": "no change point",
            "n_change_points": 0,
            "n_rows": len(df),
            "n_metrics": _n_metrics(df),
        }
    first = cps[0]
    if first >= len(df):
        # the run-length array has one entry more than the table, so the detector can place a change after the
        # last row; the scorer's split then leaves no fault rows and raises (addendum 2026-10-21): a counted miss
        return {
            "n_filled": n_filled,
            "services": None,
            "miss": "change point at the table's end",
            "n_change_points": len(cps),
            "change_point_row": first,
            "change_point_time": None,
            "n_rows": len(df),
            "n_metrics": _n_metrics(df),
        }
    out = _robust_scorer()(df, anomalies=cps)
    return {
        "n_filled": n_filled,
        "n_change_points": len(cps),
        "change_point_row": first,
        "change_point_time": int(df[TIME_COLUMN].iloc[first]),
        **_ranked([str(c) for c in out["ranks"]], df, first, services),
    }


def _n_undirected_edges(adj):
    a = np.asarray(adj)
    n = a.shape[0]
    iu = np.triu_indices(n, 1)
    return int(((a[iu] != 0) | (a.T[iu] != 0)).sum())


def rank_circa(case_dir):
    df, n_filled, services = _table(ARM_CIRCA, case_dir)
    check_input(ARM_CIRCA, INPUT_WINDOW)
    inject_time = read_window(case_dir)["inject_time"]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):  # the port's wrapper prints its fallback notice to standard error
        out = _circa()(df.copy(), inject_time=inject_time, dataset=None)
    if isinstance(out["adj"], list) and len(out["adj"]) == 0:
        # D-TBR-8: the port caught an exception and returned the unranked column order — a counted miss
        return {
            "n_filled": n_filled,
            "services": None,
            "miss": "port fallback",
            "port_stderr": err.getvalue().strip()[:500],
            "n_rows": len(df),
            "n_metrics": _n_metrics(df),
        }
    n_pre = int((df[TIME_COLUMN] < inject_time).sum())
    return {
        "n_filled": n_filled,
        "n_edges": _n_undirected_edges(out["adj"]),
        **_ranked([str(c) for c in out["ranks"]], df, n_pre, services),
    }


def structural_graph(columns, edges):
    """CIRCA's Algorithm 1 on this directory's four series (D-TBR-9): the adjacency over `columns` (metric
    names, `time` excluded, in table order) in the encoding the port's `rht()` reads — for a cause column c and
    an effect column e, `adj[c, e] = 1` and `adj[e, c] = -1`, which after the port's edge construction and
    `reverse()` makes c a parent of e. Returns `(adj, n_edges)`; refuses a pair set in both directions."""
    idx = {c: i for i, c in enumerate(columns)}
    by_meta = {}
    for c in columns:
        svc, metric = c.split(METRIC_SEP, 1)
        by_meta.setdefault(svc, {}).setdefault(CIRCA_META[metric], []).append(c)
    adj = np.zeros((len(columns), len(columns)), dtype=int)
    pairs = set()

    def add(cause, effect):
        if (effect, cause) in pairs:
            raise ValueError(f"structural graph: {cause} and {effect} would be causes of each other")
        if (cause, effect) in pairs:
            return
        pairs.add((cause, effect))
        adj[idx[cause], idx[effect]] = 1
        adj[idx[effect], idx[cause]] = -1

    for _svc, metas in by_meta.items():
        for cm, em in CIRCA_WITHIN:
            for cause in metas.get(cm, []):
                for effect in metas.get(em, []):
                    add(cause, effect)
    for caller, callee, _ in edges:
        for cm, em, direction in CIRCA_ACROSS:
            src, dst = (caller, callee) if direction == "caller->callee" else (callee, caller)
            for cause in by_meta.get(src, {}).get(cm, []):
                for effect in by_meta.get(dst, {}).get(em, []):
                    add(cause, effect)
    return adj, len(pairs)


def rank_circa_topology(case_dir):
    df, n_filled, services = _table(ARM_CIRCA_TOPOLOGY, case_dir)
    check_input(ARM_CIRCA_TOPOLOGY, INPUT_WINDOW)
    check_input(ARM_CIRCA_TOPOLOGY, INPUT_TOPOLOGY)
    inject_time = read_window(case_dir)["inject_time"]
    edges = read_topology(case_dir)
    columns = [c for c in df.columns if c != TIME_COLUMN]
    adj, n_edges = structural_graph(columns, edges)
    scored = _rht()(adj, inject_time, df)  # the port's head, as circa() calls it; any exception fails the run
    ranks = [name for name, _ in sorted(scored, key=lambda x: x[1], reverse=True)]  # the port's own re-sort
    n_pre = int((df[TIME_COLUMN] < inject_time).sum())
    return {
        "n_filled": n_filled,
        "n_edges": n_edges,
        "n_call_edges": len(edges),
        **_ranked(ranks, df, n_pre, services),
    }


def tracerca_frame(spans):
    """The port's span table from the exporter's columns: `serviceName`, `methodName`, `operationName`, `startTime`,
    `duration` (microseconds), plus the (service, operation) pairs that map the port's `serviceName_methodName`
    keys back to services."""
    df = spans.rename(columns=TRACERCA_PORT_COLUMNS)[list(TRACERCA_PORT_COLUMNS.values())].copy()
    df["operationName"] = df["methodName"]
    service_of = {f"{s}_{o}": s for s, o in spans[["service", "operation"]].drop_duplicates().itertuples(index=False)}
    return df, service_of


def abnormal_span_count(df, inject_us):
    """The number of post-fault spans the port flags (its rule: duration/1000 >= mean + 3 SD of the operation's
    pre-fault durations/1000; an operation with an undefined SD never flags) — for the record only."""
    op = df["serviceName"] + "_" + df["methodName"]
    pre = df["startTime"] + df["duration"] < inject_us
    ms = df["duration"] / 1000.0
    stats = ms[pre].groupby(op[pre]).agg(["mean", "std"])
    post = ~pre
    thr = op[post].map(stats["mean"]) + TRACERCA_SIGMAS * op[post].map(stats["std"])
    return int((ms[post] >= thr).sum()), int(post.sum()), int(pre.sum())


def rank_tracerca(case_dir):
    check_input(ARM_TRACERCA, INPUT_SPANS)
    check_input(ARM_TRACERCA, INPUT_SERVICES)
    check_input(ARM_TRACERCA, INPUT_WINDOW)
    spans_read = read_spans(case_dir)
    is_client = spans_read["service"] == CLIENT_SERVICE  # correction 2026-10-26: the client is not a candidate
    spans = spans_read[~is_client].reset_index(drop=True)
    services = [s for s, _ in read_services(case_dir)]
    inject_us = read_window(case_dir)["inject_time"] * US_PER_S
    df, service_of = tracerca_frame(spans)
    n_abnormal, n_post, n_pre = abnormal_span_count(df, inject_us)
    base = {
        "n_spans_read": int(len(spans_read)),
        "n_spans_client": int(is_client.sum()),
        "n_traces_read": int(spans_read["trace_id"].nunique()),
        "n_spans": int(len(spans)),
        "n_traces": int(spans["trace_id"].nunique()),
        "n_spans_pre": n_pre,
        "n_spans_post": n_post,
        "n_abnormal_spans": n_abnormal,
        "n_operations": len(service_of),
        "n_filled": 0,
    }
    try:
        out = _tracerca()(df.copy(), inject_time=inject_us, dataset=None)
    except KeyError as e:
        # the port's SLO table lacks an operation first seen after the injection (addendum 2026-10-25, later (2))
        return {**base, "services": None, "miss": "operation unseen before the injection", "detail": str(e)[:200]}
    ranked_ops = [str(o) for o in out["ranks"]]
    unknown = [o for o in ranked_ops if o not in service_of]
    if unknown:
        raise ValueError(f"the port ranked operations outside the spans' pairs: {unknown[:5]}")
    seen = []
    for o in ranked_ops:
        if service_of[o] not in seen:
            seen.append(service_of[o])
    return {
        **base,
        "services": seen,
        "n_operations_ranked": len(ranked_ops),
        "n_rows": int(len(spans)),
        "n_pre_rows": n_pre,
        "n_metrics": len(service_of),
        "n_metrics_lt_min_pre": 0,
        "time_rank": None,
        "n_services_unranked": len(set(services) - set(seen)),
    }


RULE_SERIES = ("latency-95", "error", "count")  # plans/controls.md §2: the median latency is not a rule
RULE_SIGMAS = 3.0


def first_alert(pre, post, column):
    """The time of the first post-fault bin more than `RULE_SIGMAS` population standard deviations from
    the pre-fault mean, in either direction; `None` if the series never alerts (a zero standard
    deviation, or no pre-fault row, never alerts)."""
    base = pre[column].to_numpy(dtype=float)
    if len(base) == 0:
        return None
    sd = float(base.std())  # divisor n (addendum 2026-10-23)
    if sd == 0.0:
        return None
    hit = np.abs(post[column].to_numpy(dtype=float) - base.mean()) > RULE_SIGMAS * sd
    return int(post[TIME_COLUMN].to_numpy()[hit][0]) if hit.any() else None


def rank_rule(case_dir):
    df, n_filled, services = _table(ARM_CONTROL_RULE, case_dir)
    check_input(ARM_CONTROL_RULE, INPUT_WINDOW)
    inject_time = read_window(case_dir)["inject_time"]
    pre, post = split_at(df, inject_time)
    alerts = {}
    for s in services:
        times = {m: first_alert(pre, post, f"{s}{METRIC_SEP}{m}") for m in RULE_SERIES}
        alerts[s] = {m: t for m, t in times.items() if t is not None}

    def key(i):
        a = alerts[services[i]]
        return (-len(a), min(a.values()) if a else math.inf, i)

    ranked = [services[i] for i in sorted(range(len(services)), key=key)]
    return {
        "n_filled": n_filled,
        "services": ranked,
        "n_rows": len(df),
        "n_pre_rows": len(pre),
        "n_alerting_services": sum(1 for s in services if alerts[s]),
        "alerts": {s: alerts[s] for s in ranked if alerts[s]},
    }


RANKERS = {
    ARM_BARO: rank_baro,
    ARM_BARO_BOCPD: rank_baro_bocpd,
    ARM_CIRCA: rank_circa,
    ARM_CIRCA_TOPOLOGY: rank_circa_topology,
    ARM_TRACERCA: rank_tracerca,
    ARM_CONTROL_RULE: rank_rule,
}


def read_export_record(export_record, rung):
    path = Path(export_record) / RUN_DIR / RESULTS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{export_record} holds no export record ({RUN_DIR}/{RESULTS_JSON})")
    res = read_json(path)
    if res.get("status") != "ok" or res.get("stage") != "export":
        raise ValueError(f"{path} is not a completed export record")
    if res.get("rung") != rung:
        raise ValueError(f"the export is a rung {res.get('rung')!r} export; this read is rung {rung!r}")
    return res, sha256_file(path)


def check_prior_arm(staging, arm):
    """The staging rule names the same stage at the previous rung; the root-cause stage also needs the same arm."""
    if staging["form"] == "record":
        prior_arm = read_json(Path(staging["path"]) / RUN_DIR / RESULTS_JSON).get("arm")
    elif staging["form"] == "ledger":
        prior_arm = staging["entry"].get("arm")
    else:
        return staging
    if prior_arm != arm:
        raise StagingRefusal(f"the prior-rung record is an {prior_arm!r} read; this read is {arm!r}")
    return staging


def rank(arm, export_record, rung, prior, ledger, out):
    if arm not in RANKERS:
        raise ValueError(f"arm {arm!r} has no ranking stage; ranking arms: {RANK_ARMS}")
    export_res, export_sha = read_export_record(export_record, rung)
    staging = check_prior_arm(require_prior_rung(prior, rung, STAGE, ledger), arm)
    cases = []
    positional = {}
    for d in list_cases(export_res["export_dir"]):
        row = {"case_dir": d.name, **RANKERS[arm](d)}
        if row["services"] is not None:
            positional[d.name] = row["services"]
        cases.append(row)
        log({"event": "rank_case", "arm": arm, "case_dir": d.name, "ranked": row["services"] is not None})
    out = Path(out)
    write_json(out / POSITIONAL_JSON, {"cases": positional})
    return {
        "stage": STAGE,
        "arm": arm,
        "code": code_of(arm),
        "rung": rung,
        "corpus_id": export_res["corpus_id"],
        "export_record_sha256": export_sha,
        "staging": staging,
        "ranking_file": POSITIONAL_JSON,
        "ranking_sha256": sha256_file(out / POSITIONAL_JSON),
        "n_cases": len(cases),
        "n_unranked_cases": len(cases) - len(positional),
        "misses_by_reason": {
            r: sum(1 for c in cases if c.get("miss") == r) for r in sorted({c["miss"] for c in cases if c.get("miss")})
        },
        # the pinned scorer returned no metric at all (the full pipeline's per-segment constant-column drop); scored as
        # a miss by the benchmark's scorer, counted here
        "n_empty_rankings": sum(1 for c in cases if c["services"] == []),
        "n_filled_cells": sum(c["n_filled"] for c in cases),
        "n_metrics_lt_min_pre": sum(c.get("n_metrics_lt_min_pre", 0) for c in cases),
        "cases": cases,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=RANK_ARMS)
    p.add_argument("--export-record", required=True, help="the export stage's output folder (its run/ record)")
    p.add_argument("--rung", required=True, choices=RUNGS, help="must equal the export's rung")
    p.add_argument(
        "--prior-rung-record",
        required=True,
        help="smallest-rung | the previous rung's rank folder of the same arm | its ledger sha256 | an owner-dated exception (.md)",
    )
    p.add_argument("--staging-ledger", required=True, help="the shared staging ledger (constants.STAGING_LEDGER)")
    p.add_argument("--output-folder", required=True, help="where the ranking and this stage's run record are written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = rank(
            args.arm, args.export_record, args.rung, args.prior_rung_record, args.staging_ledger, args.output_folder
        )
        log(
            {
                "event": "rank",
                "arm": args.arm,
                "corpus_id": res["corpus_id"],
                "n_cases": res["n_cases"],
                "n_unranked": res["n_unranked_cases"],
            }
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
