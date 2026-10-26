# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Structural constants: names, the arm registry and the values the plans pin.

Nothing here is a tunable. Every algorithm knob is a required command-line flag whose value
rides the run record (`run/arguments.json`); the values below are data semantics (file names,
the export's shape, the arm registry with each arm's declared inputs and code pins) or numbers
the pre-registered plans fix (`plans/rca-arms.md`, `plans/controls.md`), which a run may report
against but never choose.
"""

from tracebench.constants import (  # noqa: F401  (re-exported: one definition across the repositories)
    CASES_JSON,
    COMPLETE_MARKER,
    CONFIG_YAML,
    GRAPHS_DIR,
    INSTANTIATION_JSON,
    KIND_CLIENT,
    KIND_NAMES,
    LABELS_DIR,
    MANIFEST_JSON,
    RAW_DIR,
    TOPOLOGY_DIR,
    VIEWS_DIR,
)

TOOL_NAME = "trace-bench-rca"
SIBLING = "alex-chadyuk/trace-cmi-bench@f54776a"  # the ultimate source of every copied or adapted module (NOTICE)
HARNESS = "benchmarks/trace-bench (seq2cause-trace-bench) at trace-bench-harness 14edb93c"  # the directory they were copied from

# --- the benchmark this directory scores --------------------------------------------------
DATASET_REPO = "chadyuk/trace-bench"  # dataset host id (public)
BENCHMARK_TAG = "v0.3.0"  # the release tag whose corpora the paper scores
BENCHMARK_TOOL_VERSION = "0.3.0"  # manifest.tool_version every scored corpus must carry
RUNGS = ("xs", "s", "m")  # the registered rungs of the root-cause arms (plans/rca-arms.md §1)
SEEDS = (0, 1, 2, 3, 4)
VARIANT = "latent"  # the only variant the root-cause arms run on

# --- the export's shape (tracebench.export_rca writes it; mirrored here so the method side never imports the exporter) ---
CASE_DIR_FMT = "case-{k:02d}"
CASE_DIR_GLOB = "case-*"
DATA_CSV = "data.csv"
WINDOW_JSON = "window.json"
SERVICES_JSON = "services.json"
TOPOLOGY_JSON = "topology.json"  # exporter extra (2026-10-26): deployed-service call edges, caller -> callee as stored
SPANS_CSV = "spans.csv"  # exporter extra (2026-10-26): the end-ordered request view's spans per case
CASE_FILES = (DATA_CSV, WINDOW_JSON, SERVICES_JSON, TOPOLOGY_JSON, SPANS_CSV)  # the ONLY files a method may open
BASE_CASE_FILES = (DATA_CSV, WINDOW_JSON, SERVICES_JSON)  # the three every export carries
# the synthetic client's service name in the benchmark's views (`tracebench.topology` at the pin: `Service(name="client",
# kind=KIND_CLIENT)`); it is not a deployed service (absent from services.json and topology.json) and the only service
# `spans.csv` may name outside services.json — `rca/tracerca` drops its spans before the port (correction 2026-10-26)
CLIENT_SERVICE = "client"
WINDOW_KEYS = ("inject_time", "end_time")  # the two label fields a method receives, UNIX seconds
TIME_COLUMN = "time"
METRICS = ("count", "error", "latency-50", "latency-95")  # per service, log-derived from the access lines
METRIC_SEP = (
    "_"  # `<service>_<metric>`: the standard suite's column shape; the service is the part before the first `_`
)
BIN_WIDTH_S = {
    "xs": 30,
    "s": 15,
    "m": 60,
}  # the registered rule's values (plans/rca-arms.md §3); equal to the exporter's table
MIN_PRE_BINS = 4  # RobustScorer's minimum pre-fault length
MIN_POST_BINS = 5  # the rule's fault-resolution bound

# --- run record (one directory per run) --------------------------------------------------
RUN_DIR = "run"
ARGUMENTS_JSON = "arguments.json"
RESULTS_JSON = "results.json"
RUN_META_JSON = "run_meta.json"
ARTIFACTS_JSON = "artifacts.json"
ARTIFACTS_SCHEMA = "tracebenchrca/artifacts@1"
LOG_JSONL = "log.jsonl"

# --- layout --------------------------------------------------------------------------------
RESULTS_DIR = "results"
TABLES_DIR = "tables"
PLANS_DIR = "plans"
FINDINGS_DIR = "findings"
EXPORTS_DIR = "exports"  # local, git-ignored: one export root per corpus
# the run ledger is SHARED with the reference harness beside this directory (one freeze tag covers both;
# publication plan question 5 — the owner confirms the sharing; a separate ledger is one constant away)
STAGING_LEDGER = "../trace-bench/plans/staging-ledger.json"
STAGING_LEDGER_SCHEMA = "seq2causebench/staging-ledger@1"
SMALLEST_RUNG_TOKEN = "smallest-rung"  # --prior-rung-record at the smallest rung

# --- arm classes (publication plan §3; plans/rca-arms.md §1) -------------------------------------
CLASS_DISCOVERY = "discovery"  # reads the exported series only, no fault window: learns a structure
CLASS_ROOT_CAUSE = "root-cause"  # reads the exported series and the fault window
CLASS_FLOOR_CONTROL = "floor-control"  # declared inputs, no learning (plans/controls.md)
CLASSES = (CLASS_DISCOVERY, CLASS_ROOT_CAUSE, CLASS_FLOOR_CONTROL)

# --- declared inputs: what a stage of an arm may read ------------------------------------------
INPUT_SERIES = "series"  # <case>/data.csv
INPUT_WINDOW = "window"  # <case>/window.json
INPUT_SERVICES = "services"  # <case>/services.json
INPUT_TOPOLOGY = "topology"  # `floor/topology`: score side (instantiation.json); `rca/circa-topology`: <case>/topology.json (D-TBR-9)
INPUT_SPANS = (
    "spans"  # <case>/spans.csv — the correlated request view's spans (a trace-reading, call-tree-informed arm)
)
INPUTS = (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES, INPUT_TOPOLOGY, INPUT_SPANS)

# --- axes a result is scored on (tracebench.score_rca, tracebench.score_service) -----------------
AXIS_RANKING = "ranking"
AXIS_GRAPH = "graph"
AXES = (AXIS_RANKING, AXIS_GRAPH)

# --- code pins (registered in plans/rca-arms.md with the two outreach mails of 2026-10-14) -------
BARO_REPO = "phamquiluan/baro"
BARO_COMMIT = "e35f4ec1095e5cac891d52de9ad18a5b32a37ec8"
RCAEVAL_REPO = "phamquiluan/RCAEval"
RCAEVAL_COMMIT = "259ea4167160256a74ad30004aa3d96a998c846e"
CAUSALRCA_REPO = "AXinx/CausalRCA_code"
CAUSALRCA_COMMIT = "21f4eb0d8139aa5046402f15e66fd56b65fd3844"

# --- CausalRCA's code path (plans/rca-arms.md, addenda 2026-10-22, later (D-TBR-5) and 2026-10-23) ---
CAUSALRCA_SCRIPT = "train_latency.py"
CAUSALRCA_FILES_SHA256 = {  # read at the pin on 2026-10-23; the adapter refuses any other content
    "train_latency.py": "6e86acfed2231ca907ae5356aaab0b211c2dd71319ab277389f768628cf58184",
    "modules.py": "5e1237aa156ca7f1d8dacadee50965c6f8f6d0c41bec5a78657bcec55d88c30e",
    "utils.py": "c394ed6f45acd7505c7ffdfa2993b7e58be5a30917718a2da9430a035ecd6b6b",
    "config.py": "451459d9c60cc74fbbf739f9a4cea0fe01df576ab11858b8b5b4269e64bfd746",
}
CAUSALRCA_BLOCK_START = "names = ['front-end', 'user', 'catalogue', 'orders', 'carts', 'payment', 'shipping']"
CAUSALRCA_BLOCK_END = "data = all_data[name]"
CAUSALRCA_EPOCHS = 1000  # D-TBR-4: the sent test plan's value; the repository's config.py says 500
CAUSALRCA_ARGV = ("train_latency.py", "--gamma", "0.25", "--eta", "10")  # the script's defaults = the sent plan
CAUSALRCA_RUNS = 10  # trainings per case and series, seeded with the run index
CAUSALRCA_SERIES = ("latency-50", "latency-95")  # one row per series
CAUSALRCA_GRAPH_PKL = "cpu-hog_front-end_adj.pkl"  # the script's own output names
CAUSALRCA_PNG = "metrics_causality.png"
CAUSALRCA_GRAPH_TAUS = (
    0,
    0.03,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.7,
    1,
    1.5,
    2,
    3,
    5,
)  # strict `> tau` on mean |adj|

# --- CIRCA's structural graph (plans/rca-arms.md, addendum 2026-10-25; D-TBR-9) --------------------------
# meta metric of each exported series: Traffic / Latency / Errors; Saturation is empty (no resource metrics)
CIRCA_META = {"count": "T", "latency-50": "L", "latency-95": "L", "error": "E"}
CIRCA_WITHIN = (("T", "L"), ("T", "E"))  # cause meta -> effect meta inside one service
CIRCA_ACROSS = (("T", "T", "caller->callee"), ("L", "L", "callee->caller"), ("E", "E", "callee->caller"))

# --- RCD (plans/rca-arms.md, addendum 2026-10-25, later; D-TBR-6, D-TBR-7) ---------------------------
RCD_RUNS = 10  # seeded runs per case, seed = run index (the paper averages over runs)
RCD_DATASET = "trace-bench"  # any name but "sock-shop" selects the suite's named-dataset preprocessing (D-TBR-6)
RCD_KWARGS = {
    "gamma": 5,
    "localized": True,
    "bins": 5,
    "dk_select_useful": False,
}  # the port's defaults, passed explicitly
RCD_ENV = "rcaeval-rcd-259ea41"  # the pinned environment (environment-rcd.yml; Python 3.10, the lock's versions)
RCD_OVERLAY_FILES = (  # the suite's lib/causallearn files copied over the installed causal-learn (script/link.sh)
    "utils/PCUtils/SkeletonDiscovery.py",
    "utils/Fas.py",
    "search/ConstraintBased/FCI.py",
    "graph/GraphClass.py",
)
RCD_OVERLAY_SHA256 = {  # read from the suite's lib/causallearn at the pin on 2026-10-25 (full values asserted by the runner)
    "utils/PCUtils/SkeletonDiscovery.py": "96b65269d34c89f1",
    "utils/Fas.py": "4f7e07434dc09e83",
    "search/ConstraintBased/FCI.py": "8b4acd094cdc1d83",
    "graph/GraphClass.py": "1d75e73450d635a8",
}  # 16-hex prefixes; the runner records the full digests and refuses an environment whose files differ

# --- TraceRCA (plans/rca-arms.md, addendum 2026-10-25, later (2); D-TBR-10) ---------------------------------
SPANS_COLUMNS = (
    "trace_id",
    "service",
    "operation",
    "start_us",
    "duration_us",
    "parent_pos",
)  # the exporter's spans.csv
TRACERCA_PORT_COLUMNS = {  # the exporter's columns -> the port's names (RCAEval/e2e/tracerca.py); microseconds throughout
    "service": "serviceName",
    "operation": "methodName",
    "start_us": "startTime",
    "duration_us": "duration",
}
TRACERCA_SIGMAS = 3.0  # the port's threshold: mean + 3 SD of the pre-fault durations per operation (count only)
US_PER_S = 1_000_000

# --- arms (plans/rca-arms.md §1, plans/controls.md) ----------------------------------------------
ARM_BARO = "rca/baro"
ARM_CIRCA = "rca/circa"
ARM_CIRCA_TOPOLOGY = "rca/circa-topology"
ARM_RCD = "rca/rcd"
ARM_TRACERCA = "rca/tracerca"
ARM_BARO_BOCPD = "rca/baro-bocpd"
ARM_CAUSALRCA = "rca/causalrca"
ARM_CONTROL_RANDOM = "control/random"
ARM_CONTROL_RULE = "control/rule"
ARM_FLOOR_TOPOLOGY = "floor/topology"
ARMS = {
    ARM_BARO: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES),
        "axes": (AXIS_RANKING,),
        "code": {
            "repo": BARO_REPO,
            "commit": BARO_COMMIT,
            "runner": RCAEVAL_REPO,
            "runner_commit": RCAEVAL_COMMIT,
            "entry": "RCAEval/e2e/baro.py::baro (RobustScorer at the labelled injection time; no BOCPD)",
        },
        "licence": "MIT",
    },
    ARM_BARO_BOCPD: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SERIES, INPUT_SERVICES),  # the detector locates the change point: no injection time
        "axes": (AXIS_RANKING,),
        "code": {
            "repo": BARO_REPO,
            "commit": BARO_COMMIT,
            "runner": None,
            "runner_commit": None,
            "entry": "baro.anomaly_detection.bocpd -> baro.root_cause_analysis.robust_scorer(anomalies=...)",
        },
        "licence": "MIT",
    },
    ARM_CAUSALRCA: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES),
        "axes": (AXIS_RANKING, AXIS_GRAPH),
        "code": {
            "repo": CAUSALRCA_REPO,
            "commit": CAUSALRCA_COMMIT,
            "runner": None,
            "runner_commit": None,
            "entry": "the coarse-grained (service-level) configuration: DAG-GNN on the post-injection latency window, "
            "reversed absolute weights, PageRank; 10 runs averaged",
        },
        "licence": "author repository, no licence file; authors contacted 2026-10-14",
    },
    ARM_CIRCA: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES),
        "axes": (AXIS_RANKING,),
        "code": {
            "repo": RCAEVAL_REPO,
            "commit": RCAEVAL_COMMIT,
            "runner": None,
            "runner_commit": None,
            "entry": "RCAEval/e2e/circa.py::circa (PC graph over every metric column, then the port's regression-based "
            "hypothesis testing; the port of NetManAIOps/CIRCA, BSD-3-Clause, which is not run)",
        },
        "licence": "MIT (the suite's port); the ported repository BSD-3-Clause",
    },
    ARM_CIRCA_TOPOLOGY: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES, INPUT_TOPOLOGY),
        "axes": (AXIS_RANKING,),
        "code": {
            "repo": RCAEVAL_REPO,
            "commit": RCAEVAL_COMMIT,
            "runner": None,
            "runner_commit": None,
            "entry": "RCAEval/graph_heads/rht.py::rht on the structural graph of CIRCA's Algorithm 1 built from the "
            "deployment call topology (D-TBR-9); no PC",
        },
        "licence": "MIT (the suite's port); the ported repository BSD-3-Clause",
    },
    ARM_RCD: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES),
        "axes": (AXIS_RANKING,),
        "code": {
            "repo": RCAEVAL_REPO,
            "commit": RCAEVAL_COMMIT,
            "runner": None,
            "runner_commit": None,
            "entry": "RCAEval/e2e/rcd.py::rcd (the authors' azamikram/rcd, MIT, as carried by the suite, on its patched "
            "causal-learn; named-dataset preprocessing, D-TBR-6; ten seeded runs in the suite's RCD environment, D-TBR-7)",
        },
        "licence": "MIT",
    },
    ARM_TRACERCA: {
        "class": CLASS_ROOT_CAUSE,
        "inputs": (INPUT_SPANS, INPUT_WINDOW, INPUT_SERVICES),
        "axes": (AXIS_RANKING,),
        "code": {
            "repo": RCAEVAL_REPO,
            "commit": RCAEVAL_COMMIT,
            "runner": None,
            "runner_commit": None,
            "entry": "RCAEval/e2e/tracerca.py::tracerca — the suite's reimplementation of TraceRCA (per-operation SLO from the "
            "pre-fault spans, support / confidence over flagged spans) on the view's spans (D-TBR-10); the authors' "
            "NetManAIOps/TraceRCA (no licence file) is cited, not run",
        },
        "licence": "MIT (the suite's reimplementation); the authors' repository has no licence file; authors contacted",
    },
    ARM_CONTROL_RANDOM: {
        "class": CLASS_FLOOR_CONTROL,
        "inputs": (INPUT_SERVICES,),
        "axes": (AXIS_RANKING,),
        "code": None,  # analytic: AC@k = min(k, n) / n over the n deployed services (tracebench.score_rca.random_expectation)
        "licence": None,
    },
    ARM_CONTROL_RULE: {
        "class": CLASS_FLOOR_CONTROL,
        "inputs": (INPUT_SERIES, INPUT_WINDOW, INPUT_SERVICES),
        "axes": (AXIS_RANKING,),
        "code": None,  # the three-rule alert counter of plans/controls.md §2 (confirmed by the owner 2026-10-22)
        "licence": None,
    },
    ARM_FLOOR_TOPOLOGY: {
        "class": CLASS_FLOOR_CONTROL,
        "inputs": (INPUT_TOPOLOGY,),
        "axes": (AXIS_GRAPH,),
        "code": None,  # tracebench.score_service --prediction topology: the deployment topology coarsened to services
        "licence": None,
    },
}
ROOT_CAUSE_ARMS = tuple(a for a, s in ARMS.items() if s["class"] == CLASS_ROOT_CAUSE)
CONTROL_ARMS = tuple(a for a, s in ARMS.items() if s["class"] == CLASS_FLOOR_CONTROL)
DISCOVERY_ARMS = tuple(a for a, s in ARMS.items() if s["class"] == CLASS_DISCOVERY)  # none registered yet
GRAPH_ARMS = tuple(a for a, s in ARMS.items() if AXIS_GRAPH in s["axes"])

# --- per-arm result files (one directory per arm per corpus) ----------------------------------------
RANKING_JSON = "ranking.json"  # {"cases": {"<case_id>": ["<service>", ...]}} — tracebench.score_rca's prediction shape
GRAPH_JSON = "graph.json"  # tracebench.score_service's prediction shape
SCORE_RCA_JSON = "score-rca.json"
SCORE_SERVICE_JSON = "score-service.json"

# --- registered scoring values (docs/scorers-service-rca.md in the benchmark; plans/rca-arms.md §5) ----
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
