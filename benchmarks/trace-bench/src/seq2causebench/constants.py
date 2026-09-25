# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Structural constants: names, registries and the values the PRD pins.

Nothing here is a tunable. Every algorithm knob is a required command-line
flag whose value rides the run record (`run/arguments.json`); the values below
are data semantics (token ids, file names, the arm registry, the shipped tool's
own threshold rule) or numbers the PRD itself fixes (the in-regime boundary),
which a run may report against but never choose.
"""

from tracebench.constants import (  # noqa: F401  (re-exported: one definition across the repositories)
    BOS,
    DEFAULT_FLOOR,
    EOS,
    FLOOR_SWEEP,
    GRAINS,
    METHOD_READABLE_PREFIXES,
    N_SPECIALS,
    ORDERINGS,
    OUTCOME_IDS,
    OUTCOME_NAMES,
    OUTCOME_OK,
    PAD,
    SPLITS,
    UNK,
    VARIANT_LATENT,
    VARIANT_TWIN,
    VARIANTS,
)

TOOL_NAME = "seq2cause-trace-bench"
SIBLING = (
    "alex-chadyuk/trace-cmi-bench@f54776a"  # the source of every copied or adapted module (NOTICE)
)

# --- the method under test ---------------------------------------------------------------
SHIPPED_VERSION = "0.1.9"  # the seq2cause release every "as shipped" claim refers to
SHIPPED_COMMIT = "67416a2"  # its commit on the author's main line

# --- the benchmark this directory scores --------------------------------------------------
DATASET_REPO = "chadyuk/trace-bench"  # dataset host id (public)
BENCHMARK_TAG = "v0.3.0"  # the release tag pinned in pyproject.toml
BENCHMARK_TOOL_VERSION = "0.3.0"  # manifest.tool_version every scored corpus must carry
RUNGS = ("xs", "s", "m", "l", "xl")
SEEDS = (0, 1, 2, 3, 4)
COMPLETE_MARKER = "COMPLETE"
MANIFEST_JSON = "manifest.json"
MODEL_VOCAB_JSON = "model-vocab.json"
EXPORT_STATS_JSON = "export-stats.json"
VIEWS_DIR = "views"
SEQUENCES_DIR = "sequences"
GRAPHS_DIR = "graphs"

# --- run record (one directory per run) --------------------------------------------------
RUN_DIR = "run"
ARGUMENTS_JSON = "arguments.json"
RESULTS_JSON = "results.json"
RUN_META_JSON = "run_meta.json"
ARTIFACTS_JSON = "artifacts.json"
ARTIFACTS_SCHEMA = "seq2causebench/artifacts@1"
LOG_JSONL = "log.jsonl"

# --- per-stage artifacts ------------------------------------------------------------------
PREPARE_JSON = "prepare.json"
MODEL_DIR = "model"  # Hugging Face directory format (save_pretrained)
CHECKPOINT_DIR_FMT = "checkpoint-{step:07d}"
MODEL_WEIGHTS = "model.safetensors"  # the file whose sha256 is the model hash
MATRICES_NPZ_FMT = "matrices-{probe}-{noise}-{grain}.npz"  # full strict-upper triangle per sequence
SCORES_NPZ_FMT = (
    "scores-{probe}-{noise}-{grain}.npz"  # type-pair accumulator, one column set per path
)
SWEEP_SCORES_NPZ_FMT = "scores-{probe}-{noise}-c{c}-N{n}-{grain}.npz"
SHIPPED_CUT_JSON_FMT = "shippedcut-{probe}-{noise}-c{c}-N{n}-{grain}.json"
PREDICTION_JSON_FMT = "prediction-{grain}-{arm}-{path}-{cut}.json"  # arm with '/' -> '-'
RANKING_JSON_FMT = "ranking-{grain}-{arm}-{path}.json"
RANKING_LAG_JSON_FMT = "ranking-{grain}-{arm}-{path}-lag{lag}.json"
PREDICTION_LAG_JSON_FMT = "prediction-{grain}-{arm}-{path}-lag{lag}.json"
VAL_TABLE_JSON = "val-table.json"
SEQSCORE_JSON = "seqscore.json"
FREEZES_DIR = "freezes"
RESULTS_DIR = "results"
TABLES_DIR = "tables"
PLANS_DIR = "plans"
FINDINGS_DIR = "findings"
STAGING_LEDGER = (
    "plans/staging-ledger.json"  # completed-record hashes the owner appends per rung (scenario 46)
)

# --- probes: the forward-sharing units of the shipped engine ---------------------------------
PROBE_CORE = "core"  # SampleLevelCausalDiscovery's tensor build, strategy full, ancestral history
PROBE_CLI_FULL = "cli-full"  # compute_cmi_matrix internals, strategy full, real prefix
PROBE_CLI_ATOMIC = (
    "cli-atomic"  # compute_cmi_matrix internals, strategy atomic (the shipped default)
)
PROBE_SALIENCY = "saliency"  # calc_neural_saliency (captum InputXGradient), no particles
PROBE_SHAPLEY = "shapley"  # calc_neural_shapley (captum ShapleyValueSampling), no particles
PROBES = (PROBE_CORE, PROBE_CLI_FULL, PROBE_CLI_ATOMIC, PROBE_SALIENCY, PROBE_SHAPLEY)
PARTICLE_PROBES = (PROBE_CORE, PROBE_CLI_FULL, PROBE_CLI_ATOMIC)

# the noise draw of a pass: the whole id range (v0.1.9) or real event ids only (fix PR #3)
NOISE_ALL = "all"
NOISE_REAL = "real"
NOISES = (NOISE_ALL, NOISE_REAL)

# the Bernoulli-KL algebra of the shipped estimators (fix PR #1, `seq2cause.kl.KL_MODES`)
KL_SHIPPED = "clamp"
KL_FIXED = "logspace"

# --- paths: which merged fixes a read-out runs with (Bug Policy) ------------------------------
PATH_SHIPPED = "shipped"  # v0.1.9 algebra: noise over [0, V), float32 clamp
PATH_FIXED_KL = (
    "fixed-kl"  # PR #1 only: same forward and draws as `shipped` (common random numbers)
)
PATH_FIXED = "fixed"  # every merged fix: PR #1 + PR #3 noise over real ids (a separate forward)
PATH_NONE = "none"  # the read-out has no shipped/fixed dimension (saliency, shapley)
PATHS = (PATH_SHIPPED, PATH_FIXED_KL, PATH_FIXED, PATH_NONE)
PATH_SPEC = {
    PATH_SHIPPED: {"noise": NOISE_ALL, "kl": KL_SHIPPED},
    PATH_FIXED_KL: {"noise": NOISE_ALL, "kl": KL_FIXED},
    PATH_FIXED: {"noise": NOISE_REAL, "kl": KL_FIXED},
    PATH_NONE: {"noise": None, "kl": None},
}

# --- cuts: how a type-level edge set is selected from the scores ----------------------------
CUT_FROZEN = "frozen"  # the validation-swept tau every arm uses
CUT_SHIPPED = (
    "shipped"  # the tool's own label-free pooled-percentile rule with lag decay (cli arms only)
)
CUTS = (CUT_FROZEN, CUT_SHIPPED)

# --- arms (PRD Arm Register, amended 2026-09-24) ------------------------------------------------
ARM_CORE = "trace/core"
ARM_CLI = "trace/cli"
ARM_CLI_ATOMIC = "trace/cli-atomic"
ARM_GRANGER = "baseline/granger"
ARM_SALIENCY = "baseline/saliency"
ARM_SHAPLEY = "baseline/shapley"
ARMS = {
    ARM_CORE: {
        "class": "reference",
        "probe": PROBE_CORE,
        "statistic": "lag_info_gain",
        "paths": (PATH_SHIPPED, PATH_FIXED_KL, PATH_FIXED),
        "cuts": (CUT_FROZEN,),
    },
    ARM_CLI: {
        "class": "reference",
        "probe": PROBE_CLI_FULL,
        "statistic": "mean_then_kl",
        "paths": (PATH_SHIPPED, PATH_FIXED_KL, PATH_FIXED),
        "cuts": (CUT_FROZEN, CUT_SHIPPED),
    },
    ARM_CLI_ATOMIC: {
        "class": "reference",
        "probe": PROBE_CLI_ATOMIC,
        "statistic": "atomic_kl",
        "paths": (PATH_SHIPPED, PATH_FIXED_KL, PATH_FIXED),
        "cuts": (CUT_FROZEN, CUT_SHIPPED),
    },
    ARM_GRANGER: {
        "class": "baseline",
        "probe": PROBE_CORE,
        "statistic": "granger_diff",
        "paths": (PATH_SHIPPED, PATH_FIXED),
        "cuts": (CUT_FROZEN,),
    },
    ARM_SALIENCY: {
        "class": "baseline",
        "probe": PROBE_SALIENCY,
        "statistic": "input_x_gradient_l2",
        "paths": (PATH_NONE,),
        "cuts": (CUT_FROZEN,),
    },
    ARM_SHAPLEY: {
        "class": "baseline",
        "probe": PROBE_SHAPLEY,
        "statistic": "shapley_sampling",
        "paths": (PATH_NONE,),
        "cuts": (CUT_FROZEN,),
    },
}
REFERENCE_ARMS = (ARM_CORE, ARM_CLI, ARM_CLI_ATOMIC)
BASELINE_ARMS = (ARM_GRANGER, ARM_SALIENCY, ARM_SHAPLEY)
AGGREGATIONS = ("max",)  # the type-level score of a pair is the max over occurrences;
# mean and count are recorded columns, never sub-arms
SEQUENCE_SAMPLES = ("head", "uniform")  # --sequence-sample
LR_SCHEDULES = ("cosine", "constant")
AMP_MODES = ("none", "bf16")  # --amp (pretrain) and --probe-amp (probing)
DEVICES = ("cpu", "cuda")

# the shipped tool's own threshold rule (cli.py + threshold.AdaptiveThreshold defaults, v0.1.9),
# passed explicitly as required knobs and recorded verbatim on every shipped-cut read
CLI_SHIPPED_RULE = {
    "method": "percentile",
    "per_lag": False,
    "decay": True,
    "decay_type": "exponential",
    "decay_rate": 0.3,
    "exponent": 0.5,
    "floor": None,
    "min_group_size": 8,
}
CLI_SHIPPED_DEFAULTS = {"context": 4, "particles": 32, "strategy": "atomic"}  # cli.build_arg_parser

# no arm here can emit a bidirected edge: the assumption every arm rests on (PRD scenario 6)
STRUCTURAL_LIMITATION = "causal_sufficiency"

# --- PRD-pinned values (reported against, never chosen) -------------------------------------
ORACLE_IN_REGIME = 0.1  # eps_hat below this = in regime (paper's phase transition)
CHECKPOINT_TRIGGER_RATIO = (
    1.01  # final val loss above this x its minimum -> argmin checkpoint also swept
)
