# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Structural constants: names, registries and the values the PRD pins.

Nothing here is a tunable. Every algorithm knob is a required command-line
flag whose value rides the run record (`run/arguments.json`); the values below
are data semantics (token ids, file names, the arm registry, the shipped tool's
own threshold rule) or numbers the PRD itself fixes (the in-regime boundary),
which a run may report against but never choose.
"""

from tracebench.constants import (  # noqa: F401  (re-exported: one definition across the repositories)
    ALPHABET_JSON,
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
    PRIOR_JSON,
    SPLITS,
    TOPOLOGY_DIR,
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
SHIPPED_VERSION = "0.1.9"  # the seq2cause release every landed cell and "as shipped" claim refers to
SHIPPED_COMMIT = "67416a2"  # its commit on the author's main line
# the releases merged on 2026-10-30 (v0.3.0 `b63b4328`, RUN.md D-SB-18: `calc_lag_cmi` and the
# `prepend` / `none` context modes of the core class) and 2026-10-31 (v0.3.1 `280ed948`, D-SB-19:
# `compute_cmi_matrix(score="cmi")`, `prepend_len` / `n_pasts` on the cli path); the "overwrite" and
# information-gain layouts are byte-identical to v0.1.9 on the fixture (the golden check of each
# deviation), so every landed cell stands; run records written after the merges say 0.3.1
CONTEXT_FAMILY_VERSION = "0.3.1"
CONTEXT_FAMILY_COMMIT = "280ed948"

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
# the context family (plans/reference-arms.md, addendum 2026-10-31; RUN.md D-SB-20, D-SB-23)
PROBE_CORE_KEEP = "core-keep"  # the core build with v0.3.0 `context_mode = "none"`: the first g = c
# observed tokens kept, nothing sampled, every later token tested (the window offset stays c)
PROBE_CLI_ATOMIC_CMI = "cli-atomic-cmi"  # v0.3.1 compute_cmi_matrix(score="cmi", strategy="atomic")
# internals: the cause-absent branch drawn from the model, two forwards per sequence
PROBES = (
    PROBE_CORE,
    PROBE_CORE_KEEP,
    PROBE_CLI_FULL,
    PROBE_CLI_ATOMIC,
    PROBE_CLI_ATOMIC_CMI,
    PROBE_SALIENCY,
    PROBE_SHAPLEY,
)
PARTICLE_PROBES = (
    PROBE_CORE,
    PROBE_CORE_KEEP,
    PROBE_CLI_FULL,
    PROBE_CLI_ATOMIC,
    PROBE_CLI_ATOMIC_CMI,
)
CORE_PROBES = (PROBE_CORE, PROBE_CORE_KEEP)  # SampleLevelCausalDiscovery's tensor build
SHIPPED_CUT_PROBES = (  # the cli probes: every matrix kept, the tool's own cut fitted per cell
    PROBE_CLI_FULL,
    PROBE_CLI_ATOMIC,
    PROBE_CLI_ATOMIC_CMI,
)
# how the past before the tested tokens is built (v0.3.0 `sampling.context_mode`; core probes only;
# `prepend` is not run — the addendum's P1): the window offset equals c in both modes
CONTEXT_OVERWRITE = "overwrite"  # the first g observed tokens kept, g..c-1 sampled, c..L-1 tested
CONTEXT_NONE = "none"  # the first g = c observed tokens kept, nothing sampled, c..L-1 tested
CONTEXT_MODES = (CONTEXT_OVERWRITE, CONTEXT_NONE)
CONTEXT_MODE_OF_PROBE = {PROBE_CORE: CONTEXT_OVERWRITE, PROBE_CORE_KEEP: CONTEXT_NONE}

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

# --- arms (PRD Arm Register, amended 2026-09-24; the floor family of plans/floors.md, 2026-10-09) ---
ARM_CORE = "trace/core"
ARM_CLI = "trace/cli"
ARM_CLI_ATOMIC = "trace/cli-atomic"
# the context family (addendum 2026-10-31): score × context on the core probe, one atomic CMI arm
ARM_CORE_IG_OVERWRITE = "trace/core-ig-overwrite"  # = trace/core's fixed-kl column, by construction
ARM_CORE_CMI_OVERWRITE = "trace/core-cmi-overwrite"
ARM_CORE_IG_KEEP = "trace/core-ig-keep"
ARM_CORE_CMI_KEEP = "trace/core-cmi-keep"
ARM_CLI_ATOMIC_CMI_KEEP = "trace/cli-atomic-cmi-keep"
# the statistics the engine reads out (`engine.core_readouts` iterates a pass's arms by statistic)
STAT_LAG_INFO_GAIN = "lag_info_gain"
STAT_LAG_CMI = "lag_cmi"  # v0.3.0 calc_lag_cmi on the information gain's log-probabilities
STAT_ATOMIC_CMI = "atomic_cmi"  # v0.3.1 _cmi_matrix_atomic: the model-drawn cause-absent branch
STAT_GRANGER = "granger_diff"
# where a CMI column's corrupted-cell counts come from (D-SB-22): calc_lag_cmi fills none, so the
# core CMI columns carry the information-gain counts of the same pass; the atomic CMI has none
COUNTS_SOURCE = {STAT_LAG_CMI: "ig-same-pass", STAT_ATOMIC_CMI: "none"}
ARM_GRANGER = "baseline/granger"
ARM_SALIENCY = "baseline/saliency"
ARM_SHAPLEY = "baseline/shapley"
ARM_FLOOR_TOPOLOGY = (
    "floor/topology"  # the shipped deployment topology, callee -> caller, score p_call
)
ARM_FLOOR_BIGRAM = (
    "floor/bigram"  # the views only: share of sequences in which b immediately follows a
)
# a floor's score table names its own pass word where a probe would stand (plans/floors.md)
PROBE_FLOOR_TOPOLOGY = "topology"
PROBE_FLOOR_BIGRAM = "bigram"
FLOOR_PROBES = (PROBE_FLOOR_TOPOLOGY, PROBE_FLOOR_BIGRAM)
NOISE_NONE = "none"  # a floor has no noise draw
MODEL_NONE = "none"  # the model hash of every model-free record (RUN.md D-SB-16)
FLOOR_STAGE = "floors"
FAMILY_MODEL = "model"  # a freeze of model-bound cells (the arms)
FAMILY_FLOOR = "floor"  # a freeze of floor cells only (`…-floors.json`)
ARMS = {
    ARM_CORE: {
        "class": "reference",
        "probe": PROBE_CORE,
        "statistic": STAT_LAG_INFO_GAIN,
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
        "statistic": STAT_GRANGER,
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
    ARM_FLOOR_TOPOLOGY: {
        "class": "floor",
        "probe": PROBE_FLOOR_TOPOLOGY,
        "statistic": "p_call",
        "paths": (PATH_NONE,),
        "cuts": (CUT_FROZEN,),
    },
    ARM_FLOOR_BIGRAM: {
        "class": "floor",
        "probe": PROBE_FLOOR_BIGRAM,
        "statistic": "bigram_share",
        "paths": (PATH_NONE,),
        "cuts": (CUT_FROZEN,),
    },
    # the context family: the fixed-kl path only on the core probes (v0.3.0's own CMI defaults:
    # log-space KL, noise over [0, V)); both paths and both cuts on the atomic CMI arm (D-SB-23)
    ARM_CORE_IG_OVERWRITE: {
        "class": "reference",
        "probe": PROBE_CORE,
        "statistic": STAT_LAG_INFO_GAIN,
        "paths": (PATH_FIXED_KL,),
        "cuts": (CUT_FROZEN,),
    },
    ARM_CORE_CMI_OVERWRITE: {
        "class": "reference",
        "probe": PROBE_CORE,
        "statistic": STAT_LAG_CMI,
        "paths": (PATH_FIXED_KL,),
        "cuts": (CUT_FROZEN,),
    },
    ARM_CORE_IG_KEEP: {
        "class": "reference",
        "probe": PROBE_CORE_KEEP,
        "statistic": STAT_LAG_INFO_GAIN,
        "paths": (PATH_FIXED_KL,),
        "cuts": (CUT_FROZEN,),
    },
    ARM_CORE_CMI_KEEP: {
        "class": "reference",
        "probe": PROBE_CORE_KEEP,
        "statistic": STAT_LAG_CMI,
        "paths": (PATH_FIXED_KL,),
        "cuts": (CUT_FROZEN,),
    },
    ARM_CLI_ATOMIC_CMI_KEEP: {
        "class": "reference",
        "probe": PROBE_CLI_ATOMIC_CMI,
        "statistic": STAT_ATOMIC_CMI,
        "paths": (PATH_FIXED_KL, PATH_FIXED),
        "cuts": (CUT_FROZEN, CUT_SHIPPED),
    },
}
REFERENCE_ARMS = (ARM_CORE, ARM_CLI, ARM_CLI_ATOMIC)  # the landed reference arms; never extended
CONTEXT_ARMS = (  # the context family of 2026-10-31: its own freeze family, its own results tree
    ARM_CORE_IG_OVERWRITE,
    ARM_CORE_CMI_OVERWRITE,
    ARM_CORE_IG_KEEP,
    ARM_CORE_CMI_KEEP,
    ARM_CLI_ATOMIC_CMI_KEEP,
)
TAU_GRID_ARMS = REFERENCE_ARMS + CONTEXT_ARMS  # the arms cut on the absolute `--taus` grid
FAMILY_CONTEXT = "context"  # a freeze of context-family cells only (`…-context.json`, D-SB-21)
CONTEXT_FAMILY_GRID = {  # the registered grid of the family (addendum 2026-10-31 §4): N = 8;
    # c per probe and grain; the freeze keeps only rows inside it
    "N": (8,),
    "c": {
        PROBE_CORE: {"request": (1, 2, 3), "session": (2, 4, 8)},
        PROBE_CORE_KEEP: {"request": (1,), "session": (1,)},
        PROBE_CLI_ATOMIC_CMI: {"request": (1,), "session": (1,)},
    },
}
BASELINE_ARMS = (ARM_GRANGER, ARM_SALIENCY, ARM_SHAPLEY)
FLOOR_ARMS = (ARM_FLOOR_TOPOLOGY, ARM_FLOOR_BIGRAM)  # model-free; no (c, N, g); plans/floors.md
FLOOR_ARM_OF_PROBE = {
    PROBE_FLOOR_TOPOLOGY: ARM_FLOOR_TOPOLOGY,
    PROBE_FLOOR_BIGRAM: ARM_FLOOR_BIGRAM,
}
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
MODEL_CHOICES = ("last", "argmin-val")  # --model-choice (pretrain): which checkpoint is model/
# (the 1.01x checkpoint trigger of 2026-09-24 was withdrawn on 2026-09-25: plans/caps.md addendum)

# --- added at M6 ---------------------------------------------------------------------------------
SHIPPED_CUT_READ_JSON_FMT = (
    "shippedcut-{grain}-{arm}-{path}.json"  # a discover read's shipped-cut record
)
SEQUENCES_JSON_FMT = "sequences-{grain}.json"  # the probed sequences (trace ids, lengths) of a read
SMALLEST_RUNG_TOKEN = "smallest-rung"  # --prior-rung-record at the smallest rung
STAGING_LEDGER_SCHEMA = "seq2causebench/staging-ledger@1"
