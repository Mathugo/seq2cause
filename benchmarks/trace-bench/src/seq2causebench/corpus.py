# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The corpus adapter: what a method sees of a trace-bench corpus (PRD scenario 1).

Every file is opened through the benchmark's own accessor,
`tracebench.allowlist.open_for_method`, which permits the raw feed and the
correlated views and refuses everything else, so no code path in this module
can reach the graphs, the labels, the oracle, the manifest or the topology;
`NotMethodReadable` propagates and fails the run. Parquet is read from the
handle the accessor returns.

A `Corpus` is one corpus directory at one ordering and one grain — one of the
four views `views/<ordering>-<grain>/`. The vocabulary is the view's own
`model-vocab.json`, never the benchmark's alphabet record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from tracebench.allowlist import NotMethodReadable, open_for_method  # noqa: F401

from .constants import (
    EXPORT_STATS_JSON,
    GRAINS,
    MODEL_VOCAB_JSON,
    ORDERINGS,
    SEQUENCES_DIR,
    SPLITS,
    VIEWS_DIR,
)

SEQUENCE_COLUMNS = ("trace_id", "ops", "outcomes", "n_spans")


class Corpus:
    def __init__(self, corpus_dir, ordering, grain):
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
        if grain not in GRAINS:
            raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")
        self.dir = Path(corpus_dir)
        self.ordering, self.grain = ordering, grain
        self.view = f"{ordering}-{grain}"
        self.view_rel = f"{VIEWS_DIR}/{self.view}"

    # --- the one door ---------------------------------------------------------------------
    def open(self, relative_path, mode="rb"):
        """Open a corpus file on behalf of the method; refuses anything outside the method-readable set."""
        return open_for_method(self.dir, relative_path, mode)

    def _json(self, relative_path):
        with self.open(relative_path, "r") as f:
            return json.load(f)

    # --- view-level records --------------------------------------------------------------------
    def vocab_json(self):
        return self._json(f"{self.view_rel}/{MODEL_VOCAB_JSON}")

    def export_stats(self):
        return self._json(f"{self.view_rel}/{EXPORT_STATS_JSON}")

    # --- sequences ------------------------------------------------------------------------------
    def parts(self, split):
        """Relative paths of the view's parquet parts for one split, in file order."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        root = self.dir / self.view_rel / SEQUENCES_DIR / f"split={split}"
        return [p.relative_to(self.dir).as_posix() for p in sorted(root.rglob("part-*.parquet"))]

    def n_rows(self, split):
        n = 0
        for rel in self.parts(split):
            with self.open(rel) as f:
                n += pq.ParquetFile(f).metadata.num_rows
        return n

    def sequences(self, split, columns=SEQUENCE_COLUMNS, batch_rows=8192):
        """Yield `(trace_id, ops, outcomes, n_spans)` per sequence of one split.
        `ops[j]` is the mechanism op id and `outcomes[j]` the int8 outcome class."""
        for rel in self.parts(split):
            with self.open(rel) as f:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=batch_rows, columns=list(columns)):
                    cols = [batch.column(c).to_pylist() for c in columns]
                    yield from zip(*cols, strict=False)

    @property
    def corpus_id(self):
        """`<rung>/<variant>/seed=<k>` when the directory follows the dataset host's layout
        (`instances/<rung>/<variant>/seed=<k>`), else the directory name."""
        parts = self.dir.resolve().parts
        if len(parts) >= 4 and parts[-4] == "instances" and parts[-1].startswith("seed="):
            return "/".join(parts[-3:])
        return self.dir.name

    def __repr__(self):
        return f"Corpus({self.dir.name!r}, ordering={self.ordering!r}, grain={self.grain!r})"
