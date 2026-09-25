# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""JSON-line logging to stdout, one dict per line, flushed; the benchmark's idiom.

A run's log streams to a file on the executing host and is never part of the
record: the record is `run/*.json`. Values that are not JSON serialisable fail
loudly at the boundary rather than rendering as repr().
"""

from tracebench.log import log, now_iso  # noqa: F401

__all__ = ["log", "now_iso"]
