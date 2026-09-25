"""Ordered staging (PRD scenario 46): a stage at any rung above the smallest names a
completed record of the same stage at the next-smaller rung, or an owner-dated
exception, and is refused otherwise.

Records live in object storage, so on a fresh instance `--prior-rung-record` may be
a **sha256** that must appear in the tracked staging ledger
(`plans/staging-ledger.json`, appended by the owner after each rung lands) rather than
a local path. Accepted forms:

- `smallest-rung` — allowed only at the smallest rung;
- a path to a completed run directory of the same stage at the previous rung
  (`run/results.json` with status ok), whose results file is hashed and recorded;
- a 64-hex sha256 present in the ledger for the same stage at the previous rung;
- a path to an owner-dated exception file (`*.md` carrying an ISO date), recorded verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

from .constants import RESULTS_JSON, RUN_DIR, RUNGS, SMALLEST_RUNG_TOKEN, STAGING_LEDGER_SCHEMA
from .record import read_json, sha256_file

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


class StagingRefusal(RuntimeError):
    pass


def previous_rung(rung):
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {RUNGS}, got {rung!r}")
    i = RUNGS.index(rung)
    return None if i == 0 else RUNGS[i - 1]


def read_ledger(path):
    ledger = read_json(path)
    if ledger.get("schema") != STAGING_LEDGER_SCHEMA:
        raise StagingRefusal(f"{path} is not a staging ledger ({STAGING_LEDGER_SCHEMA})")
    return ledger


def require_prior_rung(prior, rung, stage, ledger_path):
    """Returns the staging facts to record, or raises `StagingRefusal`."""
    prev = previous_rung(rung)
    if prev is None:
        if prior != SMALLEST_RUNG_TOKEN:
            raise StagingRefusal(
                f"rung {rung!r} is the smallest; pass --prior-rung-record {SMALLEST_RUNG_TOKEN}"
            )
        return {"rung": rung, "previous_rung": None, "form": SMALLEST_RUNG_TOKEN}
    if prior == SMALLEST_RUNG_TOKEN:
        raise StagingRefusal(
            f"rung {rung!r} needs a completed {stage} record at rung {prev!r} (PRD scenario 46)"
        )
    if SHA_RE.match(prior):
        ledger = read_ledger(ledger_path)
        for e in ledger.get("entries", []):
            if (
                e.get("results_sha256") == prior
                and e.get("stage") == stage
                and e.get("rung") == prev
            ):
                return {
                    "rung": rung,
                    "previous_rung": prev,
                    "form": "ledger",
                    "results_sha256": prior,
                    "entry": e,
                }
        raise StagingRefusal(
            f"sha256 {prior[:12]}… is not in {ledger_path} for stage {stage!r} at rung {prev!r}"
        )
    path = Path(prior)
    if path.suffix == ".md":
        if not path.exists():
            raise StagingRefusal(f"exception file {path} does not exist")
        text = path.read_text(encoding="utf-8")
        m = DATE_RE.search(text)
        if not m:
            raise StagingRefusal(
                f"exception file {path} carries no ISO date; an owner-dated exception is required"
            )
        return {
            "rung": rung,
            "previous_rung": prev,
            "form": "exception",
            "file": str(path),
            "date": m.group(0),
            "sha256": sha256_file(path),
        }
    results = path / RUN_DIR / RESULTS_JSON
    if not results.exists():
        raise StagingRefusal(f"{path} holds no completed run ({RUN_DIR}/{RESULTS_JSON})")
    r = read_json(results)
    if r.get("status") != "ok":
        raise StagingRefusal(f"{results} has status {r.get('status')!r}")
    if r.get("rung") != prev:
        raise StagingRefusal(
            f"{results} is a rung {r.get('rung')!r} record; rung {rung!r} needs rung {prev!r}"
        )
    if r.get("stage", r.get("command")) != stage:
        raise StagingRefusal(
            f"{results} is a {r.get('stage', r.get('command'))!r} record; this stage is {stage!r}"
        )
    return {
        "rung": rung,
        "previous_rung": prev,
        "form": "record",
        "path": str(path),
        "results_sha256": sha256_file(results),
    }


__all__ = ["StagingRefusal", "previous_rung", "read_ledger", "require_prior_rung"]
