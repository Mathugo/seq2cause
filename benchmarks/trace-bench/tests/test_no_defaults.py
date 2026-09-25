# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""PRD non-negotiable 11 and scenario 43: no algorithm knob has a default, so a
run's recorded arguments reconstruct it completely by construction.

Every `default=` in `src/seq2causebench/` must be one of the structural
exceptions listed here — transport knobs that say where bytes live, never how
the method behaves. Anything else fails the build. (The shipped seq2cause
package keeps its own defaults; this gate covers the harness only.)"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "seq2causebench"

# file -> substrings, one of which must appear on any line of that file that carries `default=`
ALLOWED = {
    # object-store transport, values from the environment only (the substrings sit on the `default=` line itself)
    "artifacts.py": ('os.environ.get("AWS_PROFILE")', "os.environ.get(ENV_URI)"),
}

DEFAULT_RE = re.compile(r"\bdefault\s*=")


def test_no_hyperparameter_defaults():
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        allowed = ALLOWED.get(path.name, ())
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if DEFAULT_RE.search(line) and not any(a in line for a in allowed):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_every_cli_module_requires_its_knobs():
    """Every module with a parser declares its arguments `required=True` (or as
    positionals / store_true flags); `add_argument` with neither is a knob with
    an implicit default of None."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "add_argument(" not in text:
            continue
        allowed = ALLOWED.get(path.name, ())
        for m in re.finditer(r"add_argument\((.*?)\)\n", text, flags=re.S):
            call = m.group(1)
            if any(a in call for a in allowed):
                continue
            first = call.strip().split(",")[0].strip()
            positional = not first.startswith(("'-", '"-'))
            if (
                positional
                or "required=True" in call
                or "store_true" in call
                or "store_false" in call
            ):
                continue
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.name}:{line}: add_argument({call.strip()[:80]}...)")
    assert not offenders, "\n".join(offenders)
