"""Code copied or adapted from the sibling harness keeps its MIT copyright notice and names both
its sources in a tracked file, and every such source file says so in its first lines."""

from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
SIBLING_LINE = "# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE"
COPYRIGHT = "Copyright (c) 2026 Alex Chadyuk, Hugo Math, Alicia Zhang, and Roy Kucukates"


def test_notice_names_both_sources_and_the_copyright_line():
    text = (BENCH / "NOTICE").read_text()
    assert "alex-chadyuk/trace-cmi-bench" in text and "f54776a" in text
    assert "benchmarks/trace-bench" in text and "14edb93c" in text
    assert COPYRIGHT in text
    assert "MIT License" in (BENCH / "LICENSE").read_text()
    # the methods under test are named with their licence state, none vendored
    for name in ("phamquiluan/baro", "phamquiluan/RCAEval", "AXinx/CausalRCA_code"):
        assert name in text


def test_every_adapted_source_file_carries_the_header():
    from tracebenchrca.constants import HARNESS, SIBLING

    assert SIBLING == "alex-chadyuk/trace-cmi-bench@f54776a" and "14edb93c" in HARNESS
    n = 0
    for path in sorted((BENCH / "src" / "tracebenchrca").glob("*.py")) + sorted((BENCH / "tests").glob("*.py")):
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:3])
        if "Adapted from" in head:
            assert SIBLING_LINE in head, path.name
            n += 1
    assert n >= 6  # record, log, pull, artifacts, staging, constants at least
