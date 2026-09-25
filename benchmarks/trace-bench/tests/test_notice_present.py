"""PRD non-negotiable 14 and scenario 33: code copied or adapted from the sibling
harness keeps its MIT copyright notice and names its source commit in a tracked
file, and every such source file says so in its first lines."""

from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
SIBLING_LINE = "# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE"
COPYRIGHT = "Copyright (c) 2026 Alex Chadyuk, Hugo Math, Alicia Zhang, and Roy Kucukates"


def test_notice_names_the_sibling_commit_and_its_copyright_line():
    text = (BENCH / "NOTICE").read_text()
    assert "alex-chadyuk/trace-cmi-bench" in text
    assert "f54776a" in text
    assert COPYRIGHT in text
    assert "MIT License" in (BENCH / "LICENSE").read_text()


def test_every_adapted_source_file_carries_the_header():
    from seq2causebench.constants import SIBLING

    assert SIBLING == "alex-chadyuk/trace-cmi-bench@f54776a"
    for path in sorted((BENCH / "src" / "seq2causebench").glob("*.py")) + sorted(
        (BENCH / "tests").glob("*.py")
    ):
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:3])
        if "Adapted from" in head:
            assert SIBLING_LINE in head, path.name
