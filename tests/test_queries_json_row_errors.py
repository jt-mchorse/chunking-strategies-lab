"""`json.loads` raises more than `JSONDecodeError`, and so must the readers catch (#244).

A deeply nested value raises `RecursionError`, and an integer literal past
CPython's 4300-digit int-string limit raises a plain `ValueError`. Both readers
caught only `JSONDecodeError`. `validate` then crashed at exit 1, the code this
CLI uses for "findings", without printing a finding or checking the rows after
it. `load_queries` raised `RecursionError` rather than its documented
`ValueError`, and neither shape carried the `path:lineno:` prefix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chunking_lab.queries import load_queries
from chunking_lab.validate import main, validate_queries

_GOOD = {
    "id": "q1",
    "question": "What is HNSW?",
    "expected_doc": "01_hnsw.md",
    "expected_snippet": "ef_construction",
}

_DEPTH = 200_000  # far past the default recursion limit on every supported version

BAD_LINES = {
    "deep_nesting": "[" * _DEPTH + "]" * _DEPTH,
    "deep_nesting_in_a_field": '{"id": ' + "[" * _DEPTH + "]" * _DEPTH + "}",
    "int_past_4300_digits": '{"id": 1' + "0" * 5000 + "}",
}


def _write(tmp_path: Path, bad: str) -> Path:
    good_after = dict(_GOOD, id="q2", question="")  # an empty question: a finding of its own
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join([json.dumps(_GOOD), bad, json.dumps(good_after)]) + "\n", encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("case", sorted(BAD_LINES))
def test_validate_reports_the_line_and_keeps_collecting(tmp_path: Path, case: str) -> None:
    report = validate_queries(_write(tmp_path, BAD_LINES[case]))
    codes = [(f.line_no, f.code) for f in report.findings]
    # The bad line is a finding, and line 3 after it was still checked.
    assert codes == [(2, "malformed_json"), (3, "empty_question")]
    assert report.n_rows == 3
    assert report.n_valid == 1


def test_recursion_reason_is_fixed_text(tmp_path: Path) -> None:
    report = validate_queries(_write(tmp_path, BAD_LINES["deep_nesting"]))
    assert report.findings[0].reason == "invalid JSON: nested too deeply to parse"


def test_int_limit_reason_names_the_limit(tmp_path: Path) -> None:
    report = validate_queries(_write(tmp_path, BAD_LINES["int_past_4300_digits"]))
    assert report.findings[0].reason.startswith("invalid JSON: Exceeds the limit")


@pytest.mark.parametrize("case", sorted(BAD_LINES))
def test_validate_cli_exits_1_with_the_finding_printed(
    tmp_path: Path, case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([str(_write(tmp_path, BAD_LINES[case]))])
    err = capsys.readouterr().err
    assert rc == 1
    assert "line 2 [malformed_json]" in err
    assert "line 3 [empty_question]" in err


@pytest.mark.parametrize("case", sorted(BAD_LINES))
def test_load_queries_raises_valueerror_with_path_and_line(tmp_path: Path, case: str) -> None:
    path = _write(tmp_path, BAD_LINES[case])
    with pytest.raises(ValueError, match=r":2: invalid JSON: ") as exc:
        load_queries(path)
    assert str(path) in str(exc.value)


def test_syntax_error_messages_unchanged(tmp_path: Path) -> None:
    """Control: a plain JSONDecodeError reads exactly as it did before #244."""
    path = _write(tmp_path, "{not json")
    report = validate_queries(path)
    assert report.findings[0].reason.startswith("invalid JSON: Expecting property name")
    with pytest.raises(ValueError, match=r":2: invalid JSON: Expecting property name .*line 1"):
        load_queries(path)
