"""Tests for ``validate_queries`` and the ``python -m chunking_lab.validate`` CLI (#37).

Coverage matrix:

- happy path on the shipped ``data/queries.jsonl`` (12 rows, all valid)
  → ``ok=True``, no findings.
- accumulating-errors path: synthetic file with multiple bad rows
  surfaces every finding in line-number order, not failing fast.
- one parametrized positive case per finding code (15 codes: malformed_json,
  not_an_object, four missing_*, four non_string_*, four empty_*,
  duplicate_id; ``expected_doc_not_found`` and ``empty`` are tested
  separately because they require non-default args / inputs).
- duplicate-``id`` detection: validator reports the duplicate and does
  not count the shadowed row as a second valid row.
- blank lines silently skipped: present in input but absent from
  ``n_rows`` and from the findings.
- ``--corpus-dir`` cross-file check: happy path (no findings) and bad
  case (every row's ``expected_doc`` missing → one finding per row).
- empty-file `empty` finding with ``line_no=0``.
- missing file: ``FileNotFoundError`` propagates from the library; CLI
  surfaces exit code 2.
- missing ``--corpus-dir``: same.
- ``ValidationReport.to_dict`` JSON shape is stable.
- Dataclasses are frozen.
- CLI: clean fixture exits 0 with ``ok:`` summary on stdout, empty stderr.
- CLI: malformed fixture exits 1 with one stderr line per finding.
- CLI: ``--json`` emits report dict and respects exit code.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_lab.validate import (
    REQUIRED_FIELDS,
    ValidationFinding,
    ValidationReport,
    validate_queries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_QUERIES = REPO_ROOT / "data" / "queries.jsonl"
SHIPPED_CORPUS = REPO_ROOT / "data" / "corpus"


def _write_jsonl(path: Path, rows: list[dict | str]) -> None:
    """Write rows to ``path``; ``str`` rows are emitted verbatim so a test
    can inject a malformed line. ``dict`` rows go through ``json.dumps``.
    """
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, str):
                fh.write(row + "\n")
            else:
                fh.write(json.dumps(row) + "\n")


def _valid_row(qid: str = "q01") -> dict[str, str]:
    return {
        "id": qid,
        "question": "What parameter controls HNSW build candidate list size?",
        "expected_doc": "01_hnsw.md",
        "expected_snippet": "ef_construction",
    }


# ---------------------------------------------------------------------------
# Library: validate_queries
# ---------------------------------------------------------------------------


def test_shipped_queries_validate_clean() -> None:
    report = validate_queries(SHIPPED_QUERIES)
    assert report.ok, f"shipped data/queries.jsonl unexpectedly has findings: {report.findings}"
    assert report.n_rows == 12
    assert report.n_valid == 12


def test_validate_queries_handles_utf8_bom(tmp_path: Path) -> None:
    # Issue #93: parity with load_queries — a UTF-8 BOM must not produce a
    # spurious malformed_json finding on line 1. utf-8-sig strips it; `.strip()`
    # does not (U+FEFF is not whitespace).
    body = json.dumps(_valid_row("q01")) + "\n" + json.dumps(_valid_row("q02")) + "\n"
    p = tmp_path / "bom.jsonl"
    p.write_text(body, encoding="utf-8-sig")  # prepends the BOM
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")

    report = validate_queries(p)
    assert report.ok, f"BOM-prefixed file unexpectedly has findings: {report.findings}"
    assert report.n_rows == 2
    assert report.n_valid == 2


def test_shipped_queries_with_corpus_dir_validate_clean() -> None:
    """Cross-file invariant holds for the pinned substrate."""
    report = validate_queries(SHIPPED_QUERIES, corpus_dir=SHIPPED_CORPUS)
    assert report.ok
    assert report.n_valid == 12


def test_collects_multiple_findings_not_failing_fast(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("q01"),
            "{not valid json",
            {**_valid_row("q02"), "question": ""},  # empty_question
            {k: v for k, v in _valid_row("q03").items() if k != "expected_doc"},  # missing
            _valid_row("q04"),
        ],
    )
    report = validate_queries(p)
    assert not report.ok
    assert report.n_rows == 5
    assert report.n_valid == 2
    codes = [f.code for f in report.findings]
    line_nos = [f.line_no for f in report.findings]
    assert codes == ["malformed_json", "empty_question", "missing_expected_doc"]
    assert line_nos == [2, 3, 4]


@pytest.mark.parametrize(
    ("row", "code"),
    [
        # parse / shape
        ("{not valid json", "malformed_json"),
        ('"bare_string"', "not_an_object"),
        # missing_<field>
        *(
            ({k: v for k, v in _valid_row().items() if k != field}, f"missing_{field}")
            for field in REQUIRED_FIELDS
        ),
        # non_string_<field>
        *(({**_valid_row(), field: 123}, f"non_string_{field}") for field in REQUIRED_FIELDS),
        # empty_<field>
        *(({**_valid_row(), field: ""}, f"empty_{field}") for field in REQUIRED_FIELDS),
        # empty_<field> via whitespace-only (#92): a blank field is flagged under
        # the same code as a literal-empty one, in lockstep with the loader.
        *(({**_valid_row(), field: "   "}, f"empty_{field}") for field in REQUIRED_FIELDS),
    ],
)
def test_one_positive_case_per_finding_code(tmp_path: Path, row: dict | str, code: str) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [row])
    report = validate_queries(p)
    assert not report.ok
    assert len(report.findings) == 1, f"{code} produced unexpected findings: {report.findings}"
    assert report.findings[0].code == code
    assert report.findings[0].line_no == 1
    assert report.n_valid == 0


def test_duplicate_id_reports_second_occurrence(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("dup"),
            _valid_row("unique"),
            _valid_row("dup"),  # shadow
        ],
    )
    report = validate_queries(p)
    assert not report.ok
    assert report.n_rows == 3
    assert report.n_valid == 2
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.code == "duplicate_id"
    assert finding.line_no == 3
    assert "line 1" in finding.reason


def test_duplicate_id_and_field_error_on_same_row_both_reported(tmp_path: Path) -> None:
    # Collecting mode must surface every finding in one pass: a row that is both
    # a duplicate id and field-invalid used to report only the field error
    # (early continue skipped the dup check). Both must now appear.
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("dup"),
            {**_valid_row("dup"), "question": ""},  # duplicate id AND empty_question
        ],
    )
    report = validate_queries(p)
    assert not report.ok
    assert report.n_rows == 2
    assert report.n_valid == 1
    codes = {f.code for f in report.findings}
    assert codes == {"empty_question", "duplicate_id"}
    # Order within the row: field findings first, then the duplicate-id finding.
    row2 = [f for f in report.findings if f.line_no == 2]
    assert [f.code for f in row2] == ["empty_question", "duplicate_id"]


def test_valid_id_on_field_invalid_row_still_shadows_later_reuse(tmp_path: Path) -> None:
    # A row with a valid id but another field error still registers its id, so a
    # later clean row reusing it is flagged as a duplicate (the id collision is
    # real regardless of the first row's unrelated field problem).
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            {**_valid_row("dup"), "question": ""},  # valid id "dup", empty_question
            _valid_row("dup"),  # clean row reusing the id
        ],
    )
    report = validate_queries(p)
    assert not report.ok
    assert report.n_valid == 0
    dup = [f for f in report.findings if f.code == "duplicate_id"]
    assert len(dup) == 1
    assert dup[0].line_no == 2
    assert "line 1" in dup[0].reason


def test_empty_file_surfaces_empty_finding(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    p.write_text("", encoding="utf-8")
    report = validate_queries(p)
    assert not report.ok
    assert report.n_rows == 0
    assert report.n_valid == 0
    assert len(report.findings) == 1
    assert report.findings[0].code == "empty"
    assert report.findings[0].line_no == 0


def test_blank_lines_silently_skipped(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    p.write_text("\n\n" + json.dumps(_valid_row()) + "\n\n", encoding="utf-8")
    report = validate_queries(p)
    assert report.ok
    assert report.n_rows == 1
    assert report.n_valid == 1


def test_corpus_dir_check_flags_missing_expected_doc(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("q01"),  # expected_doc = 01_hnsw.md, present
            {**_valid_row("q02"), "expected_doc": "no_such_file.md"},
        ],
    )
    report = validate_queries(p, corpus_dir=SHIPPED_CORPUS)
    assert not report.ok
    assert report.n_rows == 2
    # First row valid; second row shape-valid but expected_doc missing → not counted as valid.
    assert report.n_valid == 1
    assert len(report.findings) == 1
    assert report.findings[0].code == "expected_doc_not_found"
    assert report.findings[0].line_no == 2
    assert "no_such_file.md" in report.findings[0].reason


def _make_corpus(tmp_path: Path) -> Path:
    """A corpus dir with one real ``*.md`` doc, a non-``.md`` file, a directory
    named ``*.md``, and a differently-cased ``*.md`` — the four things a raw
    ``.exists()`` check confused with a loaded document (#98)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "01_hnsw.md").write_text("# HNSW\nef_construction controls the build list.\n")
    (corpus / "readme.txt").write_text("not a markdown document\n")
    (corpus / "notes.md").mkdir()  # a *directory* whose name ends in .md
    (corpus / "Guide.md").write_text("# Guide\nprose.\n")
    return corpus


def test_corpus_dir_check_flags_non_md_file(tmp_path: Path) -> None:
    # A non-.md file exists on disk under the corpus dir but is never enumerated
    # by load_corpus (`glob("*.md")`), so a query pointing at it recalls nothing
    # (recall@k silently 0.0). The old `.exists()` check passed it; membership
    # against the real *.md set must flag it (#98).
    corpus = _make_corpus(tmp_path)
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [{**_valid_row("q01"), "expected_doc": "readme.txt"}])
    report = validate_queries(p, corpus_dir=corpus)
    assert not report.ok
    assert report.n_valid == 0
    assert [f.code for f in report.findings] == ["expected_doc_not_found"]
    assert "readme.txt" in report.findings[0].reason


def test_corpus_dir_check_flags_md_directory(tmp_path: Path) -> None:
    # A directory named `*.md` matches `glob("*.md")` but isn't a file, so
    # load_corpus would crash reading it (IsADirectoryError). The validator must
    # flag a query referencing it rather than green-light a run that can't load
    # the corpus (#98).
    corpus = _make_corpus(tmp_path)
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [{**_valid_row("q01"), "expected_doc": "notes.md"}])
    report = validate_queries(p, corpus_dir=corpus)
    assert not report.ok
    assert report.n_valid == 0
    assert [f.code for f in report.findings] == ["expected_doc_not_found"]


def test_corpus_dir_check_flags_case_mismatch(tmp_path: Path) -> None:
    # load_corpus keys documents by exact `path.name` and metrics matches
    # expected_doc as an exact string, so `guide.md` never matches on-disk
    # `Guide.md` (recall 0.0). A raw `.exists()` slips this on a case-insensitive
    # FS; exact-string set membership flags it on every FS (#98).
    corpus = _make_corpus(tmp_path)
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [{**_valid_row("q01"), "expected_doc": "guide.md"}])
    report = validate_queries(p, corpus_dir=corpus)
    assert not report.ok
    assert [f.code for f in report.findings] == ["expected_doc_not_found"]


def test_corpus_dir_check_accepts_real_md_document(tmp_path: Path) -> None:
    # Over-rejection guard: a real *.md file (exact name) must still validate
    # clean, so the tightened membership check doesn't reject valid queries.
    corpus = _make_corpus(tmp_path)
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            {**_valid_row("q01"), "expected_doc": "01_hnsw.md"},
            {**_valid_row("q02"), "expected_doc": "Guide.md"},
        ],
    )
    report = validate_queries(p, corpus_dir=corpus)
    assert report.ok
    assert report.n_valid == 2
    assert report.findings == ()


def test_corpus_dir_check_skipped_when_not_provided(tmp_path: Path) -> None:
    """``expected_doc_not_found`` only fires when corpus_dir is given."""
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [{**_valid_row(), "expected_doc": "no_such_file.md"}])
    # Without corpus_dir, the row is fully valid.
    report = validate_queries(p)
    assert report.ok
    assert report.n_valid == 1
    assert report.findings == ()


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        validate_queries(p)


def test_missing_corpus_dir_raises_file_not_found(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row()])
    with pytest.raises(FileNotFoundError):
        validate_queries(p, corpus_dir=tmp_path / "no_such_dir")


def test_report_to_dict_is_json_stable(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row("a"), {**_valid_row("b"), "question": ""}])
    report = validate_queries(p)
    d = report.to_dict()
    assert set(d.keys()) == {"path", "ok", "n_rows", "n_valid", "findings"}
    assert d["ok"] is False
    assert d["n_rows"] == 2
    assert d["n_valid"] == 1
    assert all(set(f.keys()) == {"line_no", "reason", "code"} for f in d["findings"])
    assert json.loads(json.dumps(d)) == d


def test_finding_and_report_are_frozen_dataclasses() -> None:
    assert dataclasses.is_dataclass(ValidationFinding)
    assert dataclasses.is_dataclass(ValidationReport)
    f = ValidationFinding(line_no=1, reason="x", code="malformed_json")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.line_no = 2  # type: ignore[misc]
    r = ValidationReport(path="x", n_rows=0, n_valid=0, findings=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.n_rows = 1  # type: ignore[misc]


def test_required_fields_tuple_is_the_documented_four() -> None:
    """Lock REQUIRED_FIELDS against accidental schema drift."""
    assert REQUIRED_FIELDS == ("id", "question", "expected_doc", "expected_snippet")


# ---------------------------------------------------------------------------
# CLI: python -m chunking_lab.validate
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chunking_lab.validate", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_clean_queries_exit_zero(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row("a"), _valid_row("b")])
    proc = _run_cli(str(p))
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert proc.stdout.startswith("ok:")
    assert "rows=2" in proc.stdout
    assert "valid=2" in proc.stdout
    assert "findings=0" in proc.stdout


def test_cli_malformed_queries_exit_one_with_per_finding_stderr(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("ok"),
            "{not valid json",
            {**_valid_row("dup"), "question": ""},
        ],
    )
    proc = _run_cli(str(p))
    assert proc.returncode == 1
    assert proc.stdout.startswith("fail:")
    assert "rows=3" in proc.stdout
    # row 1 valid; row 2 malformed_json; row 3 has empty_question and so
    # doesn't count as valid. Only the first row is valid.
    assert "valid=1" in proc.stdout
    assert "findings=2" in proc.stdout
    err_lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert len(err_lines) == 2
    assert err_lines[0].startswith("line 2 [malformed_json]")
    assert err_lines[1].startswith("line 3 [empty_question]")


def test_cli_json_flag_emits_report_dict_and_respects_exit_code(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row("ok"), {**_valid_row("bad"), "question": ""}])
    proc = _run_cli(str(p), "--json")
    assert proc.returncode == 1
    assert proc.stderr == ""
    parsed = json.loads(proc.stdout)
    assert parsed["ok"] is False
    assert parsed["n_rows"] == 2
    assert parsed["n_valid"] == 1
    assert len(parsed["findings"]) == 1
    assert parsed["findings"][0]["code"] == "empty_question"
    assert parsed["findings"][0]["line_no"] == 2


def test_cli_corpus_dir_check_flags_missing(tmp_path: Path) -> None:
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [{**_valid_row(), "expected_doc": "no_such_file.md"}])
    proc = _run_cli(str(p), "--corpus-dir", str(SHIPPED_CORPUS))
    assert proc.returncode == 1
    assert "expected_doc_not_found" in proc.stderr


def test_cli_missing_file_exit_two(tmp_path: Path) -> None:
    proc = _run_cli(str(tmp_path / "no.jsonl"))
    assert proc.returncode == 2
    assert "file not found" in proc.stderr


def test_cli_shipped_queries_clean_with_corpus_dir() -> None:
    proc = _run_cli(str(SHIPPED_QUERIES), "--corpus-dir", str(SHIPPED_CORPUS))
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert "rows=12" in proc.stdout
    assert "valid=12" in proc.stdout


# ---------------------------------------------------------------------------
# CLI: --out sink parity (#45) — propagation of llm-eval-harness#66
# ---------------------------------------------------------------------------


def test_cli_out_writes_human_summary_to_file_not_stdout(tmp_path: Path) -> None:
    """``--out`` writes the human-readable summary to disk; stdout stays
    silent (parity with llm-eval-harness validate --out, #66)."""
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row("a"), _valid_row("b")])
    out = tmp_path / "report.txt"
    proc = _run_cli(str(p), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", f"stdout must be silent when --out is set; got {proc.stdout!r}"
    body = out.read_text(encoding="utf-8")
    assert body.startswith("ok:"), body
    assert body.endswith("\n"), "trailing newline required for parity"


def test_cli_out_writes_json_payload_to_file(tmp_path: Path) -> None:
    """``--out`` + ``--json`` writes the report dict as JSON to disk;
    stdout silent; the file parses cleanly and carries the expected shape."""
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row("ok"), {**_valid_row("bad"), "question": ""}])
    out = tmp_path / "report.json"
    proc = _run_cli(str(p), "--json", "--out", str(out))
    assert proc.returncode == 1
    assert proc.stdout == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["findings"][0]["code"] == "empty_question"


def test_cli_out_creates_parent_dirs(tmp_path: Path) -> None:
    """``atomic_write_text`` does ``parent.mkdir(parents=True)``; confirm
    the validate CLI inherits that behavior so a nested observability
    directory doesn't need pre-creation."""
    p = tmp_path / "queries.jsonl"
    _write_jsonl(p, [_valid_row("a")])
    out = tmp_path / "nested" / "sink" / "report.txt"
    proc = _run_cli(str(p), "--out", str(out))
    assert proc.returncode == 0
    assert out.exists()
    assert out.parent.is_dir()


def test_cli_out_overwrites_atomically(tmp_path: Path) -> None:
    """Two successive writes to the same path leave the second payload —
    not the concatenation, not a half-written file. No tempfile leftovers
    under the destination's parent."""
    clean = tmp_path / "clean.jsonl"
    _write_jsonl(clean, [_valid_row("a")])
    out = tmp_path / "report.txt"
    _run_cli(str(clean), "--out", str(out))
    body1 = out.read_text(encoding="utf-8")

    bad = tmp_path / "bad.jsonl"
    _write_jsonl(bad, [{**_valid_row("bad"), "question": ""}])
    _run_cli(str(bad), "--out", str(out))
    body2 = out.read_text(encoding="utf-8")
    assert body1 != body2
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".report.txt.")]
    assert leftovers == [], leftovers


def test_cli_out_findings_still_print_to_stderr(tmp_path: Path) -> None:
    """``--out`` covers stdout only — stderr stays the operator's
    diagnostic channel so a CI step capturing stdout to a file still sees
    per-finding lines on stderr. Parity with the existing
    no-``--out``/findings/stderr contract."""
    p = tmp_path / "queries.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("ok"),
            {**_valid_row("dup"), "question": ""},
        ],
    )
    out = tmp_path / "report.txt"
    proc = _run_cli(str(p), "--out", str(out))
    assert proc.returncode == 1
    assert "empty_question" in proc.stderr
    assert proc.stdout == ""
    body = out.read_text(encoding="utf-8")
    assert body.startswith("fail:"), body


def test_cli_out_not_written_on_file_not_found(tmp_path: Path) -> None:
    """Exit-2 (file-not-found) path raises before rendering, so ``--out``
    must NOT touch disk — keeps the failure mode honest (no zero-byte
    sentinel a CI step could mistake for "ran successfully")."""
    out = tmp_path / "report.txt"
    proc = _run_cli(str(tmp_path / "no.jsonl"), "--out", str(out))
    assert proc.returncode == 2
    assert not out.exists(), "exit-2 must not create the --out file"
