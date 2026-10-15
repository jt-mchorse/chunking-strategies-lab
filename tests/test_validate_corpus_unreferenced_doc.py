"""`validate --corpus-dir` reads every corpus document, not only the ones a row
names (#268).

#258 made a non-UTF-8 or unreadable corpus document exit 2 naming the document,
but the read happened on a row's first reference to it. A document that no
valid row names was never opened: a latin-1 byte in `b.md` beside a clean
`a.md` validated `ok` at exit 0, and `load_corpus` then raised on the same
directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from chunking_lab.corpus import load_corpus
from chunking_lab.validate import main

# 0xE9 is latin-1 `é`: a UTF-8 lead byte with no continuation byte after it.
_LATIN1_BODY = b"caf" + bytes([0xE9]) + b"\n"


def _row(qid: str, doc: str, snippet: str) -> str:
    return json.dumps(
        {"id": qid, "question": "?", "expected_doc": doc, "expected_snippet": snippet}
    )


def _setup(tmp_path: Path, rows: list[str]) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha beta gamma\n", encoding="utf-8")
    other = corpus / "b.md"
    other.write_text("delta\n", encoding="utf-8")
    queries = tmp_path / "q.jsonl"
    queries.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return queries, corpus, other


def _assert_refused_naming(
    capsys: pytest.CaptureFixture[str], rc: int, queries: Path, doc: Path
) -> str:
    err = capsys.readouterr().err
    assert rc == 2, err
    assert f"failed to read corpus document {doc}" in err, err
    assert f"failed to read {queries}" not in err, err
    return err


def test_a_non_utf8_document_no_row_names_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries, corpus, other = _setup(tmp_path, [_row("q1", "a.md", "beta")])
    other.write_bytes(_LATIN1_BODY)
    # The library refuses this corpus; the validator must not call it ok.
    with pytest.raises(UnicodeDecodeError):
        load_corpus(corpus)
    rc = main([str(queries), "--corpus-dir", str(corpus)])
    err = _assert_refused_naming(capsys, rc, queries, other)
    assert "not valid UTF-8" in err


def test_a_document_named_only_by_a_failing_row_is_still_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second row names `b.md` but is a duplicate id, so it never reached
    the per-row read either."""
    queries, corpus, other = _setup(
        tmp_path, [_row("q1", "a.md", "beta"), _row("q1", "b.md", "delta")]
    )
    other.write_bytes(_LATIN1_BODY)
    rc = main([str(queries), "--corpus-dir", str(corpus)])
    _assert_refused_naming(capsys, rc, queries, other)


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not deny reads on Windows or to root",
)
def test_an_unreadable_document_no_row_names_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries, corpus, other = _setup(tmp_path, [_row("q1", "a.md", "beta")])
    other.chmod(0)
    try:
        rc = main([str(queries), "--corpus-dir", str(corpus)])
    finally:
        other.chmod(0o644)
    _assert_refused_naming(capsys, rc, queries, other)


def test_a_loadable_corpus_with_an_unreferenced_document_still_validates_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The neighbour the fix must leave alone: an unreferenced document is not
    itself a finding, and a BOM-prefixed one loads (#95)."""
    queries, corpus, other = _setup(tmp_path, [_row("q1", "a.md", "beta")])
    other.write_text("naïve\n", encoding="utf-8-sig")
    load_corpus(corpus)
    rc = main([str(queries), "--corpus-dir", str(corpus)])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "findings=0" in out.out
