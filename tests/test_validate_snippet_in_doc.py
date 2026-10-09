"""`validate --corpus-dir` checks that `expected_snippet` occurs in `expected_doc` (#255).

It checked `expected_doc` only. snippet-hit@k is a substring test on chunk text
(`metrics.py`: `q.expected_snippet in c.text`), and a chunk is a slice of its
document, so a snippet the document lacks can never hit from it. Measured on
main with q02's snippet typo'd to `M=16, ef_constuction=64`: validate printed
`ok ... findings=0` at exit 0, and `evaluate_strategy` gave q02 0 snippet hits
in its top 1000 (snippet_hit@1000 1.0 -> 0.9167).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_lab.corpus import load_corpus
from chunking_lab.queries import load_queries
from chunking_lab.validate import validate_queries

REPO = Path(__file__).resolve().parent.parent


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "hnsw.md").write_text("# HNSW\nUse M=16, ef_construction=64.\n", encoding="utf-8")
    (corpus / "rrf.md").write_text("# RRF\nThe constant k is 60.\n", encoding="utf-8")
    # A BOM that `load_corpus` strips: a snippet at the very start still matches.
    (corpus / "bom.md").write_bytes(b"\xef\xbb\xbfLeading words matter.\n")
    return corpus


def _queries(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "q.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"q{i}", "question": "?", "expected_doc": d, "expected_snippet": s})
            + "\n"
            for i, (d, s) in enumerate(rows)
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("doc", "snippet"),
    [
        ("hnsw.md", "M=16, ef_constuction=64"),  # the issue's typo
        ("hnsw.md", "The constant k is 60."),  # in another document, not this one
        ("hnsw.md", "m=16"),  # the match is exact, as snippet-hit's is
    ],
)
def test_a_snippet_the_document_lacks_is_a_finding(tmp_path: Path, doc: str, snippet: str) -> None:
    report = validate_queries(_queries(tmp_path, [(doc, snippet)]), corpus_dir=_corpus(tmp_path))
    assert [f.code for f in report.findings] == ["expected_snippet_not_in_doc"]
    assert report.n_valid == 0
    assert not report.ok


@pytest.mark.parametrize(
    ("doc", "snippet"),
    [("hnsw.md", "M=16, ef_construction=64"), ("rrf.md", "k is 60"), ("bom.md", "Leading words")],
)
def test_a_snippet_in_its_document_passes(tmp_path: Path, doc: str, snippet: str) -> None:
    report = validate_queries(_queries(tmp_path, [(doc, snippet)]), corpus_dir=_corpus(tmp_path))
    assert report.ok
    assert report.n_valid == 1


def test_the_check_reads_documents_as_load_corpus_does(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    texts = {d.filename: d.text for d in load_corpus(corpus)}
    for doc, text in texts.items():
        snippet = text[:7]
        report = validate_queries(_queries(tmp_path, [(doc, snippet)]), corpus_dir=corpus)
        assert report.ok, (doc, snippet, report.findings)


def test_without_corpus_dir_no_document_is_read(tmp_path: Path) -> None:
    report = validate_queries(_queries(tmp_path, [("hnsw.md", "not there")]))
    assert report.ok


def test_every_pinned_query_passes_and_its_snippet_is_in_its_doc() -> None:
    queries = REPO / "data" / "queries.jsonl"
    corpus = REPO / "data" / "corpus"
    assert validate_queries(queries, corpus_dir=corpus).ok
    texts = {d.filename: d.text for d in load_corpus(corpus)}
    assert all(q.expected_snippet in texts[q.expected_doc] for q in load_queries(queries))


def test_the_cli_exits_1_and_names_the_row(tmp_path: Path) -> None:
    path = _queries(tmp_path, [("hnsw.md", "M=16, ef_constuction=64")])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "chunking_lab.validate",
            str(path),
            "--corpus-dir",
            str(_corpus(tmp_path)),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 1
    assert "line 1 [expected_snippet_not_in_doc]" in proc.stdout + proc.stderr


def test_an_undecodable_document_is_exit_2_not_a_traceback(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    (corpus / "hnsw.md").write_bytes(b"\xff\xfe not utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "chunking_lab.validate",
            str(_queries(tmp_path, [("hnsw.md", "x")])),
            "--corpus-dir",
            str(corpus),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
