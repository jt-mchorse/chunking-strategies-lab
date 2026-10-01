"""`validate --corpus-dir` refuses a corpus with no documents, as `load_corpus` does (#214).

Measured on `d11fcb3`: `--corpus-dir data/corpus/01_hnsw.md` (a file) and an
empty directory each produced 12 `expected_doc_not_found` findings at exit 1 --
the first telling the operator `01_hnsw.md` is not a corpus document under
`.../01_hnsw.md` -- while `load_corpus` refuses both paths outright.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chunking_lab.corpus import load_corpus
from chunking_lab.validate import main, validate_queries

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data" / "queries.jsonl"
CORPUS = ROOT / "data" / "corpus"


def _corpus_dir(kind: str, tmp_path: Path) -> Path:
    if kind == "a-file":
        return CORPUS / "01_hnsw.md"
    d = tmp_path / kind
    d.mkdir()
    if kind == "no-md-files":
        (d / "notes.txt").write_text("hi", encoding="utf-8")
        (d / "bundle.md").mkdir()  # a directory named *.md is not a document
    return d


KINDS = ["a-file", "empty-dir", "no-md-files"]


@pytest.mark.parametrize("kind", KINDS)
def test_the_library_refuses_with_the_loaders_words(kind: str, tmp_path: Path) -> None:
    corpus_dir = _corpus_dir(kind, tmp_path)
    with pytest.raises(FileNotFoundError) as from_validate:
        validate_queries(QUERIES, corpus_dir=corpus_dir)
    with pytest.raises(FileNotFoundError) as from_loader:
        load_corpus(corpus_dir)
    assert str(from_validate.value) == str(from_loader.value)


@pytest.mark.parametrize("kind", KINDS)
def test_the_cli_exits_two_and_reports_no_findings(
    kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([str(QUERIES), "--corpus-dir", str(_corpus_dir(kind, tmp_path))])
    captured = capsys.readouterr()
    assert rc == 2
    assert "no markdown documents in:" in captured.err
    assert "expected_doc_not_found" not in captured.out + captured.err


def test_a_real_corpus_still_validates_clean(tmp_path: Path) -> None:
    copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, copy)
    assert main([str(QUERIES), "--corpus-dir", str(copy)]) == 0
