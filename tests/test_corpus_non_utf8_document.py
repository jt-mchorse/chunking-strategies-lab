"""`load_corpus` names the corpus document that is not valid UTF-8 (#264).

#258 fixed this for `validate --corpus-dir`, which reads the same files. A
`UnicodeDecodeError` carries no filename, so `load_corpus` on a three-document
directory with one latin-1 byte raised `'utf-8' codec can't decode byte 0xe9
in position 3: invalid continuation byte` and nothing said which document it
came from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chunking_lab.corpus import load_corpus

# 0xE9 is latin-1 `é`: a UTF-8 lead byte with no continuation byte after it.
_LATIN1_BODY = b"caf" + bytes([0xE9]) + b"\n"


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("fine\n", encoding="utf-8")
    bad = corpus / "b.md"
    bad.write_bytes(_LATIN1_BODY)
    (corpus / "c.md").write_text("ok\n", encoding="utf-8")
    return corpus, bad


def test_the_non_utf8_document_is_named(tmp_path: Path) -> None:
    corpus, bad = _corpus(tmp_path)
    with pytest.raises(UnicodeDecodeError) as excinfo:
        load_corpus(corpus)
    message = str(excinfo.value)
    assert f"corpus document {bad}" in message, message
    # The other two documents are fine and must not be blamed.
    assert "a.md" not in message, message
    assert "c.md" not in message, message


def test_the_error_keeps_its_type_and_codec_detail(tmp_path: Path) -> None:
    """Same type and offsets as the codec's own error, so callers that catch
    `UnicodeDecodeError` or `ValueError` around `load_corpus` still do."""
    corpus, _ = _corpus(tmp_path)
    with pytest.raises(UnicodeDecodeError) as excinfo:
        load_corpus(corpus)
    e = excinfo.value
    assert isinstance(e, ValueError)
    assert e.start == 3
    assert e.end == 4
    assert e.object[e.start] == 0xE9
    assert "invalid continuation byte" in str(e)
    assert isinstance(e.__cause__, UnicodeDecodeError)


def test_a_valid_corpus_loads_unchanged(tmp_path: Path) -> None:
    """The neighbour the fix must leave alone, BOM case included (#95)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("café\n", encoding="utf-8")
    (corpus / "b.md").write_text("naïve\n", encoding="utf-8-sig")
    docs = load_corpus(corpus)
    assert [(d.filename, d.text) for d in docs] == [("a.md", "café\n"), ("b.md", "naïve\n")]
