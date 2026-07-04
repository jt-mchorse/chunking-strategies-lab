"""Substrate tests: corpus loads, queries parse, both align."""

from __future__ import annotations

import pytest

from chunking_lab import (
    CANONICAL_EMBEDDING_MODEL,
    Document,
    Embedder,
    HashEmbedder,
    Query,
    load_corpus,
    load_queries,
)

# --- Corpus ---------------------------------------------------------------


def test_corpus_loads_from_default_path():
    docs = load_corpus()
    assert len(docs) == 5
    for d in docs:
        assert isinstance(d, Document)
        assert d.filename.endswith(".md")
        assert d.char_count > 500  # each document is meaningful-length


def test_corpus_filenames_are_unique():
    docs = load_corpus()
    names = [d.filename for d in docs]
    assert len(names) == len(set(names))


def test_corpus_documents_are_sorted_for_determinism():
    docs = load_corpus()
    names = [d.filename for d in docs]
    assert names == sorted(names)


def test_corpus_loader_rejects_missing_directory(tmp_path):
    missing = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        load_corpus(missing)


def test_corpus_loader_rejects_empty_directory(tmp_path):
    (tmp_path / "data" / "corpus").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no markdown"):
        load_corpus(tmp_path / "data" / "corpus")


def test_corpus_loader_handles_utf8_bom(tmp_path):
    # Issue #95: a UTF-8 BOM (EF BB BF — Windows Notepad / some doc exporters)
    # is not whitespace, so reading with plain "utf-8" leaves U+FEFF as an
    # invisible leading char in the document text, shifting first-chunk offsets
    # and leaking into snippet matches. utf-8-sig strips it transparently. A
    # BOM-prefixed document must load identically to its BOM-less twin. Parity
    # with the queries loader (#93).
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    body = "# Heading\n\nFirst paragraph of technical prose about chunking.\n"
    bomful = corpus / "bom.md"
    bomless = corpus / "plain.md"
    bomful.write_text(body, encoding="utf-8-sig")  # prepends the BOM
    bomless.write_text(body, encoding="utf-8")
    # Sanity: the on-disk bytes actually differ by the BOM.
    assert bomful.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not bomless.read_bytes().startswith(b"\xef\xbb\xbf")

    docs = {d.filename: d.text for d in load_corpus(corpus)}
    assert docs["bom.md"] == body  # no leading U+FEFF
    assert not docs["bom.md"].startswith("﻿")
    assert docs["bom.md"] == docs["plain.md"]


# --- Queries --------------------------------------------------------------


def test_queries_load_from_default_path():
    qs = load_queries()
    assert len(qs) >= 10  # session plan committed to ≥10
    for q in qs:
        assert isinstance(q, Query)
        assert q.id
        assert q.question
        assert q.expected_doc.endswith(".md")
        assert q.expected_snippet


def test_query_ids_are_unique():
    qs = load_queries()
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids))


def test_each_query_targets_a_document_in_the_corpus():
    docs = {d.filename for d in load_corpus()}
    for q in load_queries():
        assert q.expected_doc in docs, (
            f"query {q.id} targets {q.expected_doc!r} which is not in the corpus"
        )


def test_each_expected_snippet_is_verbatim_in_its_document():
    docs = {d.filename: d.text for d in load_corpus()}
    for q in load_queries():
        assert q.expected_snippet in docs[q.expected_doc], (
            f"query {q.id} snippet {q.expected_snippet!r} not verbatim in {q.expected_doc}"
        )


def test_queries_loader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_queries(tmp_path / "nope.jsonl")


def test_queries_loader_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_queries(p)


def test_queries_loader_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text(
        '{"id":"q1","question":"a","expected_doc":"x.md","expected_snippet":"y"}\n'
        '{"id":"q1","question":"b","expected_doc":"x.md","expected_snippet":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_queries(p)


def test_queries_loader_rejects_invalid_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_queries(p)


def test_queries_loader_rejects_missing_required_field(tmp_path):
    p = tmp_path / "missing.jsonl"
    p.write_text('{"id":"q1","question":"q"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected_doc"):
        load_queries(p)


def test_queries_loader_handles_utf8_bom(tmp_path):
    # Issue #93: a UTF-8 BOM (EF BB BF — Windows Notepad / some spreadsheet
    # exports) survives `.strip()` (U+FEFF is not whitespace) and reaches
    # json.loads on line 1. utf-8-sig strips it transparently. A BOM file must
    # load identically to its BOM-less twin.
    body = (
        '{"id":"q1","question":"a","expected_doc":"x.md","expected_snippet":"y"}\n'
        '{"id":"q2","question":"b","expected_doc":"z.md","expected_snippet":"w"}\n'
    )
    bomful = tmp_path / "bom.jsonl"
    bomless = tmp_path / "plain.jsonl"
    bomful.write_text(body, encoding="utf-8-sig")  # prepends the BOM
    bomless.write_text(body, encoding="utf-8")
    # Sanity: the on-disk bytes actually differ by the BOM.
    assert bomful.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not bomless.read_bytes().startswith(b"\xef\xbb\xbf")

    from_bom = load_queries(bomful)
    from_plain = load_queries(bomless)
    assert [q.id for q in from_bom] == ["q1", "q2"]
    assert from_bom == from_plain


def test_queries_loader_rejects_whitespace_only_field_with_lineno(tmp_path):
    # #92: `_require_str` must reject a blank field on the load path too, with
    # file:lineno context. Pre-fix the `if not value` guard let "   " through.
    p = tmp_path / "blank.jsonl"
    p.write_text(
        '{"id":"q1","question":"a","expected_doc":"x.md","expected_snippet":"y"}\n'
        '{"id":"q2","question":"b","expected_doc":"x.md","expected_snippet":"   "}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="expected_snippet.*must be non-empty and not whitespace-only"
    ):
        load_queries(p)


# --- Query dataclass invariant (direct construction) ----------------------


def _valid_query_kwargs() -> dict[str, str]:
    return {
        "id": "q1",
        "question": "What is in bananas?",
        "expected_doc": "bananas.md",
        "expected_snippet": "potassium",
    }


def test_query_constructs_with_valid_fields():
    q = Query(**_valid_query_kwargs())
    assert q.id == "q1"
    assert q.expected_snippet == "potassium"


@pytest.mark.parametrize("field", ["id", "question", "expected_doc", "expected_snippet"])
def test_query_rejects_empty_field_on_direct_construction(field):
    # `Query` is public and built directly (not only via `load_queries`), so the
    # dataclass itself must enforce non-empty fields. An empty `expected_snippet`
    # is the worst case: `"" in chunk.text` is always True, so snippet-hit@k
    # reads a trivial 1.0 for every strategy and silently corrupts the
    # comparison (#72). Empty `expected_doc` is the mirror (recall trivially 0).
    kwargs = _valid_query_kwargs()
    kwargs[field] = ""
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        Query(**kwargs)


@pytest.mark.parametrize("field", ["id", "question", "expected_doc", "expected_snippet"])
def test_query_rejects_non_string_field_on_direct_construction(field):
    kwargs: dict[str, object] = dict(_valid_query_kwargs())
    kwargs[field] = 123
    with pytest.raises(ValueError, match=f"{field} must be a string"):
        Query(**kwargs)  # type: ignore[arg-type]


def test_empty_snippet_query_cannot_reach_snippet_hit_trivial_pass():
    # Regression for the exact bypass #72 closes: before the __post_init__ guard,
    # Query(expected_snippet="") was constructible and made snippet-hit a trivial
    # 1.0. The guard now stops it at construction, so the corrupt query can never
    # reach `evaluate_strategy`.
    with pytest.raises(ValueError, match="expected_snippet must be non-empty"):
        Query(id="bad", question="q", expected_doc="d.md", expected_snippet="")


@pytest.mark.parametrize("field", ["id", "question", "expected_doc", "expected_snippet"])
@pytest.mark.parametrize("blank", ["   ", "\t", "\n", " \t\n "])
def test_query_rejects_whitespace_only_field_on_direct_construction(field, blank):
    # #92 completes #72: a whitespace-only field is as corrupting as an empty one
    # (`"   " in chunk.text` is True for any chunk with three consecutive spaces,
    # so snippet-hit@k still reads a trivial 1.0). The old `if not value` guard
    # missed these because `"   "` is truthy. Inverse safety net: pre-fix these
    # were constructible.
    kwargs = _valid_query_kwargs()
    kwargs[field] = blank
    with pytest.raises(ValueError, match=f"{field} must be non-empty and not whitespace-only"):
        Query(**kwargs)


def test_query_accepts_field_with_internal_whitespace():
    # Over-rejection guard: only *blank* fields are rejected; a field with
    # internal/surrounding-plus-content whitespace is legitimate content.
    q = Query(id="q 1", question="what is x?", expected_doc="a b.md", expected_snippet="a b c")
    assert q.id == "q 1"
    assert q.expected_snippet == "a b c"


def test_whitespace_only_snippet_query_cannot_reach_snippet_hit_trivial_pass():
    # The whitespace twin of test_empty_snippet_query_cannot_reach_snippet_hit_
    # trivial_pass: a "   " snippet must be stopped at construction so it can
    # never inflate snippet-hit@k inside `evaluate_strategy`.
    with pytest.raises(
        ValueError, match="expected_snippet must be non-empty and not whitespace-only"
    ):
        Query(id="bad", question="q", expected_doc="d.md", expected_snippet="   ")


# --- Embedder -------------------------------------------------------------


def test_hash_embedder_default_dim_matches_canonical_model_dim():
    """The dep-free reference must produce vectors that are
    drop-in-compatible with the canonical model's dimensionality.
    Otherwise strategies tested under HashEmbedder won't transfer.
    """
    e = HashEmbedder()
    assert len(e.embed("any")) == 384


def test_hash_embedder_deterministic_and_unit_length():
    e = HashEmbedder()
    v1 = e.embed("a")
    v2 = e.embed("a")
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_hash_embedder_satisfies_protocol():
    e: Embedder = HashEmbedder()
    assert isinstance(e, Embedder)


def test_canonical_model_is_pinned():
    """D-002 contract: the canonical model name doesn't drift silently."""
    assert CANONICAL_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
