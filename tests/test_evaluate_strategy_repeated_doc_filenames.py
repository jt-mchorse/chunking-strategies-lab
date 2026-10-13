"""`evaluate_strategy` refuses a corpus where two documents share a filename (#262).

`Document.filename` is the attribution key: every chunk's `source_doc_id` is
set from it, and recall@k is `expected_doc in retrieved_doc_ids[:k]`. #250
moved the query-id uniqueness rule to the metric boundary; the corpus had the
same hole, and the #192 paragraph that declines a corpus guard gives a reason
about emptiness only.

Measured on the unguarded code, pinned substrate, `FixedSizeStrategy(600, 80)`,
`HashEmbedder`: the RRF document's text added as a sixth document scored
recall@5 0.750 named `06_other.md` and 0.833 named `01_hnsw.md` -- same chunks,
same scores, HNSW queries credited for RRF text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.corpus import Document, load_corpus  # noqa: E402
from chunking_lab.embedder import HashEmbedder  # noqa: E402
from chunking_lab.metrics import evaluate_strategy, validate_corpus  # noqa: E402
from chunking_lab.queries import Query, load_queries  # noqa: E402
from chunking_lab.strategies import Chunk, FixedSizeStrategy  # noqa: E402

_ALPHA = Document(filename="alpha.md", text="alpha beta gamma delta epsilon " * 40)
_ZETA = Document(filename="zeta.md", text="zeta eta theta iota kappa " * 40)
_QUERIES = [
    Query(id="q1", question="alpha?", expected_doc="alpha.md", expected_snippet="alpha beta"),
]


def test_two_documents_sharing_a_filename_are_refused() -> None:
    impostor = Document(filename="alpha.md", text=_ZETA.text)
    with pytest.raises(ValueError, match="corpus document filenames must be unique"):
        evaluate_strategy(FixedSizeStrategy(), [_ALPHA, _ZETA, impostor], _QUERIES, HashEmbedder())


def test_the_same_document_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="repeated: \\['alpha.md'\\]"):
        evaluate_strategy(FixedSizeStrategy(), [_ALPHA, _ZETA, _ALPHA], _QUERIES, HashEmbedder())


def test_the_message_names_every_repeated_filename_once() -> None:
    with pytest.raises(ValueError, match="filenames must be unique") as exc:
        validate_corpus([_ALPHA, _ZETA, _ALPHA, _ZETA, _ALPHA])
    assert "['alpha.md', 'zeta.md']" in str(exc.value)


def test_the_guard_fires_before_any_chunking() -> None:
    calls: list[str] = []

    class _Watching(FixedSizeStrategy):
        def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
            calls.append(source_doc_id)
            return super().chunk(text, source_doc_id=source_doc_id)

    with pytest.raises(ValueError, match="filenames must be unique"):
        evaluate_strategy(_Watching(), [_ALPHA, _ALPHA], _QUERIES, HashEmbedder())
    assert calls == []


def test_pinned_substrate_repro_is_refused() -> None:
    """The issue's measurement: a second `01_hnsw.md` carrying RRF text."""
    corpus = load_corpus()
    rrf = next(d for d in corpus if d.filename == "02_rrf.md")
    with pytest.raises(ValueError, match="repeated: \\['01_hnsw.md'\\]"):
        evaluate_strategy(
            FixedSizeStrategy(chunk_chars=600, overlap_chars=80),
            [*corpus, Document(filename="01_hnsw.md", text=rrf.text)],
            load_queries(),
            HashEmbedder(),
        )


# --- Anti-vacuity: the neighbours the guard must NOT refuse -------------------


def test_identical_text_under_distinct_filenames_still_evaluates() -> None:
    """The rule is on the key, not on the content: rejects the over-broad
    neighbour that dedupes documents by text."""
    twin = Document(filename="alpha_copy.md", text=_ALPHA.text)
    run = evaluate_strategy(FixedSizeStrategy(), [_ALPHA, twin], _QUERIES, HashEmbedder())
    assert run.n_queries == 1


def test_an_empty_corpus_still_evaluates_to_a_truthful_zero() -> None:
    """#192's decision stands: emptiness is not what this guard is about."""
    run = evaluate_strategy(FixedSizeStrategy(), [], _QUERIES, HashEmbedder(), ks=(1,))
    assert run.n_chunks_total == 0
    assert run.recall_at_k == {1: 0.0}


def test_the_pinned_corpus_passes() -> None:
    validate_corpus(load_corpus())


def test_a_non_str_filename_is_still_check_chunk_inputs_to_refuse() -> None:
    """An unhashable filename must not escape the uniqueness check as a raw
    TypeError; the existing #176 message still wins."""
    bad = Document(filename=["alpha.md"], text="alpha")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_doc_id must be a str"):
        evaluate_strategy(FixedSizeStrategy(), [bad, bad], _QUERIES, HashEmbedder())
