"""Tests for the five chunking strategies (issue #2).

Per-strategy hermetic unit tests covering:
- Boundary preservation: chunks reconstruct (or recoverably-reconstruct) source.
- Size invariants: no chunk exceeds the configured cap.
- Strategy-specific behavior:
  * fixed: stride math.
  * recursive: actually recurses on too-long pieces.
  * semantic: splits at high-distance points.
  * late: returns (chunk, vector) pairs and the vector blends doc + chunk.
  * structure: respects markdown headings.
- Plus a runtime benchmark across the committed 5-doc corpus.
"""

from __future__ import annotations

import time

import pytest

from chunking_lab import (
    Chunk,
    FixedSizeStrategy,
    HashEmbedder,
    LateChunkingStrategy,
    RecursiveStrategy,
    SemanticBoundaryStrategy,
    Strategy,
    StructureAwareStrategy,
    load_corpus,
)

# ----------------------------------------------------------------------
# Common
# ----------------------------------------------------------------------

DOC_TEXT = (
    "Postgres tuning starts with shared_buffers and work_mem. "
    "Once the foundational settings are right, autovacuum becomes the "
    "next frontier.\n\n"
    "Vector search benchmarks compare engines on three axes: recall, "
    "latency, and cost per query. The interesting cases are not at 1M "
    "vectors but at 100M, where the index spills off-RAM.\n\n"
    "Anthropic prompt caching cuts the price of repeated context by 90% "
    "on read and adds a 25% surcharge on first write. The math favors "
    "long stable system prompts."
)


def _all_strategies() -> list[Strategy]:
    e = HashEmbedder()
    return [
        FixedSizeStrategy(chunk_chars=200, overlap_chars=50),
        RecursiveStrategy(chunk_chars=200),
        SemanticBoundaryStrategy(embedder=e, distance_threshold=0.4, min_chunk_chars=20),
        LateChunkingStrategy(embedder=e, chunk_chars=200, overlap_chars=50),
        StructureAwareStrategy(),
    ]


# ----------------------------------------------------------------------
# Boundary preservation contract (every strategy)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("strategy", _all_strategies(), ids=lambda s: s.name)
def test_chunks_have_valid_offsets(strategy):
    chunks = strategy.chunk(DOC_TEXT, source_doc_id="doc-1")
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source_doc_id == "doc-1"
        assert c.strategy_name == strategy.name
        # Offsets must be within bounds and consistent with text length.
        assert 0 <= c.start_offset <= len(DOC_TEXT)
        assert c.start_offset <= c.end_offset <= len(DOC_TEXT)
        # The text at those offsets matches the chunk's text (sliding-window
        # strategies guarantee this; structure/semantic strategies preserve
        # text from the source so it should hold there too).
        assert DOC_TEXT[c.start_offset : c.end_offset] == c.text


@pytest.mark.parametrize("strategy", _all_strategies(), ids=lambda s: s.name)
def test_empty_input_returns_empty(strategy):
    assert strategy.chunk("", source_doc_id="d") == []


# ----------------------------------------------------------------------
# FixedSizeStrategy
# ----------------------------------------------------------------------


def test_fixed_respects_chunk_size_cap():
    s = FixedSizeStrategy(chunk_chars=50, overlap_chars=0)
    chunks = s.chunk(DOC_TEXT)
    for c in chunks:
        assert len(c.text) <= 50


def test_fixed_overlap_creates_overlap():
    s = FixedSizeStrategy(chunk_chars=50, overlap_chars=10)
    chunks = s.chunk(DOC_TEXT)
    # Adjacent chunks should overlap by `overlap_chars` (except possibly the last).
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start_offset == a.end_offset - 10


def test_fixed_rejects_overlap_geq_chunk_size():
    with pytest.raises(ValueError, match="overlap_chars"):
        FixedSizeStrategy(chunk_chars=50, overlap_chars=50)


def test_fixed_rejects_zero_chunk_chars():
    with pytest.raises(ValueError, match="chunk_chars"):
        FixedSizeStrategy(chunk_chars=0)


# ----------------------------------------------------------------------
# RecursiveStrategy
# ----------------------------------------------------------------------


def test_recursive_respects_chunk_size_cap_for_normal_text():
    s = RecursiveStrategy(chunk_chars=120)
    chunks = s.chunk(DOC_TEXT)
    for c in chunks:
        # Recursive *tries* to fit; the brute-force fallback at "" guarantees
        # the cap is respected for sane inputs (no separator-less words longer
        # than chunk_chars).
        assert len(c.text) <= 120


def test_recursive_chunks_reconstruct_source():
    s = RecursiveStrategy(chunk_chars=120)
    chunks = s.chunk(DOC_TEXT)
    # Concat in offset order should equal source (recursive doesn't overlap).
    chunks_by_start = sorted(chunks, key=lambda c: c.start_offset)
    reconstructed = "".join(c.text for c in chunks_by_start)
    assert reconstructed == DOC_TEXT


def test_recursive_rejects_empty_separators():
    with pytest.raises(ValueError, match="separators"):
        RecursiveStrategy(separators=())


# ----------------------------------------------------------------------
# SemanticBoundaryStrategy
# ----------------------------------------------------------------------


def test_semantic_emits_at_least_one_chunk_for_multi_topic_text():
    e = HashEmbedder()
    # Three unrelated sentences — should split somewhere.
    text = (
        "Postgres autovacuum tuning is essential. "
        "The weather in Reykjavik is famously volatile. "
        "Quantum entanglement remains poorly understood."
    )
    s = SemanticBoundaryStrategy(embedder=e, distance_threshold=0.5, min_chunk_chars=0)
    chunks = s.chunk(text)
    # At least one chunk; at most three (one per sentence).
    assert 1 <= len(chunks) <= 3


def test_semantic_respects_distance_threshold_validation():
    with pytest.raises(ValueError, match="distance_threshold"):
        SemanticBoundaryStrategy(embedder=HashEmbedder(), distance_threshold=2.5)


def test_semantic_handles_single_sentence():
    e = HashEmbedder()
    s = SemanticBoundaryStrategy(embedder=e, distance_threshold=0.4, min_chunk_chars=0)
    chunks = s.chunk("Just one sentence with no terminator")
    assert len(chunks) == 1


# ----------------------------------------------------------------------
# LateChunkingStrategy
# ----------------------------------------------------------------------


def test_late_chunking_returns_chunks_with_vectors():
    e = HashEmbedder()
    s = LateChunkingStrategy(embedder=e, chunk_chars=200, overlap_chars=50)
    late_chunks = s.chunk_with_vectors(DOC_TEXT, source_doc_id="d")
    assert len(late_chunks) > 1
    for lc in late_chunks:
        assert isinstance(lc.chunk, Chunk)
        assert isinstance(lc.vector, tuple)
        assert len(lc.vector) > 0


def test_late_chunking_vector_blends_doc_and_chunk():
    """document_weight=1.0 → every chunk gets the document vector."""
    e = HashEmbedder()
    s = LateChunkingStrategy(embedder=e, chunk_chars=100, overlap_chars=0, document_weight=1.0)
    late_chunks = s.chunk_with_vectors(DOC_TEXT)
    doc_vec = list(late_chunks[0].vector)
    for lc in late_chunks[1:]:
        # All chunk vectors equal the doc vector when weight=1.0
        assert list(lc.vector) == pytest.approx(doc_vec)


def test_late_chunking_chunk_method_returns_only_chunks():
    e = HashEmbedder()
    s = LateChunkingStrategy(embedder=e, chunk_chars=200, overlap_chars=50)
    chunks = s.chunk(DOC_TEXT)
    for c in chunks:
        assert isinstance(c, Chunk)


def test_late_chunking_rejects_invalid_doc_weight():
    with pytest.raises(ValueError, match="document_weight"):
        LateChunkingStrategy(embedder=HashEmbedder(), document_weight=1.5)


# ----------------------------------------------------------------------
# StructureAwareStrategy
# ----------------------------------------------------------------------


MD_DOC = """
# Top heading
Paragraph under the top.

## Subsection one
Text under subsection one.

## Subsection two
Text under subsection two.

# Second top heading
More text.
"""


def test_structure_splits_on_top_level_headings_only_when_capped():
    s = StructureAwareStrategy(max_heading_level=1)
    chunks = s.chunk(MD_DOC)
    # max_heading_level=1 → only `#` lines split → 2 sections (+ possible preamble).
    section_chunks = [c for c in chunks if c.metadata.get("heading_level") == 1]
    assert len(section_chunks) == 2


def test_structure_splits_on_all_headings():
    s = StructureAwareStrategy(max_heading_level=6)
    chunks = s.chunk(MD_DOC)
    titles = [c.metadata["title"] for c in chunks if c.metadata.get("heading_level") is not None]
    assert "Top heading" in titles
    assert "Subsection one" in titles
    assert "Subsection two" in titles
    assert "Second top heading" in titles


def test_structure_falls_back_for_unheaded_text():
    s = StructureAwareStrategy()
    chunks = s.chunk("just some plain text\nwith no headings")
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_level"] is None


def test_structure_rejects_invalid_heading_level():
    with pytest.raises(ValueError, match="max_heading_level"):
        StructureAwareStrategy(max_heading_level=0)


# ----------------------------------------------------------------------
# Run-time benchmark across the committed corpus (acceptance criterion 3)
# ----------------------------------------------------------------------


def test_runtime_benchmark_across_committed_corpus():
    """Acceptance criterion: run-time per strategy recorded.

    Sanity-check the strategies don't blow up time-wise on the 5-doc corpus.
    Real per-strategy numbers go in `docs/strategy_runtimes.md` once the
    operator runs the dedicated benchmark; this test just enforces a
    generous CI budget.
    """
    docs = load_corpus()
    assert len(docs) >= 5
    runtimes_s: dict[str, float] = {}
    for strategy in _all_strategies():
        t0 = time.perf_counter()
        for doc in docs:
            strategy.chunk(doc.text, source_doc_id=doc.filename)
        runtimes_s[strategy.name] = time.perf_counter() - t0
    for name, dt in runtimes_s.items():
        assert dt < 5.0, f"{name} took {dt:.3f}s on the 5-doc corpus (>5s budget)"


# Issue #29: strategy dataclasses validate `chunk_chars` / `overlap_chars` /
# `min_chunk_chars` / `max_chunk_chars` as `isinstance(int)` so non-int (float,
# NaN, fractional, bool) is rejected at construction rather than failing deep
# inside the chunking loop with TypeError or producing a spinning loop. Same
# shape as embedding-model-shootout #32 SweepResult count fields.
_BAD_INT = [1.5, float("nan"), float("inf"), True, "5"]


@pytest.mark.parametrize("bad", _BAD_INT)
def test_fixed_chunk_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="chunk_chars must be an int"):
        FixedSizeStrategy(chunk_chars=bad, overlap_chars=10)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_fixed_overlap_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="overlap_chars must be an int"):
        FixedSizeStrategy(chunk_chars=100, overlap_chars=bad)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_late_chunk_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="chunk_chars must be an int"):
        LateChunkingStrategy(embedder=HashEmbedder(), chunk_chars=bad, overlap_chars=10)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_recursive_chunk_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="chunk_chars must be an int"):
        RecursiveStrategy(chunk_chars=bad)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_semantic_min_chunk_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="min_chunk_chars must be an int"):
        SemanticBoundaryStrategy(embedder=HashEmbedder(), min_chunk_chars=bad, max_chunk_chars=1000)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_semantic_max_chunk_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="max_chunk_chars must be an int"):
        SemanticBoundaryStrategy(embedder=HashEmbedder(), min_chunk_chars=80, max_chunk_chars=bad)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_hash_embedder_dim_must_be_int(bad):
    with pytest.raises(ValueError, match="dim must be an int"):
        HashEmbedder(dim=bad)


def test_acceptance_regression_strategies_construct_with_valid_ints():
    # Boundary acceptance: every prior valid-int construction continues to work.
    assert FixedSizeStrategy(chunk_chars=100, overlap_chars=10).chunk_chars == 100
    assert (
        LateChunkingStrategy(
            embedder=HashEmbedder(), chunk_chars=100, overlap_chars=10
        ).overlap_chars
        == 10
    )
    assert RecursiveStrategy(chunk_chars=100).chunk_chars == 100
    assert (
        SemanticBoundaryStrategy(
            embedder=HashEmbedder(), min_chunk_chars=80, max_chunk_chars=1000
        ).max_chunk_chars
        == 1000
    )
    assert HashEmbedder(dim=64).dim == 64
