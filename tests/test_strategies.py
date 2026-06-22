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


# Multi-sentence offset<->text contract (#50). `test_chunks_have_valid_offsets`
# only ever produces single-sentence semantic chunks (every adjacent pair trips
# threshold=0.4 under HashEmbedder), so the concatenation path went unchecked.
_SEMANTIC_SPAN_TEXT = (
    "Cats are small animals. Dogs are loyal companions. "
    "The sky is blue today. Rain falls in spring."
)


def test_semantic_multi_sentence_block_preserves_offsets():
    # threshold=2.0 → no boundaries → all four sentences land in ONE chunk,
    # exercising the multi-sentence concatenation path.
    s = SemanticBoundaryStrategy(
        embedder=HashEmbedder(), distance_threshold=2.0, min_chunk_chars=0, max_chunk_chars=10_000
    )
    chunks = s.chunk(_SEMANTIC_SPAN_TEXT, source_doc_id="d")
    assert len(chunks) == 1
    c = chunks[0]
    # The offset<->text contract every strategy is supposed to uphold.
    assert _SEMANTIC_SPAN_TEXT[c.start_offset : c.end_offset] == c.text
    # Inter-sentence whitespace is preserved (the bug dropped it).
    assert " " in c.text
    assert c.text == _SEMANTIC_SPAN_TEXT


def test_semantic_snippet_spanning_sentence_boundary_is_retrievable():
    # A snippet that straddles a sentence boundary (the space between two
    # sentences) must survive into chunk.text — the whitespace-dropping bug
    # made such a substring unmatchable, undercounting snippet-hit faithfulness.
    s = SemanticBoundaryStrategy(
        embedder=HashEmbedder(), distance_threshold=2.0, min_chunk_chars=0, max_chunk_chars=10_000
    )
    c = s.chunk(_SEMANTIC_SPAN_TEXT, source_doc_id="d")[0]
    assert "animals. Dogs" in c.text


def test_semantic_min_merge_preserves_offsets():
    # min_chunk_chars forces a forward merge; the merged chunk must still
    # satisfy source[start:end] == text.
    s = SemanticBoundaryStrategy(
        embedder=HashEmbedder(), distance_threshold=0.0, min_chunk_chars=40, max_chunk_chars=10_000
    )
    chunks = s.chunk(_SEMANTIC_SPAN_TEXT, source_doc_id="d")
    assert len(chunks) >= 1
    for c in chunks:
        assert _SEMANTIC_SPAN_TEXT[c.start_offset : c.end_offset] == c.text


def test_semantic_max_split_preserves_offsets():
    # Small max_chunk_chars forces the size-capped split path within a block.
    s = SemanticBoundaryStrategy(
        embedder=HashEmbedder(), distance_threshold=2.0, min_chunk_chars=0, max_chunk_chars=30
    )
    chunks = s.chunk(_SEMANTIC_SPAN_TEXT, source_doc_id="d")
    assert len(chunks) > 1
    for c in chunks:
        assert _SEMANTIC_SPAN_TEXT[c.start_offset : c.end_offset] == c.text
        # Hard ceiling holds on every emitted chunk (#54).
        assert len(c.text) <= 30


# ----------------------------------------------------------------------
# #54 — a single sentence longer than max_chunk_chars must be char-split so the
# documented hard ceiling holds (sentence-boundary splitting alone can't reduce
# an oversized single sentence).
# ----------------------------------------------------------------------

# One short sentence, then one deliberately long single sentence (no internal
# sentence punctuation), then a short one. The middle sentence alone is far
# longer than the cap below.
_OVERSIZED_SENTENCE_TEXT = (
    "Intro line here. "
    "This single run on sentence keeps going and going with many words and no "
    "internal full stops so it cannot be reduced at sentence boundaries at all. "
    "Outro line."
)


def test_semantic_oversized_single_sentence_respects_ceiling():
    # Pre-#54 the middle sentence (well over 40 chars) was emitted as one chunk
    # far above the cap, silently breaching the documented hard ceiling.
    cap = 40
    s = SemanticBoundaryStrategy(
        embedder=HashEmbedder(), distance_threshold=2.0, min_chunk_chars=0, max_chunk_chars=cap
    )
    chunks = s.chunk(_OVERSIZED_SENTENCE_TEXT, source_doc_id="d")
    assert chunks
    # The ceiling must hold on every chunk, including char-split pieces. This is
    # the core regression catcher: pre-#54 the long middle sentence was emitted
    # as one chunk ~75 chars, far over the 40-char cap.
    assert all(len(c.text) <= cap for c in chunks)
    # Offset<->text contract holds on every piece (#50 / D-005).
    for c in chunks:
        assert _OVERSIZED_SENTENCE_TEXT[c.start_offset : c.end_offset] == c.text
    # The oversized sentence had to be char-split, so we get more chunks than the
    # three source sentences — proof the split path actually ran.
    assert len(chunks) >= 4


# ----------------------------------------------------------------------
# #52 — the min-merge pass must not breach the max_chunk_chars hard ceiling.
# ----------------------------------------------------------------------

# Four short, topically-distinct sentences. With min_chunk_chars high enough
# to force forward merges and max_chunk_chars just above one sentence, a
# naive merge would chain sentences past the ceiling.
_MERGE_CEILING_TEXT = (
    "Postgres database configuration requires careful tuning of buffer settings. "
    "The weather today is sunny and warm for the season. "
    "Machine learning models need large training datasets. "
    "Vector operations are fundamental to linear algebra. "
)


def test_semantic_merge_never_breaches_max_chunk_chars():
    # Before #52, the forward merge (min_chunk_chars) ran after _emit_block and
    # could combine chunks past the ceiling — here it produced 127- and 106-char
    # chunks against a 100-char cap. The hard ceiling must win on conflict.
    s = SemanticBoundaryStrategy(
        embedder=HashEmbedder(),
        distance_threshold=0.4,
        min_chunk_chars=80,
        max_chunk_chars=100,
    )
    chunks = s.chunk(_MERGE_CEILING_TEXT, source_doc_id="d")
    assert chunks
    assert all(len(c.text) <= 100 for c in chunks)
    # Offset<->text contract still holds on the (now un-merged) capped path.
    for c in chunks:
        assert _MERGE_CEILING_TEXT[c.start_offset : c.end_offset] == c.text


def test_semantic_merge_still_happens_when_it_fits():
    # The ceiling guard must not disable legitimate merges: with a generous
    # ceiling, too-small chunks still merge forward, yielding fewer chunks than
    # the no-merge baseline (min_chunk_chars=0). Compare the two directly so the
    # assertion is robust to how many sentences pair up.
    unmerged = SemanticBoundaryStrategy(
        embedder=HashEmbedder(),
        distance_threshold=0.4,
        min_chunk_chars=0,
        max_chunk_chars=10_000,
    ).chunk(_MERGE_CEILING_TEXT, source_doc_id="d")
    merged = SemanticBoundaryStrategy(
        embedder=HashEmbedder(),
        distance_threshold=0.4,
        min_chunk_chars=80,
        max_chunk_chars=10_000,
    ).chunk(_MERGE_CEILING_TEXT, source_doc_id="d")
    # Merging genuinely reduced the chunk count (the guard didn't block it).
    assert len(merged) < len(unmerged)
    for c in merged:
        assert len(c.text) <= 10_000
        assert _MERGE_CEILING_TEXT[c.start_offset : c.end_offset] == c.text


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


# Issue #31: completes the #29 sweep. StructureAwareStrategy was the only
# strategy whose constructor still used range-only / sign-only checks, so
# `True` silently bound `max_heading_level=1` (degrading the chunker to only
# split on `#`), and `4000.0` / `NaN` / `Inf` silently bound `max_chunk_chars`
# and surfaced as misleading errors from the FixedSizeStrategy fallback.


@pytest.mark.parametrize("bad", _BAD_INT)
def test_structure_max_heading_level_must_be_int(bad):
    with pytest.raises(ValueError, match="max_heading_level must be an int"):
        StructureAwareStrategy(max_heading_level=bad)


@pytest.mark.parametrize("bad", _BAD_INT)
def test_structure_max_chunk_chars_must_be_int(bad):
    with pytest.raises(ValueError, match="max_chunk_chars must be an int"):
        StructureAwareStrategy(max_chunk_chars=bad)


@pytest.mark.parametrize("good", [1, 2, 3, 4, 5, 6])
def test_structure_accepts_valid_max_heading_level(good):
    s = StructureAwareStrategy(max_heading_level=good)
    assert s.max_heading_level == good


@pytest.mark.parametrize("bad", [0, -1, -6, 7, 10])
def test_structure_max_heading_level_range_check_preserved(bad):
    # Existing range error must remain reachable after the new isinstance
    # check fires for non-int / bool cases. Plain ints out-of-range still get
    # the original error message.
    with pytest.raises(ValueError, match=r"max_heading_level must be in \[1, 6\]"):
        StructureAwareStrategy(max_heading_level=bad)


@pytest.mark.parametrize("good", [1, 1000, 4000, 100_000])
def test_structure_accepts_valid_max_chunk_chars(good):
    s = StructureAwareStrategy(max_chunk_chars=good)
    assert s.max_chunk_chars == good


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_structure_max_chunk_chars_positive_check_preserved(bad):
    # Existing positive error must remain reachable for plain non-positive ints.
    with pytest.raises(ValueError, match="max_chunk_chars must be positive"):
        StructureAwareStrategy(max_chunk_chars=bad)
