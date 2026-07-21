"""Semantic embedding-boundary chunker.

Embed each sentence; compute cosine distance between adjacent sentences;
split at high-distance peaks. The intuition: cosine distance is small
between sentences in the same semantic unit and spikes at topic boundaries.

Uses the repo's pinned `Embedder` Protocol so the same call shape works
under HashEmbedder (CI) and MiniLM (operator-grade quality).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from chunking_lab.embedder import Embedder

from . import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences_with_offsets(text: str) -> list[tuple[str, int]]:
    """Return a list of (sentence_text, start_offset) preserving indices."""
    if not text:
        return []
    sentences: list[tuple[str, int]] = []
    cursor = 0
    for m in _SENTENCE_RE.finditer(text):
        end = m.start()
        if end > cursor:
            sentences.append((text[cursor:end], cursor))
        cursor = m.end()
    if cursor < len(text):
        sentences.append((text[cursor:], cursor))
    return sentences


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    dist = 1.0 - (dot / (na * nb))
    # The zero-norm degenerate input is handled above, but a non-finite
    # (NaN/±Inf) component in `a`/`b` slips that guard and makes `dist`
    # non-finite (#66). `SemanticBoundaryStrategy.chunk` then evaluates
    # `dist >= distance_threshold`, which is silently False for a NaN, so the
    # topic boundary is suppressed and the chunk is silently under-segmented.
    # `Embedder` is a BYO Protocol (MiniLM or custom), so a normalization
    # divide-by-zero / Inf overflow / NaN-poisoned model output can hand back
    # such a component. This helper is the only line of defense (no upstream
    # seam validation), so fail loud rather than return a fallback that would
    # hide the poison — the result-finiteness check llm-cost-optimizer's
    # cosine() uses (#88), but raising. Sibling of rag-production-kit #82.
    if not math.isfinite(dist):
        raise ValueError(
            "non-finite cosine distance from a NaN/Inf embedding component; "
            "the embedder returned a non-finite vector"
        )
    return dist


@dataclass
class SemanticBoundaryStrategy:
    """Split where adjacent-sentence cosine distance crosses a threshold."""

    embedder: Embedder
    name: str = "semantic"
    distance_threshold: float = 0.4
    """Distance above which adjacent sentences are considered topic-changing."""
    min_chunk_chars: int = 80
    """Don't emit chunks shorter than this; merge them with the next."""
    max_chunk_chars: int = 1600
    """Hard ceiling — even within a single semantic block, split if exceeded."""

    def __post_init__(self) -> None:
        # Type/bool guard before the range check — the sibling of the #29 integer
        # guards below, which `distance_threshold` (the one float field) skipped.
        # bool subclasses int, so `0.0 <= True <= 2.0` is True and a boolean
        # threshold silently acts as 1.0, flattening operator intent; a
        # present-but-non-numeric value (a string/None from a hand-authored
        # config) raises a raw TypeError at `0.0 <= x` instead of the clean
        # ValueError the sibling fields produce. (The range check alone already
        # rejects NaN/inf — every comparison against them is False.) Mirrors the
        # portfolio float-field pattern (prs Prompt.temperature, ems recall_at_k).
        if isinstance(self.distance_threshold, bool) or not isinstance(
            self.distance_threshold, (int, float)
        ):
            raise ValueError(
                f"distance_threshold must be a number; got {self.distance_threshold!r}"
            )
        if not 0.0 <= self.distance_threshold <= 2.0:
            raise ValueError(f"distance_threshold must be in [0, 2]; got {self.distance_threshold}")
        # Integer guards (#29) — see FixedSizeStrategy for harm rationale.
        if not isinstance(self.min_chunk_chars, int) or isinstance(self.min_chunk_chars, bool):
            raise ValueError(f"min_chunk_chars must be an int; got {self.min_chunk_chars!r}")
        if self.min_chunk_chars < 0:
            raise ValueError(f"min_chunk_chars must be >= 0; got {self.min_chunk_chars}")
        if not isinstance(self.max_chunk_chars, int) or isinstance(self.max_chunk_chars, bool):
            raise ValueError(f"max_chunk_chars must be an int; got {self.max_chunk_chars!r}")
        if self.max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive; got {self.max_chunk_chars}")

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        sentences = _split_sentences_with_offsets(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            # Route through _emit_block so max_chunk_chars is applied even on the
            # single-sentence path. The hand-rolled early return that lived here
            # emitted the whole sentence uncapped, silently breaching the
            # documented hard ceiling (line 61-62) — the #54 fix hardened only
            # the multi-sentence path, which never reaches this branch (#64).
            # _emit_block emits a single {distance_threshold} chunk within the
            # cap (identical to the old behavior) and char-splits into
            # size_capped pieces when over, preserving the offset contract (#50).
            single: list[Chunk] = []
            self._emit_block(text, sentences, 0, 1, source_doc_id, single)
            return single

        # Embed each sentence; compute pairwise distances.
        embeddings = [self.embedder.embed(s) for s, _ in sentences]
        boundaries: list[int] = []  # indices where a split happens (between i and i+1)
        for i in range(len(sentences) - 1):
            dist = _cosine_distance(embeddings[i], embeddings[i + 1])
            if dist >= self.distance_threshold:
                boundaries.append(i + 1)

        # Walk sentences and emit chunks at boundaries; respect min/max sizes.
        chunks: list[Chunk] = []
        cursor = 0  # sentence index
        while cursor < len(sentences):
            # Find the next boundary at or beyond cursor.
            next_b = next((b for b in boundaries if b > cursor), len(sentences))
            # Build a chunk from sentences[cursor:next_b], honoring max_chunk_chars.
            self._emit_block(text, sentences, cursor, next_b, source_doc_id, chunks)
            cursor = next_b

        # Merge any chunk shorter than min_chunk_chars with its successor.
        if self.min_chunk_chars > 0:
            chunks = self._merge_too_small(text, chunks, source_doc_id)
        return chunks

    def _emit_block(
        self,
        text: str,
        sentences: list[tuple[str, int]],
        start_idx: int,
        end_idx: int,
        source_doc_id: str,
        out: list[Chunk],
    ) -> None:
        # Emit sentences[start_idx:end_idx] as a chunk, splitting if it exceeds
        # max. Chunk text is sliced from the original `text` (not a join of the
        # stripped sentence list) so inter-sentence whitespace is preserved and
        # `text[start_offset:end_offset] == chunk.text` holds — the same
        # offset<->text contract fixed/recursive uphold (#50). A join of the
        # stripped sentences dropped the separating whitespace, which
        # undercounted end_offset and broke the snippet-substring metric for
        # passages spanning a sentence boundary.
        if start_idx >= end_idx:
            return
        block_start = sentences[start_idx][1]
        # Tile blocks contiguously: end this block at the NEXT block's start (the
        # next sentence's offset), or len(text) for the final block — carrying the
        # inter-block separator as this block's trailing text. Deriving block_end
        # from the last sentence's text end instead dropped the whitespace BETWEEN
        # topic blocks: those characters fell into no chunk (a coverage gap), so
        # concatenating chunks by offset no longer reconstructed the source and a
        # snippet straddling a topic boundary became unretrievable. This is the
        # between-block twin of the within-block gaps #50/#74 closed — those fixes
        # only covered the whole-block and greedy-split-within-a-block paths.
        # Per-chunk `text[start:end] == chunk.text` held either way (each block's
        # offsets sliced its own text), which is why the offset tests missed it.
        # Because the next block's `block_start` is exactly this `sentences[
        # end_idx][1]`, consecutive blocks now abut with no gap and no overlap.
        block_end = sentences[end_idx][1] if end_idx < len(sentences) else len(text)

        if (block_end - block_start) <= self.max_chunk_chars:
            out.append(
                Chunk(
                    text=text[block_start:block_end],
                    start_offset=block_start,
                    end_offset=block_end,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={"distance_threshold": self.distance_threshold},
                )
            )
            return

        # Block too big — greedily pack whole sentences into runs that stay
        # within the cap. Span length is measured source-side (end - start) so
        # the whitespace now carried in the chunk text is counted against the
        # cap consistently. A single sentence longer than the cap can't be
        # reduced at sentence granularity, so `_append_capped` char-splits it
        # (mirroring StructureAwareStrategy's section fallback) — otherwise the
        # documented hard ceiling (line 61-62) is silently breached.
        run_start = block_start
        run_end = block_start
        for sent, sent_start in sentences[start_idx:end_idx]:
            sent_end = sent_start + len(sent)
            if run_start < run_end and (sent_end - run_start) > self.max_chunk_chars:
                # Flush the packed run, then begin the next run where this one
                # ended (`run_start = run_end`) — NOT at the next sentence's
                # start. Starting at the sentence start dropped the
                # inter-sentence whitespace in the gap, so concatenating chunks
                # by offset no longer reconstructed the source and a snippet
                # straddling a greedy split (inside one semantic block) became
                # unmatchable (#74). Carrying the gap as the next run's leading
                # text keeps coverage identical to the whole-block path above,
                # extending the #50 whitespace-preservation contract to the cap
                # path. Cap accounting is unchanged: the span is still measured
                # from `run_start` to the candidate sentence end.
                self._append_capped(text, run_start, run_end, source_doc_id, out)
                run_start = run_end
            run_end = sent_end
        # Extend the final run to `block_end` (not the last sentence's text end)
        # so the trailing inter-block separator is carried in the last piece too,
        # matching the whole-block tiling above. Intermediate runs still flush at
        # sentence ends, carrying within-block gaps as the next run's leading text.
        if run_start < block_end:
            self._append_capped(text, run_start, block_end, source_doc_id, out)

    def _append_capped(
        self,
        text: str,
        start: int,
        end: int,
        source_doc_id: str,
        out: list[Chunk],
    ) -> None:
        """Emit ``text[start:end]`` as one or more chunks, each within the
        ``max_chunk_chars`` hard ceiling.

        A run packed from whole sentences is normally already within the cap and
        emits as a single chunk. But a single sentence longer than
        ``max_chunk_chars`` cannot be reduced at sentence granularity, so it is
        char-split into cap-sized pieces — the same fallback
        ``StructureAwareStrategy`` uses for an oversized heading section — so the
        documented hard ceiling holds even for an unsplittable-at-sentence span.
        Offsets are sliced from ``text`` so ``text[start_offset:end_offset] ==
        chunk.text`` (#50) is preserved on every piece.
        """
        cursor = start
        while cursor < end:
            piece_end = min(cursor + self.max_chunk_chars, end)
            out.append(
                Chunk(
                    text=text[cursor:piece_end],
                    start_offset=cursor,
                    end_offset=piece_end,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={
                        "distance_threshold": self.distance_threshold,
                        "size_capped": True,
                    },
                )
            )
            cursor = piece_end

    def _merge_too_small(self, text: str, chunks: list[Chunk], source_doc_id: str) -> list[Chunk]:
        if not chunks:
            return chunks
        merged: list[Chunk] = []
        for c in chunks:
            # Merge a too-small chunk into its successor — but only when the
            # combined span still fits the max_chunk_chars hard ceiling (#52).
            # `max_chunk_chars` is documented as a hard ceiling (line 61-62)
            # while `min_chunk_chars` is a soft "prefer not shorter than this",
            # so on conflict the ceiling wins: if merging would breach it, leave
            # the small chunk as-is rather than emit a chunk over the cap. The
            # span is measured source-side (`c.end_offset - last.start_offset`),
            # matching `_emit_block`'s cap check and equal to the length of the
            # merged text slice, so the offset<->text contract (#50) holds.
            if (
                merged
                and len(merged[-1].text) < self.min_chunk_chars
                and (c.end_offset - merged[-1].start_offset) <= self.max_chunk_chars
            ):
                last = merged[-1]
                merged[-1] = Chunk(
                    text=text[last.start_offset : c.end_offset],
                    start_offset=last.start_offset,
                    end_offset=c.end_offset,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata=last.metadata,
                )
            else:
                merged.append(c)
        return merged
