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
    return 1.0 - (dot / (na * nb))


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
            sent, start = sentences[0]
            return [
                Chunk(
                    text=sent,
                    start_offset=start,
                    end_offset=start + len(sent),
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={"distance_threshold": self.distance_threshold},
                )
            ]

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
        block_end = sentences[end_idx - 1][1] + len(sentences[end_idx - 1][0])

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

        # Block too big — split at sentence boundaries. Span length is measured
        # source-side (end - start) so the whitespace now carried in the chunk
        # text is counted against the cap consistently.
        run_start: int | None = None
        run_end = block_start
        for sent, sent_start in sentences[start_idx:end_idx]:
            sent_end = sent_start + len(sent)
            if run_start is not None and (sent_end - run_start) > self.max_chunk_chars:
                out.append(
                    Chunk(
                        text=text[run_start:run_end],
                        start_offset=run_start,
                        end_offset=run_end,
                        source_doc_id=source_doc_id,
                        strategy_name=self.name,
                        metadata={
                            "distance_threshold": self.distance_threshold,
                            "size_capped": True,
                        },
                    )
                )
                run_start = sent_start
                run_end = sent_end
            else:
                if run_start is None:
                    run_start = sent_start
                run_end = sent_end
        if run_start is not None:
            out.append(
                Chunk(
                    text=text[run_start:run_end],
                    start_offset=run_start,
                    end_offset=run_end,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={"distance_threshold": self.distance_threshold},
                )
            )

    def _merge_too_small(self, text: str, chunks: list[Chunk], source_doc_id: str) -> list[Chunk]:
        if not chunks:
            return chunks
        merged: list[Chunk] = []
        for c in chunks:
            if merged and len(merged[-1].text) < self.min_chunk_chars:
                last = merged[-1]
                # Rebuild the merged text from the source span so the merged
                # chunk keeps the offset<->text contract (#50): the gap between
                # `last` and `c` is whatever the source held between them.
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
