"""Late chunking.

Late chunking inverts the usual order: instead of "chunk → embed each
chunk independently", it embeds the *entire document* once and then
derives per-chunk vectors from contiguous slices of the document
embedding's token-level state.

This implementation is the "poor person's late chunking" suitable for
the lab's hermetic CI: we use a sliding window over the source text to
define chunk boundaries (so chunks have offsets like every other
strategy), but each chunk's vector comes from `embedder.embed(<full
document text>)` *blended* with the chunk's own text — concretely, an
elementwise average of the document embedding and the chunk embedding.

That's not what production late-chunking does (production uses
token-level hidden states from a long-context encoder), but the
*shape* of the result — `(chunk, vector)` pairs whose vector encodes
document-level context — is what downstream code (#3's metrics matrix)
needs to see. When the operator swaps in a real long-context embedder,
the call site doesn't change.
"""

from __future__ import annotations

from dataclasses import dataclass

from chunking_lab.embedder import Embedder

from . import Chunk, LateChunk


@dataclass
class LateChunkingStrategy:
    """Sliding-window chunks, vectors blended with document-level embedding."""

    embedder: Embedder
    name: str = "late-chunking"
    chunk_chars: int = 800
    overlap_chars: int = 100
    document_weight: float = 0.5
    """Weight applied to the document-level embedding when blending. 0.5 means
    the chunk's vector is the average of the chunk's own embedding and the
    document's. 1.0 means pure document embedding (every chunk is identical);
    0.0 means pure independent chunk embedding (degenerates to non-late)."""

    def __post_init__(self) -> None:
        # Integer guards (#29) — see FixedSizeStrategy for the harm rationale.
        if not isinstance(self.chunk_chars, int) or isinstance(self.chunk_chars, bool):
            raise ValueError(f"chunk_chars must be an int; got {self.chunk_chars!r}")
        if self.chunk_chars <= 0:
            raise ValueError(f"chunk_chars must be positive; got {self.chunk_chars}")
        if not isinstance(self.overlap_chars, int) or isinstance(self.overlap_chars, bool):
            raise ValueError(f"overlap_chars must be an int; got {self.overlap_chars!r}")
        if self.overlap_chars < 0:
            raise ValueError(f"overlap_chars must be >= 0; got {self.overlap_chars}")
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError(
                f"overlap_chars ({self.overlap_chars}) must be < chunk_chars ({self.chunk_chars})"
            )
        if not 0.0 <= self.document_weight <= 1.0:
            raise ValueError(f"document_weight must be in [0, 1]; got {self.document_weight}")

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        """Return Chunk objects (no vectors). Use `chunk_with_vectors` for late chunks."""
        return [lc.chunk for lc in self.chunk_with_vectors(text, source_doc_id=source_doc_id)]

    def chunk_with_vectors(self, text: str, *, source_doc_id: str = "doc") -> list[LateChunk]:
        if not text:
            return []
        doc_vector = self.embedder.embed(text)
        out: list[LateChunk] = []
        stride = self.chunk_chars - self.overlap_chars
        start = 0
        while start < len(text):
            end = min(start + self.chunk_chars, len(text))
            chunk_text = text[start:end]
            chunk_vector = self.embedder.embed(chunk_text)
            blended = _blend(doc_vector, chunk_vector, self.document_weight)
            chunk = Chunk(
                text=chunk_text,
                start_offset=start,
                end_offset=end,
                source_doc_id=source_doc_id,
                strategy_name=self.name,
                metadata={
                    "chunk_chars": self.chunk_chars,
                    "overlap_chars": self.overlap_chars,
                    "document_weight": self.document_weight,
                },
            )
            out.append(LateChunk(chunk=chunk, vector=tuple(blended)))
            if end == len(text):
                break
            start += stride
        return out


def _blend(doc_vec: list[float], chunk_vec: list[float], doc_weight: float) -> list[float]:
    if len(doc_vec) != len(chunk_vec):
        raise ValueError(f"vector length mismatch: {len(doc_vec)} vs {len(chunk_vec)}")
    return [
        doc_weight * d + (1.0 - doc_weight) * c for d, c in zip(doc_vec, chunk_vec, strict=True)
    ]
