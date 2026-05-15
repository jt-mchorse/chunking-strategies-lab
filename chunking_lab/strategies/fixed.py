"""Fixed-size sliding-window chunker.

The simplest strategy and the baseline against which the others are
compared. Two parameters: chunk size in characters, and overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import Chunk


@dataclass
class FixedSizeStrategy:
    """Slide a fixed-size window over the input text with optional overlap."""

    name: str = "fixed-size"
    chunk_chars: int = 800
    overlap_chars: int = 100

    def __post_init__(self) -> None:
        if self.chunk_chars <= 0:
            raise ValueError(f"chunk_chars must be positive; got {self.chunk_chars}")
        if self.overlap_chars < 0:
            raise ValueError(f"overlap_chars must be >= 0; got {self.overlap_chars}")
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError(
                f"overlap_chars ({self.overlap_chars}) must be < chunk_chars ({self.chunk_chars})"
            )

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        if not text:
            return []
        chunks: list[Chunk] = []
        stride = self.chunk_chars - self.overlap_chars
        start = 0
        while start < len(text):
            end = min(start + self.chunk_chars, len(text))
            chunks.append(
                Chunk(
                    text=text[start:end],
                    start_offset=start,
                    end_offset=end,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={"chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars},
                )
            )
            if end == len(text):
                break
            start += stride
        return chunks
