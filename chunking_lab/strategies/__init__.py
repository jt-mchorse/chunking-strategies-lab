"""Chunking strategies for the lab.

Five implementations, each in its own module, sharing one interface:

    class Strategy(Protocol):
        name: str
        def chunk(self, text: str, **opts) -> list[Chunk]: ...

Strategies (D-004 — each is a standalone module so a reader can copy one
without dragging in siblings):

- `fixed`     — fixed-size sliding window with optional overlap. Baseline.
- `recursive` — split on a hierarchy of separators until each chunk fits.
- `semantic`  — embedding-boundary chunker; splits at adjacent-sentence
                cosine peaks. Uses the repo's pinned Embedder.
- `late`      — late chunking. Returns (chunk, vector) pairs because each
                chunk's vector is derived from the document-level
                embedding, not the chunk text in isolation (D-006).
- `structure` — markdown-heading-aware chunker; one chunk per
                heading-bounded section.

`Chunk` carries `start_offset` / `end_offset` (D-005) so #3's metrics
matrix can attribute retrieved chunks back to source documents without
re-tokenizing. These are Unicode codepoint offsets (Python `str` indices):
`source_text[start_offset:end_offset] == chunk.text`, which differs from a
byte slice on multibyte text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    """One chunk produced by a strategy."""

    text: str
    # Offsets are Unicode CODEPOINT offsets (Python str indices), NOT byte
    # offsets: strategies populate them via `text[start:end]`, so the invariant
    # is `source_text[start_offset:end_offset] == chunk.text`. On multibyte
    # text these differ from byte offsets — slicing `source.encode()` with them
    # splits characters. See tests/test_strategies.py offset-contract tests.
    start_offset: int  # inclusive codepoint offset into source text
    end_offset: int  # exclusive codepoint offset into source text
    source_doc_id: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError(f"start_offset must be >= 0; got {self.start_offset}")
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be >= start_offset ({self.start_offset})"
            )


@dataclass(frozen=True)
class LateChunk:
    """One late-chunk: chunk plus a vector derived from document-level context (D-006)."""

    chunk: Chunk
    vector: tuple[float, ...]


class Strategy(Protocol):
    """Single-method seam every strategy implements."""

    name: str

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        """Split `text` into chunks. Each chunk's offsets index into `text`."""
        ...


from .fixed import FixedSizeStrategy  # noqa: E402
from .late import LateChunkingStrategy  # noqa: E402
from .recursive import RecursiveStrategy  # noqa: E402
from .semantic import SemanticBoundaryStrategy  # noqa: E402
from .structure import StructureAwareStrategy  # noqa: E402

__all__ = [
    "Chunk",
    "FixedSizeStrategy",
    "LateChunk",
    "LateChunkingStrategy",
    "RecursiveStrategy",
    "SemanticBoundaryStrategy",
    "Strategy",
    "StructureAwareStrategy",
]
