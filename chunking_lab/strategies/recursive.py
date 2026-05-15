"""Recursive chunker.

Split on a hierarchy of separators — paragraph → newline → sentence → space —
descending one level at a time only when a chunk still exceeds the size budget.
This is the LangChain `RecursiveCharacterTextSplitter` shape, dep-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Chunk

# Default separator hierarchy: try larger boundaries first to preserve
# semantic units. The empty separator at the end is the brute-force fallback.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass
class RecursiveStrategy:
    """Recursive split on a hierarchy of separators until each chunk fits."""

    name: str = "recursive"
    chunk_chars: int = 800
    separators: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SEPARATORS)

    def __post_init__(self) -> None:
        if self.chunk_chars <= 0:
            raise ValueError(f"chunk_chars must be positive; got {self.chunk_chars}")
        if not self.separators:
            raise ValueError("separators must be non-empty")

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        if not text:
            return []
        chunks: list[Chunk] = []
        for chunk_text, start in self._split_recursive(text, 0, list(self.separators)):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_offset=start,
                    end_offset=start + len(chunk_text),
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={"chunk_chars": self.chunk_chars},
                )
            )
        return chunks

    def _split_recursive(
        self, text: str, base_offset: int, separators: list[str]
    ) -> list[tuple[str, int]]:
        # If the whole text fits, return it as one chunk.
        if len(text) <= self.chunk_chars:
            return [(text, base_offset)] if text else []

        if not separators:
            # No separators left — brute-force split at chunk_chars boundaries.
            out: list[tuple[str, int]] = []
            for i in range(0, len(text), self.chunk_chars):
                out.append((text[i : i + self.chunk_chars], base_offset + i))
            return out

        sep, rest = separators[0], separators[1:]
        if sep == "":
            # Empty-separator level: brute-force split at chunk_chars.
            out: list[tuple[str, int]] = []
            for i in range(0, len(text), self.chunk_chars):
                out.append((text[i : i + self.chunk_chars], base_offset + i))
            return out

        # Split, but track each piece's offset back into the original.
        pieces: list[tuple[str, int]] = []
        cursor = 0
        while cursor < len(text):
            idx = text.find(sep, cursor)
            if idx == -1:
                pieces.append((text[cursor:], base_offset + cursor))
                break
            piece = text[cursor : idx + len(sep)]
            pieces.append((piece, base_offset + cursor))
            cursor = idx + len(sep)

        # Greedy merge adjacent pieces while the running buffer fits.
        out: list[tuple[str, int]] = []
        buf = ""
        buf_start = -1
        for piece_text, piece_start in pieces:
            if buf == "":
                buf = piece_text
                buf_start = piece_start
                continue
            if len(buf) + len(piece_text) <= self.chunk_chars:
                buf += piece_text
            else:
                if len(buf) <= self.chunk_chars:
                    out.append((buf, buf_start))
                else:
                    out.extend(self._split_recursive(buf, buf_start, rest))
                buf = piece_text
                buf_start = piece_start
        if buf:
            if len(buf) <= self.chunk_chars:
                out.append((buf, buf_start))
            else:
                out.extend(self._split_recursive(buf, buf_start, rest))
        return out
