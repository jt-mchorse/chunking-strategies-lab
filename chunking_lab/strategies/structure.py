"""Markdown-structure-aware chunker.

For markdown corpora (which is what `chunking_lab/data/corpus/` is), split
on heading boundaries (`#`, `##`, `###`, etc.) and treat each
heading-bounded section as one chunk. The heading text itself is included
in the chunk's metadata so retrieval can use it as a title field.

If the document has no headings, falls back to one chunk per document with
the document's first non-empty line as the title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class StructureAwareStrategy:
    """One chunk per markdown-heading-bounded section."""

    name: str = "structure-aware"
    max_heading_level: int = 6
    """Don't split below this heading level. e.g., max_heading_level=2 only
    splits on `#` and `##`, treating `###` and below as part of the surrounding
    `##` section."""
    max_chunk_chars: int = 4000
    """If a heading-bounded section exceeds this, fall back to fixed-size
    splitting inside the section so a single huge section doesn't produce a
    monster chunk."""

    def __post_init__(self) -> None:
        # Integer guards (#31) — completes the #29 sweep that tightened the
        # other four strategy constructors. Sign-only / range-only accepted
        # `True` (silently bound to 1 — chunker degraded to only splitting on
        # `#`), and `4000.0` / `4000.5` / `NaN` / `Inf` for max_chunk_chars
        # (silently bound, surfaced later in the FixedSizeStrategy fallback
        # with a misleading internal-site error message).
        if not isinstance(self.max_heading_level, int) or isinstance(self.max_heading_level, bool):
            raise ValueError(f"max_heading_level must be an int; got {self.max_heading_level!r}")
        if not 1 <= self.max_heading_level <= 6:
            raise ValueError(f"max_heading_level must be in [1, 6]; got {self.max_heading_level}")
        if not isinstance(self.max_chunk_chars, int) or isinstance(self.max_chunk_chars, bool):
            raise ValueError(f"max_chunk_chars must be an int; got {self.max_chunk_chars!r}")
        if self.max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive; got {self.max_chunk_chars}")

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        if not text:
            return []
        headings = [
            (m.start(), m.end(), len(m.group(1)), m.group(2))
            for m in _HEADING_RE.finditer(text)
            if len(m.group(1)) <= self.max_heading_level
        ]
        if not headings:
            # No headings — fallback: one chunk per document, title = first
            # non-empty line.
            first_line = next(
                (line.strip() for line in text.splitlines() if line.strip()), source_doc_id
            )
            return [
                Chunk(
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={"title": first_line, "heading_level": None},
                )
            ]

        chunks: list[Chunk] = []
        # Possibly emit a leading chunk before the first heading. Route it
        # through the same cap-aware emitter as the sections so a long
        # preamble (a title block / abstract / intro before the first `#`)
        # can't bypass max_chunk_chars and produce a monster chunk.
        if headings[0][0] > 0 and text[: headings[0][0]].strip():
            self._emit_capped(
                text,
                0,
                headings[0][0],
                {"title": "<preamble>", "heading_level": None},
                chunks,
                source_doc_id,
            )

        for i, (h_start, _h_end, level, title) in enumerate(headings):
            section_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            self._emit_capped(
                text,
                h_start,
                section_end,
                {"title": title, "heading_level": level},
                chunks,
                source_doc_id,
            )
        return chunks

    def _emit_capped(
        self,
        text: str,
        start: int,
        end: int,
        base_meta: dict,
        chunks: list[Chunk],
        source_doc_id: str,
    ) -> None:
        """Emit chunks for the span ``text[start:end]`` without ever exceeding
        ``max_chunk_chars``. A span within the ceiling becomes one chunk; a
        longer span is sliced into max_chunk_chars-sized pieces, each tagged
        with ``piece_idx`` and carrying ``base_meta`` so retrieval keeps the
        title on every piece."""
        span = text[start:end]
        if len(span) <= self.max_chunk_chars:
            chunks.append(
                Chunk(
                    text=span,
                    start_offset=start,
                    end_offset=end,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata=dict(base_meta),
                )
            )
            return
        # Span too long — slice into max_chunk_chars-sized pieces.
        cursor = start
        piece_idx = 0
        while cursor < end:
            piece_end = min(cursor + self.max_chunk_chars, end)
            chunks.append(
                Chunk(
                    text=text[cursor:piece_end],
                    start_offset=cursor,
                    end_offset=piece_end,
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                    metadata={**base_meta, "piece_idx": piece_idx},
                )
            )
            cursor = piece_end
            piece_idx += 1
