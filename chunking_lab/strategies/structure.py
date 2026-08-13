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

# A fenced code block opener: an optional indent, then a run of 3+ backticks or
# tildes, then an optional info string (` ```python `). CommonMark allows both
# fence characters, and a fence is closed only by a run of the SAME character at
# least as long as the opener — so a ```` block is not closed by an inner ```.
_FENCE_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)$", re.MULTILINE)


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Return the `[start, end)` character spans covered by fenced code blocks.

    `_HEADING_RE` is line-anchored and has no notion of code fences, so a `#`
    (or `###`) comment at the start of a line inside a ```` ``` ```` block used
    to be treated as an ATX heading (#152). That tore the fence in half across
    two chunks and — worse — promoted the comment text into the chunk's
    ``title`` metadata, the field the module docstring designates as a
    retrieval signal. This computes the fenced regions once so `chunk` can drop
    any heading match that starts inside one.

    An unclosed fence extends to end-of-text. That is the conservative
    direction: the strategy under-splits rather than inventing headings out of
    code, which is the failure this fixes.

    Indented (four-space) code blocks are deliberately not handled. Telling
    them apart from list continuation needs a real block parser, and this
    package is dep-free by design (D-002) — so that is a much larger change
    than this bug justifies, not an oversight.
    """
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    open_char = ""
    open_len = 0
    for m in _FENCE_RE.finditer(text):
        fence = m.group("fence")
        if open_at is None:
            open_at = m.start()
            open_char = fence[0]
            open_len = len(fence)
            continue
        # A closing fence must use the same character, be at least as long as
        # the opener, and carry no info string.
        if fence[0] == open_char and len(fence) >= open_len and not m.group("info").strip():
            spans.append((open_at, m.end()))
            open_at = None
    if open_at is not None:
        spans.append((open_at, len(text)))
    return spans


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


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
        # Drop matches inside fenced code blocks: a `# comment` line in a
        # ```python block is not a heading (#152).
        fences = _fenced_spans(text)
        headings = [
            (m.start(), m.end(), len(m.group(1)), m.group(2))
            for m in _HEADING_RE.finditer(text)
            if len(m.group(1)) <= self.max_heading_level and not _in_spans(m.start(), fences)
        ]
        if not headings:
            # No headings — fallback: title = first non-empty line. Still honor
            # max_chunk_chars: a long unheaded document (plain text, Setext
            # `===`/`---` headings or RST that the ATX-only _HEADING_RE doesn't
            # match) must not bypass the ceiling and emit one monster chunk.
            # Mirrors the cap logic the heading-bounded path uses; a follow-up
            # can unify both through `_emit_capped` once #56's PR lands.
            first_line = next(
                (line.strip() for line in text.splitlines() if line.strip()), source_doc_id
            )
            base_meta = {"title": first_line, "heading_level": None}
            if len(text) <= self.max_chunk_chars:
                return [
                    Chunk(
                        text=text,
                        start_offset=0,
                        end_offset=len(text),
                        source_doc_id=source_doc_id,
                        strategy_name=self.name,
                        metadata=dict(base_meta),
                    )
                ]
            fallback_chunks: list[Chunk] = []
            cursor = 0
            piece_idx = 0
            while cursor < len(text):
                piece_end = min(cursor + self.max_chunk_chars, len(text))
                fallback_chunks.append(
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
            return fallback_chunks

        chunks: list[Chunk] = []
        # Possibly emit a leading chunk before the first heading. Route it
        # through the same cap-aware emitter as the sections so a long
        # preamble (a title block / abstract / intro before the first `#`)
        # can't bypass max_chunk_chars and produce a monster chunk.
        first_heading_start = headings[0][0]
        if first_heading_start > 0 and text[:first_heading_start].strip():
            self._emit_capped(
                text,
                0,
                first_heading_start,
                {"title": "<preamble>", "heading_level": None},
                chunks,
                source_doc_id,
            )
            first_section_start = first_heading_start
        else:
            # An empty or whitespace-only preamble: don't emit a useless
            # whitespace chunk, but don't DROP those source codepoints either.
            # Fold any leading whitespace into the first section so the heading
            # path preserves full coverage — matching the content-preamble path
            # above (#56) and the no-headings fallback, whose full-coverage
            # invariant is locked by test_structure_caps_oversized_unheaded_
            # fallback. Pre-fix, a doc beginning with a blank line before its
            # first `#` dropped the leading whitespace (the corner #56 missed).
            # When there's no preamble at all this is just 0 == first_heading_start.
            first_section_start = 0

        for i, (h_start, _h_end, level, title) in enumerate(headings):
            start = first_section_start if i == 0 else h_start
            section_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            self._emit_capped(
                text,
                start,
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
