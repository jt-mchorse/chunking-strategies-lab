"""Markdown-structure-aware chunker.

For markdown corpora (which is what `chunking_lab/data/corpus/` is), split
on heading boundaries (`#`, `##`, `###`, etc.) and treat each
heading-bounded section as one chunk. The heading text itself is included
in the chunk's metadata so retrieval can use it as a title field.

If no heading survives filtering, falls back to one chunk per document with
the document's first non-empty line outside a code fence (or an HTML comment,
#308) as the title, de-marked if it is a heading `max_heading_level` filtered
out (#288).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import Chunk, check_chunk_input

# `[ \t]`, not `\s`, between the hashes and the title — and an *optional*
# content group. `\s` matches newlines, so the pre-#154 pattern
# (`^(#{1,6})\s+(.+?)\s*$`) did not fail on an empty ATX heading (a `#` line
# with nothing after it, which CommonMark defines as a valid heading). It
# reached forward across the blank line and took a later paragraph as the
# match's title — including, when the next block was a fenced code block, the
# fence opener itself (` ```python `). That is the same corruption #152 fixed,
# in this same regex, by a route the fence guard cannot see: the match *starts*
# at the bare `#`, outside any fence, so `_in_spans(m.start(), fences)` is False
# and only the stolen text comes from inside.
#
# With the group optional, a bare `#` still opens a section — CommonMark agrees
# it is a heading, and the boundary was never the wrong part — it just no longer
# reaches into the following text for a title. `#hashtag` still doesn't match
# (no space), and 7+ hashes still don't (`#{1,6}` leaves a `#` that satisfies
# neither branch).
# The content group is `(\S.*?)`, not `(.+?)`: `.` matches a space, so with a
# plain `.+?` a hashes-then-only-spaces line (`###   `) still took the optional
# branch and captured a single space as the title — a whitespace-only retrieval
# signal, which is the same useless-title outcome in a quieter form. Requiring
# the first captured character to be non-blank sends that line down the
# no-content branch instead. Lazy `.*?` against the `[ \t]*$` tail still trims
# trailing spaces off a real title.
# The leading `^ {0,3}` is CommonMark's ATX indent allowance (#156). Pre-fix the
# pattern anchored the hashes hard to column 0, so a heading indented by even
# ONE space was not a heading at all — and since this regex is the only thing
# that opens a section, the boundary vanished and the following section merged
# into its predecessor, carrying the wrong title. Measured: `# Alpha` + `# Beta`
# is 2 chunks, and ` # Beta` (one space) is 1.
#
# Spaces-only and count-bounded, deliberately NOT `[ \t]*`: 4+ spaces is an
# indented code block, where a `#` is content rather than a heading, and an
# unbounded indent class is exactly the bug `_FENCE_RE` had below. Tabs are left
# out on purpose — CommonMark's tab-expansion rules are a separate problem, and
# half-implementing them here would be worse than not claiming them.
#
# The closing `(?:[ \t]+#+)?` drops an ATX *closing sequence* from the captured
# title. CommonMark §4.2: the optional trailing run of `#`s "is not considered
# part of the heading's content", but the lazy `(\S.*?)` was swallowing it, so
# `## Title ##` titled as `Title ##` — a polluted retrieval signal, the same
# class as the whitespace-only title #154's `(\S.*?)` was introduced to prevent.
#
# The closer is captured (group 3) for the one case the lazy title cannot tell
# apart (#242): with nothing before it, a closing run IS the whole "title", so
# `# #` titled `#` and `### ###` titled `###` where CommonMark example 79
# renders an empty heading. A title that is all `#`s and had no separate closer
# is that closer. `### ### ###` keeps its `###` title (example 79's sibling): it
# has a closer of its own.
#
# Line ends are `str.splitlines()`'s set, not `re.MULTILINE`'s `\n` (#297).
# `_fallback_title` finds lines with `splitlines()`, which also breaks on `\r`,
# `\x0b`, `\x0c`, `\x1c`-`\x1e`, `\x85`, U+2028 and U+2029. Under MULTILINE
# the title's `.*?` ran straight through all nine, so `### Install\x0cbody`
# titled `Install\x0cbody` when it survived the level filter and `Install` when
# it did not, and a CR-only document had no line starts after the first: one
# chunk, titled with the whole text. `_LINE_START`/`_LINE_END` anchor on the
# same set, and `[^{_EOL}]` replaces `.` in the title. The old `\r?$` tail is
# subsumed: `\r` is itself a line end.
#
# The title's first character is `[^ \t{_EOL}]`, not `\S` (#315). CommonMark
# allows only space and tab around the hashes, but Python's `\S` also refuses
# NBSP, U+2003, U+202F, U+3000 and the rest of Unicode's whitespace. So
# `# \xa0Install` -- an NBSP pasted from the web, or macOS Option-Space -- had
# no title the group would take, matched no heading at all, and its section
# merged under the previous title: #156's harm. A Unicode-blank first character
# is content to CommonMark; `_heading_title` trims it for display.
_EOL = "\n\r\x0b\x0c\x1c-\x1e\x85\u2028\u2029"
_LINE_START = rf"(?:^|(?<=[{_EOL}]))"
_LINE_END = rf"(?=[{_EOL}]|\Z)"
_HEADING_RE = re.compile(
    rf"{_LINE_START} {{0,3}}(#{{1,6}})(?:[ \t]+([^ \t{_EOL}][^{_EOL}]*?))?([ \t]+#+)?[ \t]*{_LINE_END}"
)


# Title for an empty ATX heading. A sentinel rather than `""` for the same
# reason `<preamble>` is one below: `title` is documented as a retrieval
# signal, and an empty string is a silently useless one that reads as "this
# document had no title" instead of "this section's heading was empty".
EMPTY_HEADING_TITLE = "<untitled>"


def _heading_title(m: re.Match[str]) -> str:
    title = m.group(2)
    if title is None or (m.group(3) is None and set(title) == {"#"}):
        return EMPTY_HEADING_TITLE
    # One Unicode-whitespace trim for display (#315), the same `str.strip()`
    # `_fallback_title` gives a plain line, so a heading titles alike on either
    # path. The regex only spends space/tab, so a trailing NBSP stayed in the
    # title (`Install\xa0`). It runs after the closer and all-`#` checks, which
    # follow CommonMark's space/tab rule: `# \xa0#` has no closer and titles `#`.
    # A title that was only Unicode whitespace (`# \xa0`) is an empty heading.
    return title.strip() or EMPTY_HEADING_TITLE


# A fenced code block opener: an optional indent, then a run of 3+ backticks or
# tildes, then an optional info string (` ```python `). CommonMark allows both
# fence characters, and a fence is closed only by a run of the SAME character at
# least as long as the opener — so a ```` block is not closed by an inner ```.
# `^ {0,3}`, not `^[ \t]*` (#156). The unbounded indent class accepted a fence
# opener indented 4+ spaces — which CommonMark defines as an indented CODE
# BLOCK, not a fence. Because nothing later in the document closes a fence that
# was never really opened, the phantom span ran to end-of-document and
# suppressed every heading inside it. Measured: a 3-section document containing
# one `    ``` ` line collapsed to a SINGLE chunk, destroying two real headings.
# Same 0-3 rule as `_HEADING_RE` above, so the two agree.
# Same line ends as `_HEADING_RE` (#297): were the heading scan to see CR-only
# lines while this one did not, a `# comment` inside a CR-only fence would open
# a section again, which is #152 through a different line ending.
# The opener may also sit on a list item's MARKER line (#305): `- ```bash`,
# `1. ```python`. CommonMark reads the fence as the item's first content block,
# and its closer sits at the item's content indent (`  ```` under `- `). Pre-fix
# the marker kept the opener from matching at all, so the indented CLOSER was
# read as an opener and the phantom span ran to the next bare fence or to
# end-of-text: every real heading in between vanished and the list block's
# `# comment` became a chunk title -- #152 and #156 at once. The marker is a
# bullet `[-*+]` or an ordered `\d{1,9}[.)]` (CommonMark §5.2's digit cap), then
# 1-4 spaces; five or more would make the content an indented code block, so
# the fence could not follow.
#
# The indent is captured unbounded and checked in `_fenced_spans`, not bounded
# here: a list fence's closer may legitimately sit past column 3 (`10. ````
# puts it at 4), while every other opener and closer keeps #156's 0-3 rule.
_FENCE_RE = re.compile(
    rf"{_LINE_START}(?P<indent> *)(?P<marker>(?:[-*+]|\d{{1,9}}[.)]) {{1,4}})?"
    rf"(?P<fence>`{{3,}}|~{{3,}})[ \t]*(?P<info>[^{_EOL}]*){_LINE_END}"
)

# CommonMark's fence indent allowance (#156): up to 3 spaces past the container.
_MAX_FENCE_INDENT = 3

# An HTML comment block opener (#308): `<!--` at the start of a line, after
# #156's 0-3 space indent. CommonMark §4.6 type 2 -- the block runs to the
# first line containing `-->` (which may be the opener's own line), or to
# end-of-text, and a `# line` inside it is raw HTML, not a heading. Pre-fix
# `<!--\n# draft heading\n-->` opened a section titled `draft heading`: text
# that is not a heading became a chunk boundary and a `title`, #152's harm.
# Same line ends as `_HEADING_RE` (#297), for the reason `_FENCE_RE` gives.
_HTML_COMMENT_RE = re.compile(rf"{_LINE_START} {{0,3}}<!--")
_HTML_COMMENT_CLOSE = "-->"


def _block_spans(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the `[start, end)` spans of fenced code blocks and of HTML
    comment blocks, as `(fences, comments)`.

    One position-ordered pass over both openers, because each block hides the
    other's opener (#308): a `<!--` line inside an open fence is code, and a
    ```` ``` ```` line inside an open comment is HTML. Two independent passes
    would get one of those wrong -- a fence line inside a comment, read as an
    opener, has no closer and swallows every later heading (#156's harm).
    """
    fences: list[tuple[int, int]] = []
    comments: list[tuple[int, int]] = []
    events = sorted(
        [(m.start(), True, m) for m in _FENCE_RE.finditer(text)]
        + [(m.start(), False, m) for m in _HTML_COMMENT_RE.finditer(text)],
        key=lambda e: e[0],
    )
    open_at: int | None = None
    open_char = ""
    open_len = 0
    close_indent = _MAX_FENCE_INDENT
    comment_end = 0
    for start, is_fence, m in events:
        # Inside an HTML comment, fence and comment openers alike are content.
        if start < comment_end:
            continue
        if not is_fence:
            # Inside an open fence a `<!--` line is code (#152's rule).
            if open_at is not None:
                continue
            # Search from the opener's own `--`: `<!-->` and `<!--->` contain
            # `-->` and close on the opener's line, as CommonMark's "line
            # contains `-->`" end condition reads them.
            close = text.find(_HTML_COMMENT_CLOSE, m.end() - 2)
            comment_end = len(text) if close == -1 else close + len(_HTML_COMMENT_CLOSE)
            comments.append((start, comment_end))
            continue
        fence = m.group("fence")
        indent = len(m.group("indent"))
        marker = m.group("marker")
        if open_at is None:
            # #156: 4+ spaces before an opener (or before its list marker) is
            # an indented code block, not a fence.
            if indent > _MAX_FENCE_INDENT:
                continue
            # CommonMark §4.5: a BACKTICK fence's info string may not contain a
            # backtick, so ```` ```x``` is inline ```` is a paragraph that opens
            # with a code span, not a fence (#223). Read as an opener, nothing
            # ever closed it and the phantom span ran to end-of-text, erasing
            # every later heading -- #156's harm through the info-string rule
            # instead of the indent rule. Tilde fences may carry any info string.
            if fence[0] == "`" and "`" in m.group("info"):
                continue
            open_at = m.start()
            open_char = fence[0]
            open_len = len(fence)
            # #305: a fence opened after a list marker closes at the item's
            # content indent (indent + marker + its spaces), plus the usual 3.
            close_indent = _MAX_FENCE_INDENT + (indent + len(marker) if marker else 0)
            continue
        # A closing fence must use the same character, be at least as long as
        # the opener, and carry no info string. It also carries no list marker
        # (`- ```` inside an open fence is content, #305) and sits no deeper
        # than the indent its opener allows (#156: 4+ past it is content).
        # "No info string" means only spaces or tabs may follow (§4.5), so the
        # trim is `strip(" \t")`: a bare `.strip()` also ate NBSP and U+3000,
        # and a ```` ```\xa0 ```` line closed the fence early, letting a `# line`
        # still inside it open a section (#315, #152's harm).
        if (
            marker is None
            and indent <= close_indent
            and fence[0] == open_char
            and len(fence) >= open_len
            and not m.group("info").strip(" \t")
        ):
            fences.append((open_at, m.end()))
            open_at = None
    if open_at is not None:
        fences.append((open_at, len(text)))
    return fences, comments


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Return the `[start, end)` character spans covered by fenced code blocks.

    `_HEADING_RE` is line-anchored and has no notion of code fences, so a `#`
    (or `###`) comment at the start of a line inside a ```` ``` ```` block used
    to be treated as an ATX heading (#152). That tore the fence in half across
    two chunks and — worse — promoted the comment text into the chunk's
    ``title`` metadata, the field the module docstring designates as a
    retrieval signal. This computes the fenced regions once so `chunk` can drop
    any heading match that starts inside one.

    A backtick run whose info string contains a backtick is not an opener
    (CommonMark §4.5, #223); inside an open fence such a line is content, and
    it cannot close the fence either, because a closer carries no info string.

    An unclosed fence extends to end-of-text. That is the conservative
    direction: the strategy under-splits rather than inventing headings out of
    code, which is the failure this fixes.

    A fence opened on a list item's marker line (`- ```bash`) is recognised,
    and closes at up to 3 spaces past that item's content indent (#305). Only
    the marker line is read: a fence nested in a list item indented 4+ spaces,
    and a list item that ends before its fence closes (a column-0 line ending
    the item), both need the block parser the next paragraph declines.

    Indented (four-space) code blocks are deliberately not handled. Telling
    them apart from list continuation needs a real block parser, and this
    package is dep-free by design (D-002) — so that is a much larger change
    than this bug justifies, not an oversight.

    A fence line inside an HTML comment is not an opener (#308); see
    `_block_spans`, which this reads.
    """
    return _block_spans(text)[0]


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
        check_chunk_input(text, source_doc_id)
        if not text:
            return []
        # Drop matches inside fenced code blocks: a `# comment` line in a
        # ```python block is not a heading (#152). Nor is a `# line` inside an
        # HTML comment block (#308).
        fences, comments = _block_spans(text)
        skipped = fences + comments
        # `_heading_title`: the content group is optional (#154), so an empty
        # ATX heading yields None, and a lone closing run (`# #`) is captured as
        # the title unless it is recognised as the closer it is (#242).
        headings = [
            (m.start(), m.end(), len(m.group(1)), _heading_title(m))
            for m in _HEADING_RE.finditer(text)
            if len(m.group(1)) <= self.max_heading_level and not _in_spans(m.start(), skipped)
        ]
        if not headings:
            # No surviving heading — fallback: title = first non-empty line
            # outside a fence or HTML comment (`_fallback_title`, #288, #308). Still honor
            # max_chunk_chars: a long unheaded document (plain text, Setext
            # `===`/`---` headings or RST that the ATX-only _HEADING_RE doesn't
            # match) must not bypass the ceiling and emit one monster chunk.
            # Mirrors the cap logic the heading-bounded path uses; a follow-up
            # can unify both through `_emit_capped` once #56's PR lands.
            base_meta = {
                "title": self._fallback_title(text, skipped, source_doc_id),
                "heading_level": None,
            }
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

    @staticmethod
    def _fallback_title(text: str, fences: list[tuple[int, int]], source_doc_id: str) -> str:
        """Title for the no-surviving-heading fallback: the first non-empty
        line outside a fenced block, de-marked if it is an ATX heading.

        "No surviving heading" is wider than "no heading" (#288). A document
        whose headings are all deeper than ``max_heading_level``, or whose only
        `#` lines sit inside a fence, also lands here — and the raw first line
        then titled the chunk ``### Install``, `` ```python `` or
        ``## Title ##``: the markup #152, #154 and #242 removed from the
        heading path. So skip fenced lines (#152's rule), and run a heading
        that was filtered only by level through ``_heading_title`` so it
        titles exactly as it would had it survived. ``fences`` also carries
        the HTML comment blocks (#308), whose lines are no title either. Plain text, Setext and RST
        first lines match neither and keep the raw stripped line, as before.
        """
        offset = 0
        for line in text.splitlines(keepends=True):
            line_start = offset
            offset += len(line)
            if not line.strip() or _in_spans(line_start, fences):
                continue
            m = _HEADING_RE.match(line.splitlines()[0])
            return _heading_title(m) if m else line.strip()
        # Blank or wholly fenced: no line can stand as a title.
        return source_doc_id

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
