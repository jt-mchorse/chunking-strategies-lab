"""`_HEADING_RE` and `_FENCE_RE` must agree on CommonMark's indent rule (#156).

The two patterns sit six lines apart in `structure.py` and disagreed about
leading indentation — and both disagreed with CommonMark, in opposite
directions:

- `_HEADING_RE` anchored the hashes hard to column 0, so a heading indented by
  even one space was not a heading at all. Since that regex is the only thing
  that opens a section, the boundary vanished and the following section merged
  into its predecessor carrying the wrong title.
- `_FENCE_RE` accepted `^[ \t]*`, so a fence opener indented 4+ spaces — an
  indented *code block* per CommonMark, not a fence — opened a span anyway.
  Nothing closes a fence that was never really opened, so the phantom span ran
  to end-of-document and suppressed every heading inside it.

Both produce the same user-visible symptom: section boundaries silently
disappear and separate sections merge into one chunk, with exit 0 and no
diagnostic. That is precisely what this strategy exists to prevent.

Two smaller defects in the same `_HEADING_RE` polluted `title`, which the
module designates a retrieval signal: an ATX closing sequence (`## Title ##`
titled as `Title ##`) and, under CRLF, a trailing carriage return.

Assertions are anchored to the measured pre-fix chunk *counts* rather than to
post-fix correctness alone, so a later loosening of either indent class has to
bring a merged chunk back rather than merely stop raising.
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import (
    EMPTY_HEADING_TITLE,
    StructureAwareStrategy,
    _fenced_spans,
)

# ----------------------------------------------------------------------
# _HEADING_RE and _FENCE_RE agree on CommonMark's 0-3 space indent (#156)
#
# The two regexes sat six lines apart and disagreed about leading
# indentation, and both disagreed with CommonMark in opposite directions:
#
#   _HEADING_RE  ^(#{1,6})   zero indent allowed   -> real headings MISSED
#   _FENCE_RE    ^[ \t]*     any indent allowed    -> phantom fences OPENED
#
# Both produce the same user-visible symptom — section boundaries silently
# disappear and separate sections merge into one chunk — which is exactly what
# this strategy exists to prevent. Every assertion below is anchored to the
# measured pre-fix chunk COUNT, not merely to post-fix correctness, so a later
# loosening of either indent class has to bring a merged chunk back.
# ----------------------------------------------------------------------


_TWO_SECTIONS = "# Alpha\n\nalpha body\n\n{indent}# Beta\n\nbeta body\n"


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_heading_indented_0_to_3_spaces_still_opens_a_section(indent: str) -> None:
    # Pre-fix: ANY non-empty indent gave 1 chunk titled ['Alpha'] — the Beta
    # section was swallowed into Alpha's chunk and carried Alpha's title.
    chunks = StructureAwareStrategy().chunk(_TWO_SECTIONS.format(indent=indent))
    assert len(chunks) == 2, f"indent={indent!r} merged the two sections"
    assert [c.metadata["title"] for c in chunks] == ["Alpha", "Beta"]


@pytest.mark.parametrize("indent", ["    ", "     ", "\t"])
def test_heading_indented_4_or_more_is_a_code_block_not_a_heading(indent: str) -> None:
    # The other side of the same rule: 4+ spaces is an indented code block, so
    # the `#` is content. This was already correct pre-fix, but only by
    # accident (the regex allowed no indent at all), so it needs pinning now
    # that an indent allowance exists. Tabs are deliberately excluded from the
    # allowance — CommonMark's tab-expansion rules are a separate problem.
    chunks = StructureAwareStrategy().chunk(_TWO_SECTIONS.format(indent=indent))
    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == "Alpha"


def test_indented_fence_opener_does_not_swallow_the_rest_of_the_document() -> None:
    # The severest of the four. A fence opener indented 4+ spaces is an
    # indented CODE BLOCK, not a fence — but `^[ \t]*` opened a span anyway,
    # and since nothing closes a fence that was never really opened, the span
    # ran to end-of-document.
    #
    # Measured pre-fix: spans=[(21, 69)], 1 chunk, titles=['Alpha'].
    # Two real headings destroyed by one indented backtick run.
    doc = "# Alpha\n\nalpha body\n\n    ```\n\n# Beta\n\nbeta body\n\n# Gamma\n\ngamma body\n"
    assert _fenced_spans(doc) == []
    chunks = StructureAwareStrategy().chunk(doc)
    assert len(chunks) == 3
    assert [c.metadata["title"] for c in chunks] == ["Alpha", "Beta", "Gamma"]


@pytest.mark.parametrize("indent", ["", " ", "   "])
def test_fence_indented_0_to_3_still_suppresses_headings_inside_it(indent: str) -> None:
    # #152 must not regress: a `#` inside a properly-indented fence is a
    # comment, not a heading. Tightening the indent class must narrow which
    # lines OPEN a fence, not weaken what an open fence suppresses.
    doc = f"# Real\n\n{indent}```python\n# not a heading\nprint(1)\n{indent}```\n\nbody\n"
    chunks = StructureAwareStrategy().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == "Real"


def test_atx_closing_sequence_is_not_part_of_the_title() -> None:
    # CommonMark 4.2: the optional trailing run of `#`s "is not considered part
    # of the heading's content". Pre-fix the lazy `(\S.*?)` swallowed it:
    # `# Title #` -> 'Title #' and `## Title ##` -> 'Title ##'.
    s = StructureAwareStrategy()
    assert s.chunk("# Title #\n\nbody\n")[0].metadata["title"] == "Title"
    assert s.chunk("## Title ##\n\nbody\n")[0].metadata["title"] == "Title"
    assert s.chunk("### Title ######\n\nbody\n")[0].metadata["title"] == "Title"
    # A `#` that is part of the text, not a closing sequence, must survive.
    assert s.chunk("# C# and F#\n\nbody\n")[0].metadata["title"] == "C# and F#"
    # A hashes-only line is still the empty heading from #154, not a title of
    # "#" — the closing-sequence branch must not cannibalize that contract.
    assert s.chunk("#\n\nbody\n")[0].metadata["title"] == EMPTY_HEADING_TITLE


def test_crlf_document_titles_carry_no_carriage_return() -> None:
    # `$` under re.MULTILINE matches before the `\n`, and `\r` is not in
    # `[ \t]`, so the lazy `(\S.*?)` absorbed it: every heading in a CRLF
    # document titled as 'Title\r'. `title` is designated a retrieval signal
    # twice in this module's comments.
    s = StructureAwareStrategy()
    lf = "# Alpha\n\nalpha body\n\n# Beta\n\nbeta body\n"
    crlf = lf.replace("\n", "\r\n")
    assert [c.metadata["title"] for c in s.chunk(crlf)] == ["Alpha", "Beta"]
    # And a CRLF document chunks into the same sections as its LF twin.
    assert len(s.chunk(crlf)) == len(s.chunk(lf))


def test_crlf_does_not_regress_the_fence_guard() -> None:
    # The *suppression* half was already correct pre-fix and is pinned here so
    # the regex change doesn't quietly break it: a `#` inside a CRLF fenced
    # block never became a heading, and `_fenced_spans` shifted its offsets
    # but stayed correct. The *title* half was not — pre-fix this chunk titled
    # as 'Real\r' — so this test does go red on the unfixed tree.
    doc = "# Real\n\n```python\n# not a heading\nprint(1)\n```\n\nbody\n".replace("\n", "\r\n")
    chunks = StructureAwareStrategy().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == "Real"


def test_non_headings_are_still_not_headings() -> None:
    # The indent allowance must not accidentally widen what counts as a
    # heading. Both of these were correct pre-fix and stay correct.
    s = StructureAwareStrategy()
    # No space after the hashes -> not a heading (`#hashtag`).
    assert len(s.chunk("# Alpha\n\nbody\n\n#Beta\n\nmore\n")) == 1
    # 7+ hashes -> not a heading.
    assert len(s.chunk("# Alpha\n\nbody\n\n####### Beta\n\nmore\n")) == 1
