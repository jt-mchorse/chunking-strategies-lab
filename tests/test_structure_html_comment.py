"""`StructureAwareStrategy` must not read `#` inside an HTML comment as a heading (#308).

CommonMark §4.6 type 2: a line starting (after 0-3 spaces) with `<!--` opens
an HTML block that runs to the first line containing `-->`, or to end-of-text.
A `# line` inside it is raw HTML. Pre-fix `_HEADING_RE` had no notion of the
block, so `<!--\\n# draft heading\\n-->` opened a section titled
`draft heading` -- text that is not a heading became a chunk boundary and a
`title`, the retrieval signal #152 protects.

Each block hides the other's opener, so the order cases matter: a `<!--` in an
open fence is code, and a fence line in an open comment is HTML (read as an
opener, it has no closer and erases every later heading).

Latent on the pinned corpus (no `data/corpus/*.md` file contains `<!--`), so
the canonical fixtures regenerate byte-identically.
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import StructureAwareStrategy, _fenced_spans

# Built with explicit newline joins, as in test_structure_fenced_code.py.
BACKTICK = "`" * 3


def _doc(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _titles(text: str) -> list[str]:
    return [c.metadata["title"] for c in StructureAwareStrategy().chunk(text)]


def test_the_issue_308_repro() -> None:
    doc = "# A\n\n<!--\n# draft heading\n-->\n\n# B\nb\n"
    assert _titles(doc) == ["A", "B"]


# Each: (id, lines, expected titles). Red against the pre-#308 tree.
_FIXED_CASES = [
    (
        # `-->` on the opener's own line ends the block there; the `#` line
        # after it is a real heading, the one inside a later block is not.
        "inline-comment-then-multiline-comment",
        ["# A", "<!-- x -->", "# B", "<!-- note", "# hidden", "-->", "# C", "c"],
        ["A", "B", "C"],
    ),
    (
        # Indent 0-3 is still an opener (#156's allowance).
        "opener-indented-three-spaces",
        ["# A", "   <!--", "# hidden", "-->", "# B", "b"],
        ["A", "B"],
    ),
    (
        # `-->` is anywhere on a line, not only at its start.
        "closer-mid-line",
        ["# A", "<!-- start", "# hidden", "end --> trailing", "# B", "b"],
        ["A", "B"],
    ),
    (
        # No `-->` at all: the block runs to end-of-text (the conservative
        # direction `_fenced_spans` also takes for an unclosed fence).
        "unclosed-comment-runs-to-end-of-text",
        ["# A", "a", "<!--", "# hidden", "## also hidden"],
        ["A"],
    ),
    (
        # A fence line inside a comment is HTML, not an opener. Read as one,
        # nothing closed it and `# B` vanished into a phantom fence.
        "fence-line-inside-a-comment-is-not-an-opener",
        ["# A", "<!--", BACKTICK, "-->", "# B", "b"],
        ["A", "B"],
    ),
]


@pytest.mark.parametrize(
    ("lines", "titles"),
    [(lines, titles) for _, lines, titles in _FIXED_CASES],
    ids=[case_id for case_id, _, _ in _FIXED_CASES],
)
def test_hash_line_inside_an_html_comment_is_not_a_heading(
    lines: list[str], titles: list[str]
) -> None:
    assert _titles(_doc(*lines)) == titles


def test_fallback_title_skips_a_comment() -> None:
    # No surviving heading: the first line outside any skipped block titles
    # the chunk (#288), and a comment line is no more a title than a fence.
    doc = _doc("<!-- generated: do not edit -->", "Real title", "body")
    assert _titles(doc) == ["Real title"]


def test_fallback_title_skips_a_comment_hiding_the_only_heading() -> None:
    doc = _doc("<!--", "# draft", "-->", "Body text")
    assert _titles(doc) == ["Body text"]


def test_fence_line_inside_a_comment_opens_no_fence_span() -> None:
    assert _fenced_spans(_doc("<!--", BACKTICK, "-->", "# B")) == []


# Controls: right before #308 and must stay right.
_CONTROL_CASES = [
    (
        # A `<!--` inside an open fence is code: the fence closes at its own
        # closer, and the `-->` far below must not be what ends anything.
        "comment-opened-inside-a-fence-is-code",
        ["# A", BACKTICK, "<!--", "# code", BACKTICK, "# B", "b", "-->", "# C", "c"],
        ["A", "B", "C"],
    ),
    (
        # 4+ spaces is an indented code block (#156), not an HTML block.
        "opener-indented-four-spaces-is-not-a-comment",
        ["# A", "", "    <!--", "", "# B", "b", "-->", "# C", "c"],
        ["A", "B", "C"],
    ),
    (
        # `<!--` mid-line is inline HTML in a paragraph, not a block opener.
        "inline-opener-mid-line-is-not-a-block",
        ["# A", "text <!-- aside", "# B", "b -->", "# C", "c"],
        ["A", "B", "C"],
    ),
    (
        # A closed one-line comment hides nothing after it.
        "one-line-comment-hides-nothing",
        ["# A", "<!-- x -->", "## B", "b"],
        ["A", "B"],
    ),
    (
        "no-comment-at-all",
        ["# A", "a", "# B", "b"],
        ["A", "B"],
    ),
]


@pytest.mark.parametrize(
    ("lines", "titles"),
    [(lines, titles) for _, lines, titles in _CONTROL_CASES],
    ids=[case_id for case_id, _, _ in _CONTROL_CASES],
)
def test_html_comment_controls(lines: list[str], titles: list[str]) -> None:
    assert _titles(_doc(*lines)) == titles


@pytest.mark.parametrize("opener", ["<!-->", "<!--->"], ids=["empty", "dash"])
def test_degenerate_opener_closes_on_its_own_line(opener: str) -> None:
    # CommonMark's end condition is "the line contains `-->`", and both of
    # these do -- the closer may overlap the opener's own `--`. A closer
    # search that started past the opener would run this block to
    # end-of-text and hide `# B`.
    assert _titles(_doc("# A", opener, "# B", "b")) == ["A", "B"]


def test_comment_spans_keep_full_coverage() -> None:
    # The skipped block stays inside its section's chunk: no codepoint lost.
    doc = "# A\n\n<!--\n# draft heading\n-->\n\n# B\nb\n"
    chunks = StructureAwareStrategy().chunk(doc)
    assert "".join(c.text for c in chunks) == doc
    assert "<!--\n# draft heading\n-->" in chunks[0].text
