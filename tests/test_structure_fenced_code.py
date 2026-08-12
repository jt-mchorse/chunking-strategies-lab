"""`StructureAwareStrategy` must not read `#` inside a code fence as a heading (#152).

`_HEADING_RE` is line-anchored with `re.MULTILINE` and had no notion of fenced
code blocks, so a `# comment` line inside a ```` ```python ```` block became an
ATX heading. That did three things: it split the section at the comment, it
tore the fence across two chunks (opener in one, closer in another), and it
promoted the comment text into the chunk's ``title`` metadata — the field the
module docstring designates as a retrieval signal.

The assertions here are anchored to those harms rather than to chunk counts
alone: the fence-integrity tests count fence markers *within a single emitted
chunk*, so a future change that happens to produce the right number of chunks
while still splitting a block fails.

Latent on the pinned corpus (no `data/corpus/*.md` file has a heading-shaped
line inside a fence), which is why `tests/test_summary_snapshot.py` and the
canonical fixtures are unaffected — see the byte-identity check in the PR.
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import StructureAwareStrategy, _fenced_spans

# Built with explicit newline joins so the fence markers can't be mangled by
# an editor or a formatter reflowing a triple-quoted literal.
BACKTICK = "`" * 3
TILDE = "~" * 3


def _doc(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _titles(chunks) -> list[str]:
    return [c.metadata["title"] for c in chunks]


def test_hash_comment_inside_a_fence_is_not_a_heading() -> None:
    text = _doc(
        "# Tuning Postgres",
        "",
        "Prose about pooling.",
        "",
        f"{BACKTICK}python",
        "# Set the pool size to match your worker count",
        "pool = ConnectionPool(size=8)",
        BACKTICK,
        "",
        "More prose in the SAME section.",
        "",
        "## Indexing",
        "",
        "Prose about indexes.",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="pg.md")

    assert _titles(chunks) == ["Tuning Postgres", "Indexing"], (
        f"a `#` comment inside a fence must not create a section; got {_titles(chunks)}"
    )
    # The comment must not appear as a title anywhere — this is the harm that
    # reaches retrieval, distinct from the boundary being wrong.
    assert not any("pool size" in t for t in _titles(chunks)), (
        f"a code comment leaked into chunk titles: {_titles(chunks)}"
    )


def test_triple_hash_comment_inside_a_fence_is_not_a_level_3_heading() -> None:
    """`###` is an ordinary Python comment, so the bug was never limited to
    single-`#` lines — it fabricated a `heading_level=3` section."""
    text = _doc(
        "# Doc",
        "",
        f"{BACKTICK}python",
        "### three hashes are a valid python comment",
        "x = 1",
        BACKTICK,
        "",
        "Trailing prose.",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert len(chunks) == 1, f"expected a single section; got {_titles(chunks)}"
    assert chunks[0].metadata["heading_level"] == 1
    assert "three hashes" not in str(_titles(chunks))


def test_the_fenced_block_survives_intact_inside_one_chunk() -> None:
    """Anchor on fence integrity, not chunk count.

    Pre-fix the opener landed in one chunk and the closer in another, so two
    chunks each carried an unbalanced fence. Counting markers *within a single
    chunk* catches that even if a future change happens to emit the right
    number of chunks.
    """
    text = _doc(
        "# Section",
        "",
        f"{BACKTICK}python",
        "# a comment",
        "y = 2",
        BACKTICK,
        "",
        "after",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert len(chunks) == 1
    assert chunks[0].text.count(BACKTICK) == 2, (
        f"the opening and closing fences must land in the same chunk; got {chunks[0].text!r}"
    )


def test_tilde_fences_are_honored() -> None:
    """CommonMark allows `~~~` as well as backticks."""
    text = _doc(
        "# Section",
        "",
        f"{TILDE}python",
        "# a comment",
        TILDE,
        "",
        "after",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert len(chunks) == 1, f"tilde fence not honored; got {_titles(chunks)}"


def test_a_longer_opener_is_not_closed_by_a_shorter_inner_run() -> None:
    """A ```` block legitimately contains ``` lines — that inner run must not
    close it, or the `#` after it escapes the fence and becomes a heading."""
    outer = "`" * 4
    text = _doc(
        "# Section",
        "",
        f"{outer}markdown",
        "Here is how you write a fence:",
        BACKTICK,
        "# not a heading, it is sample markdown",
        BACKTICK,
        outer,
        "",
        "after",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert len(chunks) == 1, f"a shorter inner run closed a longer fence; got {_titles(chunks)}"


def test_info_string_on_the_opener_does_not_prevent_closing() -> None:
    """The opener carries ` ```python `; the closer carries nothing. A closer
    is required to have no info string, and the opener must not be rejected
    for having one."""
    spans = _fenced_spans(_doc(f"{BACKTICK}python", "x = 1", BACKTICK))
    assert len(spans) == 1, f"expected exactly one fenced span; got {spans}"


def test_an_unclosed_fence_extends_to_end_of_text() -> None:
    """Conservative direction: under-split rather than invent headings from
    code. A truncated document must not resume treating `#` as a heading."""
    text = _doc(
        "# Section",
        "",
        f"{BACKTICK}python",
        "# a comment",
        "## another comment",
        "x = 1",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert len(chunks) == 1, f"unclosed fence leaked headings; got {_titles(chunks)}"
    assert _titles(chunks) == ["Section"]


def test_headings_outside_fences_still_split_normally() -> None:
    """The fix must not suppress real headings — including ones that follow a
    fenced block, which is where an off-by-one in the span math would show."""
    text = _doc(
        "# One",
        "",
        f"{BACKTICK}sh",
        "# not a heading",
        BACKTICK,
        "",
        "## Two",
        "",
        "prose",
        "",
        "### Three",
        "",
        "prose",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert _titles(chunks) == ["One", "Two", "Three"], (
        f"real headings must still split; got {_titles(chunks)}"
    )
    assert [c.metadata["heading_level"] for c in chunks] == [1, 2, 3]


def test_document_with_no_fences_is_unaffected() -> None:
    """Regression guard for the pinned corpus, which has no heading-shaped
    lines inside fences — the fix must be a no-op there."""
    text = _doc("# One", "", "prose", "", "## Two", "", "prose")

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert _titles(chunks) == ["One", "Two"]


def test_full_text_coverage_is_preserved_across_a_fence() -> None:
    """The strategy's standing invariant: concatenating chunk spans in order
    must reproduce the document. Merging two sections into one is only correct
    if no codepoints are dropped in the process."""
    text = _doc(
        "# Section",
        "",
        f"{BACKTICK}python",
        "# a comment",
        BACKTICK,
        "",
        "after",
        "",
        "## Next",
        "",
        "more",
    )

    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    rebuilt = "".join(text[c.start_offset : c.end_offset] for c in chunks)
    assert rebuilt == text, "chunk spans must tile the document without gaps"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("no fences here", 0),
        (f"{BACKTICK}\nx\n{BACKTICK}", 1),
        (f"{BACKTICK}\nx\n{BACKTICK}\ntext\n{BACKTICK}\ny\n{BACKTICK}", 2),
        (f"{BACKTICK}\nunclosed", 1),
    ],
)
def test_fenced_spans_counts(body: str, expected: int) -> None:
    assert len(_fenced_spans(body + "\n")) == expected
