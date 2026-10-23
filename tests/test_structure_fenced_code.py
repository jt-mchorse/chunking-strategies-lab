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


# (id, document lines, expected section titles). A backtick fence's info string
# may not contain a backtick (CommonMark §4.5, #223); a tilde fence's may. The
# expected titles are the headings CommonMark's reference parser finds
# (markdown-it-py, commonmark preset), checked when this table was written.
_INFO_STRING_CASES = [
    ("control", ["# A", "alpha", "", "# B", "beta", "", "# C", "gamma"], ["A", "B", "C"]),
    (
        "inline-code-span-line",
        [
            "# A",
            BACKTICK + "x" + BACKTICK + " is inline code",
            "",
            "# B",
            "beta",
            "",
            "# C",
            "gamma",
        ],
        ["A", "B", "C"],
    ),
    (
        "backtick-in-info",
        ["# A", BACKTICK + "py`x", "", "# B", "beta", "", "# C", "gamma"],
        ["A", "B", "C"],
    ),
    (
        "longer-run-backtick-in-info",
        ["# A", "````a`b", "", "# B", "beta", "", "# C", "gamma"],
        ["A", "B", "C"],
    ),
    (
        "info-without-backtick-still-opens",
        ["# A", BACKTICK + "python", "# comment", BACKTICK, "# B", "beta"],
        ["A", "B"],
    ),
    (
        "code-span-line-inside-an-open-fence-is-content",
        [
            "# A",
            BACKTICK + "python",
            BACKTICK + "x" + BACKTICK,
            "# comment",
            BACKTICK,
            "# B",
            "beta",
        ],
        ["A", "B"],
    ),
    (
        "tilde-info-may-contain-a-backtick",
        ["# A", TILDE + "py`x", "# not a heading", TILDE, "# B", "beta"],
        ["A", "B"],
    ),
]


@pytest.mark.parametrize(
    ("lines", "titles"),
    [(lines, titles) for _, lines, titles in _INFO_STRING_CASES],
    ids=[case_id for case_id, _, _ in _INFO_STRING_CASES],
)
def test_a_backtick_in_a_backtick_info_string_is_not_a_fence(
    lines: list[str], titles: list[str]
) -> None:
    assert _titles(StructureAwareStrategy().chunk(_doc(*lines))) == titles


# --- #305: a fence opened on a list item's marker line ----------------------
#
# `- ```bash` is ordinary CommonMark: the fence is the item's first block, and
# its closer sits at the item's content indent. Pre-fix the marker hid the
# opener, the indented closer was read as an OPENER, and the phantom span ran
# to the next bare fence or to end-of-text. Expected titles are CommonMark's
# (markdown-it-py, commonmark preset), checked when this table was written.


def test_the_issue_305_readme_keeps_its_four_headings() -> None:
    text = (
        "# Install\n\n1. Fetch the deps:\n   ```bash\n   # comment A\n   ```\n\n"
        "- ```bash\n  # comment B\n  make\n  ```\n\n## Usage\n\nRun it.\n\n"
        "## Configure\n\n```toml\n# comment C\nkey = 1\n```\n\n## FAQ\n\nNone yet.\n"
    )

    titles = _titles(StructureAwareStrategy().chunk(text, source_doc_id="r.md"))

    assert titles == ["Install", "Usage", "Configure", "FAQ"], titles
    assert not any("comment" in t for t in titles), f"a code comment became a title: {titles}"


# (marker, the content indent its closer sits at).
_LIST_MARKERS = [
    ("- ", 2),
    ("* ", 2),
    ("+ ", 2),
    ("1. ", 3),
    ("1) ", 3),
]
# Markers whose closer sits past column 3. The unfixed code got these right by
# accident -- it matched neither the opener nor the 4+-space closer -- so they
# are not repros. They pin the closer half of the fix: recognise the marker
# but keep a 0-3 closer allowance, and the fence never closes and runs to
# end-of-text, erasing `# B` and `# C`.
_DEEP_LIST_MARKERS = [("10. ", 4), ("-    ", 5)]


def _list_fence_doc(marker: str, indent: int, fence: str) -> str:
    pad = " " * indent
    return _doc(
        "# A",
        "",
        f"{marker}{fence}bash",
        f"{pad}# comment in the list's code block",
        f"{pad}{fence}",
        "",
        "# B",
        "beta",
        "",
        "# C",
        "gamma",
    )


@pytest.mark.parametrize("fence", [BACKTICK, TILDE], ids=["backtick", "tilde"])
@pytest.mark.parametrize(
    ("marker", "indent"), _LIST_MARKERS, ids=[m.strip() + f"-{i}" for m, i in _LIST_MARKERS]
)
def test_a_fence_on_a_list_marker_line_is_a_fence(marker: str, indent: int, fence: str) -> None:
    text = _list_fence_doc(marker, indent, fence)
    assert _titles(StructureAwareStrategy().chunk(text)) == ["A", "B", "C"]


@pytest.mark.parametrize("fence", [BACKTICK, TILDE], ids=["backtick", "tilde"])
@pytest.mark.parametrize(
    ("marker", "indent"),
    _DEEP_LIST_MARKERS,
    ids=[m.strip() + f"-{i}" for m, i in _DEEP_LIST_MARKERS],
)
def test_a_list_fence_closes_at_its_content_indent(marker: str, indent: int, fence: str) -> None:
    text = _list_fence_doc(marker, indent, fence)
    assert _titles(StructureAwareStrategy().chunk(text)) == ["A", "B", "C"]


# Controls: shapes that were right before #305 and must stay right.
_LIST_CONTROL_CASES = [
    (
        # The issue's own control: the fence on its own line under the item.
        "fence-on-its-own-line-under-an-item",
        ["# A", "1. Fetch:", f"   {BACKTICK}bash", "   # comment", f"   {BACKTICK}", "# B", "b"],
        ["A", "B"],
    ),
    (
        # A list-marker fence line inside an open fence is content: a closer
        # carries no marker, so the block still ends at the bare closer.
        "marker-fence-line-inside-an-open-fence-is-content",
        ["# A", BACKTICK, f"- {BACKTICK}", "# comment", BACKTICK, "# B", "b"],
        ["A", "B"],
    ),
    (
        # Five spaces after the marker make the item's content an indented
        # code block, so the backticks are code, not a fence opener.
        "five-spaces-after-the-marker-is-not-a-fence",
        ["# A", f"-     {BACKTICK}", "", "# B", "b", "", "# C", "c"],
        ["A", "B", "C"],
    ),
    (
        # #156's rule still applies before a marker: 4+ spaces is indented code.
        "marker-indented-four-spaces-is-not-a-fence",
        ["# A", "", f"    - {BACKTICK}", "", "# B", "b", "", "# C", "c"],
        ["A", "B", "C"],
    ),
    (
        # A top-level fence keeps the 0-3 closer allowance: a 4-space closer
        # is content, and the bare closer after it ends the block.
        "top-level-fence-closer-indented-four-is-content",
        ["# A", BACKTICK, "# comment", f"    {BACKTICK}", "# also code", BACKTICK, "# B", "b"],
        ["A", "B"],
    ),
]


@pytest.mark.parametrize(
    ("lines", "titles"),
    [(lines, titles) for _, lines, titles in _LIST_CONTROL_CASES],
    ids=[case_id for case_id, _, _ in _LIST_CONTROL_CASES],
)
def test_list_marker_fence_controls(lines: list[str], titles: list[str]) -> None:
    assert _titles(StructureAwareStrategy().chunk(_doc(*lines))) == titles
