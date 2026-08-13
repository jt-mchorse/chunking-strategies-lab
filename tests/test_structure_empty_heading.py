"""An empty ATX heading must not steal a later line as its title (#154).

`_HEADING_RE` separated the hashes from the title with `\\s+`, and `\\s`
matches newlines. A `#` line with nothing after it — a valid empty heading in
CommonMark — therefore did not fail the match; it reached forward across the
blank line and captured a following paragraph as the match's title. `title` is
the field the module docstring designates as a retrieval signal, so that is the
same corruption #152 fixed, in the same regex, by a route the fence guard
cannot see.
"""

from __future__ import annotations

from chunking_lab.strategies.structure import EMPTY_HEADING_TITLE, StructureAwareStrategy


def _titles(text: str) -> list[tuple[str, int | None]]:
    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")
    return [(c.metadata["title"], c.metadata["heading_level"]) for c in chunks]


# ---------------------------------------------------------------------------
# The three confirmed theft variants
# ---------------------------------------------------------------------------


def test_bare_hash_does_not_steal_the_following_paragraph():
    text = "# Real Heading\n\nIntro.\n\n#\n\nShould NOT be a title.\n\n## Second\n\nbody\n"

    assert _titles(text) == [
        ("Real Heading", 1),
        (EMPTY_HEADING_TITLE, 1),
        ("Second", 2),
    ]


def test_bare_hash_does_not_steal_a_code_fence_opener():
    """The #152 fence guard cannot catch this one.

    The match *starts* at the bare `#`, which is outside every fence, so
    `_in_spans(m.start(), fences)` is False — only the text it used to steal
    came from inside the fence.
    """
    text = "# Top\n\ntext\n\n#\n\n```python\nx = 1\n```\n\n## Next\n\nbody\n"

    titles = _titles(text)
    assert titles == [("Top", 1), (EMPTY_HEADING_TITLE, 1), ("Next", 2)]
    assert "```" not in titles[1][0]


def test_hashes_followed_by_only_whitespace_are_empty_at_every_level():
    """Trailing spaces or a tab do not make a heading non-empty.

    `.` matches a space, so a naive `(.+?)` content group still captured a
    lone space here and produced a whitespace-only retrieval signal — the same
    useless title in a quieter form.
    """
    spaces = "# Top\n\n###   \n\nStolen text.\n\n## Next\n\nb\n"
    tab = "# Top\n\n#\t\n\nStolen text.\n\n## Next\n\nb\n"

    assert _titles(spaces) == [("Top", 1), (EMPTY_HEADING_TITLE, 3), ("Next", 2)]
    assert _titles(tab) == [("Top", 1), (EMPTY_HEADING_TITLE, 1), ("Next", 2)]


def test_empty_heading_at_end_of_document_opens_a_section():
    """CommonMark reads a trailing `#` as a heading; the boundary is correct.

    Pre-#154 this matched nothing at all (`(.+?)` had nothing to consume), so
    it created no boundary. Now it does, with no title to steal.
    """
    assert _titles("# A\n\nalpha\n\n#\n") == [("A", 1), (EMPTY_HEADING_TITLE, 1)]


# ---------------------------------------------------------------------------
# Locks on what must NOT change
# ---------------------------------------------------------------------------


def test_ordinary_headings_are_untouched():
    text = "# A\n\nalpha\n\n## B\n\nbeta\n\n### C\n\ngamma\n"

    assert _titles(text) == [("A", 1), ("B", 2), ("C", 3)]


def test_trailing_spaces_are_still_trimmed_from_a_real_title():
    assert _titles("# Padded Title   \n\nbody\n") == [("Padded Title", 1)]


def test_hash_without_a_space_is_still_not_a_heading():
    """`#hashtag` has no separator, so it never took the heading branch."""
    assert _titles("# A\n\n#hashtag is not a heading\n\nmore\n") == [("A", 1)]


def test_seven_hashes_is_still_not_a_heading():
    """`#{1,6}` leaves a seventh `#`, which satisfies neither branch."""
    assert _titles("# A\n\n####### seven\n\nmore\n") == [("A", 1)]


def test_hash_inside_a_fenced_block_is_still_not_a_heading():
    """Regression lock on #152, which shares this regex."""
    text = "# A\n\n```python\n# a comment\nx = 1\n```\n\n## B\n\nb\n"

    assert _titles(text) == [("A", 1), ("B", 2)]


def test_section_boundaries_are_byte_identical_for_a_heading_normal_document():
    """Only `title` metadata was ever wrong; offsets must not move."""
    text = "# A\n\nalpha\n\n## B\n\nbeta\n"
    chunks = StructureAwareStrategy().chunk(text, source_doc_id="d.md")

    assert [(c.start_offset, c.end_offset) for c in chunks] == [(0, 12), (12, len(text))]
    assert "".join(c.text for c in chunks) == text


def test_shipped_corpus_has_no_empty_headings():
    """Documents the blast radius: no committed artifact drifts from this fix."""
    from chunking_lab.corpus import load_corpus

    offenders = [
        doc.filename
        for doc in load_corpus()
        for line in doc.text.splitlines()
        if line.strip() and set(line.strip()) == {"#"}
    ]
    assert offenders == []
