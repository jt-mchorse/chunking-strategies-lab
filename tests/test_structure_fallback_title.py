"""The no-surviving-heading fallback must not title a chunk with markup (#288).

`StructureAwareStrategy` falls back to "title = first non-empty line" when no
heading *survives filtering* — not only when a document has no headings. The
raw line went straight into `title`, so a level-filtered heading, a fence
opener, or a heading's closing sequence became the retrieval signal: the exact
values #152, #154 and #242 removed from the heading path. Measured on `main`
6c40be3 (found by a hunt agent, re-run here):

    only ### headings, max_heading_level=2   -> ['### Install']
    only '#' lines are inside a fence        -> ['```python']
    closing sequence, max_heading_level=1    -> ['## Title ##']
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import EMPTY_HEADING_TITLE, StructureAwareStrategy


def _titles(text: str, max_heading_level: int = 6, max_chunk_chars: int = 4000) -> list[str]:
    s = StructureAwareStrategy(max_heading_level=max_heading_level, max_chunk_chars=max_chunk_chars)
    return [c.metadata["title"] for c in s.chunk(text, source_doc_id="d.md")]


# ----------------------------------------------------------------------
# Red on main: the three measured repros, plus their nearest variants
# ----------------------------------------------------------------------


def test_a_level_filtered_heading_titles_as_the_heading_path_would() -> None:
    assert _titles("### Install\n\nbody\n", max_heading_level=2) == ["Install"]


def test_a_closing_sequence_is_dropped_from_a_level_filtered_heading() -> None:
    assert _titles("## Title ##\n\nbody\n", max_heading_level=1) == ["Title"]


def test_an_empty_level_filtered_heading_is_the_empty_heading_sentinel() -> None:
    # #154/#242's contract, reached through the fallback instead.
    assert _titles("### ###\n\nbody\n", max_heading_level=2) == [EMPTY_HEADING_TITLE]


def test_a_fence_opener_is_not_a_title() -> None:
    doc = "```python\n# comment\nx = 1\n```\n\nPlain intro.\n"
    assert _titles(doc) == ["Plain intro."]


def test_a_fenced_first_line_falls_through_to_a_level_filtered_heading() -> None:
    doc = "~~~\n# not a heading\n~~~\n\n### Usage\n\nbody\n"
    assert _titles(doc, max_heading_level=2) == ["Usage"]


def test_a_wholly_fenced_document_takes_the_doc_id_default() -> None:
    # No line outside the fence can stand as a title, so the fallback keeps
    # the default it already used for a blank document.
    assert _titles("```python\n# comment\nx = 1\n```\n") == ["d.md"]


def test_every_capped_piece_carries_the_clean_title() -> None:
    doc = "### Install\n\n" + "word " * 200
    titles = _titles(doc, max_heading_level=2, max_chunk_chars=300)
    assert len(titles) > 1
    assert set(titles) == {"Install"}


# ----------------------------------------------------------------------
# Controls: green on main and after the fix
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("doc", "title"),
    [
        ("Plain title\n\nbody\n", "Plain title"),
        ("\n\n  Indented prose  \n\nbody\n", "Indented prose"),
        ("Setext Title\n============\n\nbody\n", "Setext Title"),
        ("#hashtag opener\n\nbody\n", "#hashtag opener"),  # not a heading
        ("####### seven\n\nbody\n", "####### seven"),  # 7 hashes: not a heading
        # 4-space indent is an indented code block, deliberately not handled
        # (see `_fenced_spans`): the raw stripped line, as before.
        ("    # code\n\nbody\n", "# code"),
        ("", None),
        ("   \n\n", "d.md"),
    ],
)
def test_titles_that_must_not_change(doc: str, title: str | None) -> None:
    assert _titles(doc) == ([] if title is None else [title])


def test_crlf_level_filtered_heading_carries_no_carriage_return() -> None:
    assert _titles("### Install\r\n\r\nbody\r\n", max_heading_level=2) == ["Install"]


def test_offsets_and_text_do_not_move() -> None:
    """Only `title` was ever wrong; the chunk bodies must be untouched."""
    doc = "```python\n# comment\n```\n\n### Install\n\nbody\n"
    chunks = StructureAwareStrategy(max_heading_level=2).chunk(doc)
    assert [(c.start_offset, c.end_offset, c.text) for c in chunks] == [(0, len(doc), doc)]
