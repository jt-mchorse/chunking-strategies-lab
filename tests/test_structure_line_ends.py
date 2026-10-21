"""The heading scan and the fallback must agree on what ends a line (#297).

`_HEADING_RE` was `re.MULTILINE`, which ends a line only at `\\n`, while
#288's `_fallback_title` finds lines with `str.splitlines()`, which also breaks
on `\\r`, `\\x0b`, `\\x0c`, `\\x1c`-`\\x1e`, `\\x85`, U+2028 and U+2029. The
heading's title group ran straight through all nine. Measured on `main`
8e10489 (found by a hunt agent, re-run here):

    '### Install\\x0cbody'                       -> ['Install\\x0cbody']
    '### Install\\x0cbody', max_heading_level=2  -> ['Install']
    '# A\\r\\rtext a\\r# B\\rtext b\\r'               -> 1 chunk, title = the whole text
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import StructureAwareStrategy

# Every separator `str.splitlines()` breaks on besides `\n` (and `\r\n`, which
# is the `\r` arm followed by a `\n`).
_SEPARATORS = ["\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]


def _titles(text: str, max_heading_level: int = 6) -> list[str]:
    s = StructureAwareStrategy(max_heading_level=max_heading_level)
    return [c.metadata["title"] for c in s.chunk(text, source_doc_id="d.md")]


def test_the_separator_list_is_exactly_what_splitlines_breaks_on() -> None:
    # Guards the parametrisation below: if the list drifted from the stdlib
    # set, a separator could go unpinned with every arm green.
    breaks = [chr(c) for c in range(0x110000) if len(("a" + chr(c) + "b").splitlines()) == 2]
    assert sorted(breaks) == sorted(["\n", *_SEPARATORS])


# ----------------------------------------------------------------------
# Red on main: one arm per separator, plus the CR-only document
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sep", _SEPARATORS, ids=lambda s: f"U+{ord(s):04X}")
def test_a_surviving_heading_stops_at_the_line_end(sep: str) -> None:
    assert _titles(f"### Install{sep}body") == ["Install"]


@pytest.mark.parametrize("sep", _SEPARATORS, ids=lambda s: f"U+{ord(s):04X}")
def test_a_heading_titles_the_same_whether_or_not_it_survives_the_filter(sep: str) -> None:
    # #288's promise: a level-filtered heading "titles exactly as it would had
    # it survived".
    doc = f"### Install{sep}body"
    assert _titles(doc) == _titles(doc, max_heading_level=2)


@pytest.mark.parametrize("sep", _SEPARATORS, ids=lambda s: f"U+{ord(s):04X}")
def test_a_heading_after_the_separator_opens_a_section(sep: str) -> None:
    doc = f"# A{sep}text a{sep}# B{sep}text b{sep}"
    chunks = StructureAwareStrategy().chunk(doc, source_doc_id="d.md")
    assert [c.metadata["title"] for c in chunks] == ["A", "B"]
    assert chunks[1].start_offset == doc.index("# B")
    assert "".join(c.text for c in chunks) == doc


def test_a_cr_only_document_splits_on_its_headings() -> None:
    doc = "# A\r\rtext a\r# B\rtext b\r"
    chunks = StructureAwareStrategy().chunk(doc, source_doc_id="d.md")
    assert [(c.metadata["title"], c.start_offset, c.end_offset) for c in chunks] == [
        ("A", 0, 12),
        ("B", 12, len(doc)),
    ]


def test_a_cr_only_fence_still_hides_its_comment_lines() -> None:
    # #152 through #297's door: the fence scan must see CR-only lines too, or
    # the heading scan would now open a section on the fenced `# comment`.
    doc = "# A\r```python\r# comment\rx = 1\r```\r# B\rb\r"
    assert _titles(doc) == ["A", "B"]


# ----------------------------------------------------------------------
# Controls: green on main and after the fix
# ----------------------------------------------------------------------


@pytest.mark.parametrize("eol", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_lf_and_crlf_documents_are_unchanged(eol: str) -> None:
    doc = f"# A{eol}text a{eol}# B{eol}text b{eol}"
    chunks = StructureAwareStrategy().chunk(doc, source_doc_id="d.md")
    assert [c.metadata["title"] for c in chunks] == ["A", "B"]
    assert chunks[1].start_offset == doc.index("# B")


@pytest.mark.parametrize("eol", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_lf_and_crlf_fences_still_hide_comment_lines(eol: str) -> None:
    doc = f"# A{eol}```python{eol}# comment{eol}```{eol}# B{eol}b{eol}"
    assert _titles(doc) == ["A", "B"]


def test_a_unit_separator_is_not_a_line_end() -> None:
    # `\x1f` sits beside `\x1c`-`\x1e` but `splitlines()` does not break on
    # it, so it stays inside the title on both paths.
    doc = "### Install\x1fbody"
    assert _titles(doc) == _titles(doc, max_heading_level=2) == ["Install\x1fbody"]
