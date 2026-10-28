"""An HTML block may open on a list item's marker line (#319).

#305 taught `_FENCE_RE` that a fence can open after a list marker
(`- ```bash`). `_HTML_BLOCK_RE` (#308, #317) allowed only a 0-3 space indent,
so `- <!--` and `* <pre>` never opened a block:

  - a `# line` inside the item's comment titled a chunk (#152's harm), and
  - a fence line inside the item's comment was read as an opener that nothing
    closed, so every later heading vanished (#156's harm).

Both now share one `_LIST_MARKER`. Expected titles below match markdown-it-py's
commonmark preset (checked in a side venv; D-002 keeps it out of the deps).
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import StructureAwareStrategy

FENCE = "`" * 3


def _doc(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _titles(text: str) -> list[str]:
    return [c.metadata["title"] for c in StructureAwareStrategy().chunk(text)]


def test_a_heading_line_inside_a_list_item_comment_is_not_a_heading() -> None:
    doc = _doc(
        "# Install",
        "",
        "- Run the installer.",
        "- <!-- TODO before 1.0",
        "  # Uninstall steps go here",
        "  -->",
        "",
        "# Usage",
        "body",
    )
    assert _titles(doc) == ["Install", "Usage"]


def test_a_fence_line_inside_a_list_item_comment_does_not_swallow_later_headings() -> None:
    doc = _doc(
        "# Install",
        "",
        "- Run the installer.",
        "- <!-- disabled until #42 ships",
        f"  {FENCE}bash",
        "  make deploy",
        "  -->",
        "",
        "# Usage",
        "use it",
        "",
        "# FAQ",
        "faq",
    )
    assert _titles(doc) == ["Install", "Usage", "FAQ"]


# Every type-1..5 opener, with its end string, behind every marker shape.
_BLOCKS = [
    ("t1-pre", "<pre>", "</pre>"),
    ("t1-script", "<script>", "</script>"),
    ("t2-comment", "<!--", "-->"),
    ("t3-pi", "<?php", "?>"),
    ("t4-declaration", "<!DOCTYPE html", ">"),
    ("t5-cdata", "<![CDATA[", "]]>"),
]
_MARKERS = ["- ", "* ", "+ ", "1. ", "2) ", "10. ", "-    ", " - ", "   1. "]


@pytest.mark.parametrize("marker", _MARKERS, ids=repr)
@pytest.mark.parametrize(("kind", "opener", "closer"), _BLOCKS, ids=[b[0] for b in _BLOCKS])
def test_every_html_type_opens_after_every_marker(marker, kind, opener, closer) -> None:
    pad = " " * len(marker)
    doc = _doc(
        "# Top",
        "",
        f"{marker}{opener}",
        f"{pad}# not a heading",
        f"{pad}{closer}",
        "",
        "# Next",
        "x",
    )
    assert _titles(doc) == ["Top", "Next"], kind


@pytest.mark.parametrize(
    "marker",
    [
        # Five spaces after the marker: the content is an indented code block,
        # so `<!--` opens nothing and the `#` line below is a real heading.
        "-     ",
        # No space: not a list item at all, just a paragraph starting `-<!--`.
        "-",
        # 4 spaces before the marker: an indented code block line.
        "    - ",
    ],
    ids=repr,
)
def test_a_marker_that_is_not_a_list_item_opens_no_block(marker: str) -> None:
    doc = _doc("# Top", "", f"{marker}<!--", "# Real heading", "-->", "", "# Next", "x")
    assert "Real heading" in _titles(doc)


def test_no_marker_cases_are_unchanged() -> None:
    assert _titles(_doc("# Install", "", "<!--", "# draft note", "-->", "", "# Usage", "body")) == [
        "Install",
        "Usage",
    ]
    assert _titles(_doc("# A", "", "- item", "", "# B", "b")) == ["A", "B"]


def test_offsets_still_reconstruct_the_source() -> None:
    doc = _doc("# A", "", "- <!--", "  # x", "  -->", "", "# B", "b")
    for c in StructureAwareStrategy().chunk(doc):
        assert doc[c.start_offset : c.end_offset] == c.text
