"""A lone ATX closing sequence is not a heading's title (#242).

The lazy title group took a closing run of `#`s as the title when nothing came
before it. Measured on `main` (found by a hunt agent, re-run here):
`StructureAwareStrategy().chunk("# #\\nbody\\n")` titled `'#'` and
`"### ###\\nbody\\n"` titled `'###'`, where CommonMark example 79 renders an
empty heading. `tests/test_structure_strategy.py` locked only the bare `#` case.
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import EMPTY_HEADING_TITLE, StructureAwareStrategy


def _title(text: str) -> str:
    return StructureAwareStrategy().chunk(text)[0].metadata["title"]


@pytest.mark.parametrize("line", ["# #", "### ###", "# ## ", "###### #", "#\t##"])
def test_a_lone_closing_sequence_is_an_empty_heading(line: str) -> None:
    assert _title(f"{line}\nbody\n") == EMPTY_HEADING_TITLE


@pytest.mark.parametrize(
    ("line", "title"),
    [
        ("### ### ###", "###"),  # CommonMark: a title of `###` plus its own closer
        ("# Title #", "Title"),
        ("# Title", "Title"),
        ("# C# #", "C#"),
        ("## a##", "a##"),  # no space before the run: content, not a closer
        ("# #hash", "#hash"),
        ("#", EMPTY_HEADING_TITLE),
    ],
)
def test_titles_that_must_not_change(line: str, title: str) -> None:
    assert _title(f"{line}\nbody\n") == title
