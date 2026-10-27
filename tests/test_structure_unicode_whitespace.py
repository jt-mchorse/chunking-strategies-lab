"""Only space and tab are whitespace to `StructureAwareStrategy`'s ATX and fence rules (#315).

CommonMark allows a space or tab after the opening `#`s, before a closing
sequence, and after a closing fence. Python's `\\S` and `str.strip()` also
count NBSP, U+2003, U+202F, U+3000 and the rest of Unicode's whitespace, so
pre-fix:

(a) `# \\xa0Install` matched no heading. The title group's `\\S` refused the
    NBSP, so the section merged under the previous title (#156's harm).
(b) `# Install\\xa0` kept the NBSP in its title, while `_fallback_title`'s
    `line.strip()` dropped it.
(c) A closing fence followed by an NBSP closed the block (`.strip()` read the
    info string as empty), so a `# line` still inside the fence opened a
    section (#152's harm).

The realistic triggers are text pasted from the web or Word (`&nbsp;`) and
macOS Option-Space. Every expected title list is what CommonMark's reference
parser finds (markdown-it-py 4.2.0, commonmark preset; an empty heading is
`EMPTY_HEADING_TITLE`), checked when this table was written. Latent on the
pinned corpus, so the canonical fixtures don't move.
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.structure import (
    EMPTY_HEADING_TITLE,
    StructureAwareStrategy,
    _fenced_spans,
)

BACKTICK = "`" * 3
TILDE = "~" * 3

# Unicode whitespace that is neither space nor tab: NBSP, EM SPACE, NARROW NBSP,
# IDEOGRAPHIC SPACE. All four are `str.isspace()`; none is CommonMark whitespace.
_UNICODE_WS = {
    "nbsp": "\xa0",
    "em-space": "\u2003",
    "narrow-nbsp": "\u202f",
    "ideographic": "\u3000",
}


def _doc(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _titles(text: str, max_heading_level: int = 6) -> list[str]:
    strategy = StructureAwareStrategy(max_heading_level=max_heading_level)
    return [c.metadata["title"] for c in strategy.chunk(text)]


def test_the_issue_315_repros() -> None:
    nbsp = "\xa0"
    assert _titles("# Intro\nalpha body.\n\n# " + nbsp + "Install\nbeta body.\n") == [
        "Intro",
        "Install",
    ]
    assert _titles("# Install" + nbsp + "\nbody") == ["Install"]
    fence = _doc("# Intro", "alpha.", "", "```bash", "# install deps", "```" + nbsp)
    assert _titles(fence + _doc("# not a heading", "```")) == ["Intro"]


# Each: (id, builder(ws) -> document lines, expected titles, or a builder of
# them when they contain `ws`). Every row runs once per codepoint in `_UNICODE_WS`.
_CASES = [
    # --- position 1: after the `#`s (and their required space) ---
    (
        "after-hash-space",
        lambda ws: ["# Intro", "alpha", "", "# " + ws + "Install", "beta"],
        ["Intro", "Install"],
    ),
    (
        "after-hash-space-and-trailing",
        lambda ws: ["# Intro", "alpha", "", "## " + ws + "Install" + ws, "beta"],
        ["Intro", "Install"],
    ),
    (
        # Unicode whitespace alone is an empty heading, not no heading.
        "whitespace-only-title",
        lambda ws: ["# Intro", "alpha", "", "# " + ws, "beta"],
        ["Intro", EMPTY_HEADING_TITLE],
    ),
    (
        # The closer needs a space or tab before it, and `ws` is neither, so
        # the `#` is content: CommonMark titles it `#`.
        "whitespace-then-hash-is-not-a-closer",
        lambda ws: ["# Intro", "alpha", "", "# " + ws + "#", "beta"],
        ["Intro", "#"],
    ),
    (
        "whitespace-then-real-closer",
        lambda ws: ["# Intro", "alpha", "", "# " + ws + " ##", "beta"],
        ["Intro", EMPTY_HEADING_TITLE],
    ),
    # --- position 2: inside and at the end of a title ---
    (
        "trailing-in-title",
        lambda ws: ["# Intro", "alpha", "", "# Install" + ws, "beta"],
        ["Intro", "Install"],
    ),
    (
        # A trailing `ws` hides the closer from CommonMark (only space/tab may
        # follow it), so `##` stays in the title and only `ws` is trimmed.
        "after-a-closing-run",
        lambda ws: ["# Intro", "alpha", "", "# Install ##" + ws, "beta"],
        ["Intro", "Install ##"],
    ),
    # --- position 3: after a closing fence ---
    (
        "after-backtick-closer",
        lambda ws: ["# Intro", "a", BACKTICK + "bash", "# x", BACKTICK + ws, "# y", BACKTICK],
        ["Intro"],
    ),
    (
        "after-tilde-closer",
        lambda ws: ["# Intro", "a", TILDE, "# x", TILDE + ws, "# y", TILDE, "# B", "b"],
        ["Intro", "B"],
    ),
    (
        "after-closer-space-then-ws",
        lambda ws: ["# Intro", "a", BACKTICK, "# x", BACKTICK + " " + ws, "# y", BACKTICK],
        ["Intro"],
    ),
]

# Green on the pre-#315 tree too: they pin what the fix must not widen.
_CONTROL_CASES = [
    (
        # `#` then `ws` with no space is not a heading: CommonMark needs a
        # space or tab after the hashes, and `ws` is neither.
        "no-space-after-hash",
        lambda ws: ["# Intro", "alpha", "", "#" + ws + "Install", "beta"],
        ["Intro"],
    ),
    (
        # `ws` is not indent: `ws# A` is a paragraph.
        "ws-as-indent",
        lambda ws: ["# Intro", "alpha", "", ws + "# A", "beta"],
        ["Intro"],
    ),
    (
        "inner-whitespace-kept",
        lambda ws: ["# Intro", "alpha", "", "# a" + ws + "b", "beta"],
        lambda ws: ["Intro", "a" + ws + "b"],
    ),
    (
        "ws-before-hashes-is-content",
        lambda ws: ["# Intro", "alpha", "", "# Install" + ws + "##", "beta"],
        lambda ws: ["Intro", "Install" + ws + "##"],
    ),
    (
        # A closer followed by a tab still closes (the space/tab rule).
        "tab-after-closer-closes",
        lambda ws: ["# Intro", "a", BACKTICK, "# x", BACKTICK + "\t", "# B", "b"],
        ["Intro", "B"],
    ),
    (
        # `ws` on an opener is its info string; the fence still opens.
        "ws-info-on-opener",
        lambda ws: ["# Intro", "a", BACKTICK + ws, "# x", BACKTICK, "# B", "b"],
        ["Intro", "B"],
    ),
]


def _params(cases: list) -> list:
    return [
        pytest.param(build(ws), titles(ws) if callable(titles) else titles, id=f"{case_id}-{ws_id}")
        for case_id, build, titles in cases
        for ws_id, ws in _UNICODE_WS.items()
    ]


@pytest.mark.parametrize(("lines", "titles"), _params(_CASES))
def test_unicode_whitespace_is_not_commonmark_whitespace(
    lines: list[str], titles: list[str]
) -> None:
    assert _titles(_doc(*lines)) == titles


@pytest.mark.parametrize(("lines", "titles"), _params(_CONTROL_CASES))
def test_unicode_whitespace_controls(lines: list[str], titles: list[str]) -> None:
    assert _titles(_doc(*lines)) == titles


@pytest.mark.parametrize("ws", list(_UNICODE_WS.values()), ids=list(_UNICODE_WS))
def test_closer_followed_by_unicode_whitespace_does_not_close(ws: str) -> None:
    text = _doc(BACKTICK, "code", BACKTICK + ws, "more code")
    assert _fenced_spans(text) == [(0, len(text))]


@pytest.mark.parametrize("ws", list(_UNICODE_WS.values()), ids=list(_UNICODE_WS))
def test_atx_and_fallback_paths_title_alike(ws: str) -> None:
    # A `##` heading titles the same whether it survives the level filter or
    # falls back to `_fallback_title`, whose plain-line arm trims with `strip()`.
    text = _doc("## Install" + ws, "body")
    assert _titles(text) == ["Install"]
    assert _titles(text, max_heading_level=1) == ["Install"]
    assert _titles(_doc("Install" + ws, "body")) == ["Install"]
