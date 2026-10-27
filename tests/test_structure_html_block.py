"""`StructureAwareStrategy` must not read `#` inside an HTML block as a heading (#317).

#308 skipped CommonMark §4.6 type 2 (`<!--` ... `-->`) and no other type, so
`<pre>\\n# install deps\\n</pre>` still titled a chunk `install deps`: text
that is not a heading became a chunk boundary and a `title`, #152's harm.
Types 1, 3, 4 and 5 end at a fixed string, the same shape as type 2:

  1  `<pre`, `<script`, `<style`, `<textarea` (any case), then space, tab, `>`
     or end of line; ended by any of the four closing tags (any case)
  3  `<?` to `?>`
  4  `<!` + ASCII letter to `>`
  5  `<![CDATA[` to `]]>`

Each block ends on the first line containing its end string (possibly the
opener's own line), or at end-of-text. Types 6 and 7 stay deferred: they end
at a blank line and need paragraph context.

Every expected title list below matches markdown-it-py 4.2.0's commonmark
preset (run in a side venv, not a dependency: D-002 keeps the package
dependency-free), except `t4-lowercase-declaration`. markdown-it-py still uses
the 0.30 type-4 rule (`<!` + an UPPERCASE letter); CommonMark 0.31 widened it
to any ASCII letter ("HTML declarations need not be limited to all capital
ASCII letters"), and this module follows 0.31.2.

Latent on the pinned corpus (no `data/corpus/*.md` file opens one of these
blocks), so the canonical fixtures regenerate byte-identically.
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


def test_the_issue_317_repro() -> None:
    doc = "# Setup\n\n<pre>\n# install deps\nmake deps\n</pre>\n\n# Usage\nu\n"
    assert _titles(doc) == ["Setup", "Usage"]


# Each: (id, lines, expected titles). Red against the pre-#317 tree.
_FIXED_CASES = [
    # Type 1: each tag name, and the three things that may follow it.
    ("t1-script", ["# A", "<script>", "# x", "</script>", "# B", "b"], ["A", "B"]),
    ("t1-style", ["# A", "<style>", "# x", "</style>", "# B", "b"], ["A", "B"]),
    ("t1-textarea", ["# A", "<textarea>", "# x", "</textarea>", "# B", "b"], ["A", "B"]),
    (
        "t1-followed-by-space",
        ["# A", '<script type="module">', "# x", "</script>", "# B", "b"],
        ["A", "B"],
    ),
    ("t1-followed-by-tab", ["# A", "<pre\tclass=x>", "# x", "</pre>", "# B", "b"], ["A", "B"]),
    ("t1-followed-by-end-of-line", ["# A", "<pre", "# x", "</pre>", "# B", "b"], ["A", "B"]),
    # Case-insensitive at both ends.
    ("t1-uppercase", ["# A", "<PRE>", "# x", "</PRE>", "# B", "b"], ["A", "B"]),
    ("t1-mixed-case", ["# A", "<Script>", "# x", "</sCrIpT>", "# B", "b"], ["A", "B"]),
    # Any of the four closing tags ends a type-1 block, not only its own.
    ("t1-closed-by-another-tag", ["# A", "<pre>", "# x", "</script>", "# B", "b"], ["A", "B"]),
    ("t1-closer-mid-line", ["# A", "<pre>", "# x", "code </pre> tail", "# B", "b"], ["A", "B"]),
    ("t1-indented-three-spaces", ["# A", "   <pre>", "# x", "</pre>", "# B", "b"], ["A", "B"]),
    # `</pre` without its `>` is not the end string.
    (
        "t1-incomplete-closer-does-not-close",
        ["# A", "<pre>", "# x", "</pre", "# y", "</pre>", "# B", "b"],
        ["A", "B"],
    ),
    ("t1-unclosed-runs-to-end-of-text", ["# A", "<pre>", "# x", "# y"], ["A"]),
    # Type 3.
    ("t3-processing-instruction", ["# A", "<?php", "# x", "?>", "# B", "b"], ["A", "B"]),
    ("t3-unclosed-runs-to-end-of-text", ["# A", "<?php", "# x"], ["A"]),
    # Type 4.
    ("t4-declaration", ["# A", "<!DOCTYPE", "# x", ">", "# B", "b"], ["A", "B"]),
    # CommonMark 0.31.2: any ASCII letter (markdown-it-py 4.2.0 says
    # ['A', 'x', 'B'] here; it implements the 0.30 uppercase-only rule).
    ("t4-lowercase-declaration", ["# A", "<!doctype", "# x", ">", "# B", "b"], ["A", "B"]),
    ("t4-unclosed-runs-to-end-of-text", ["# A", "<!ENTITY", "# x"], ["A"]),
    # Type 5. `]>` is not `]]>`.
    ("t5-cdata", ["# A", "<![CDATA[", "# x", "]]>", "# B", "b"], ["A", "B"]),
    (
        "t5-single-bracket-does-not-close",
        ["# A", "<![CDATA[", "# x", "]>", "# y", "]]>", "# B", "b"],
        ["A", "B"],
    ),
    ("t5-unclosed-runs-to-end-of-text", ["# A", "<![CDATA[", "# x"], ["A"]),
    # Only the open block's own end string closes it.
    (
        "comment-closer-does-not-close-a-pre",
        ["# A", "<pre>", "-->", "# x", "</pre>", "# B", "b"],
        ["A", "B"],
    ),
    # A fence line inside an HTML block is HTML, not an opener. Read as one,
    # nothing closed it and `# B` vanished into a phantom fence (#156's harm).
    (
        "fence-line-inside-a-pre-is-not-an-opener",
        ["# A", "<pre>", BACKTICK, "</pre>", "# B", "b"],
        ["A", "B"],
    ),
    (
        "fence-line-inside-cdata-is-not-an-opener",
        ["# A", "<![CDATA[", BACKTICK, "]]>", "# B", "b"],
        ["A", "B"],
    ),
]


@pytest.mark.parametrize(
    ("lines", "titles"),
    [(lines, titles) for _, lines, titles in _FIXED_CASES],
    ids=[case_id for case_id, _, _ in _FIXED_CASES],
)
def test_hash_line_inside_an_html_block_is_not_a_heading(
    lines: list[str], titles: list[str]
) -> None:
    assert _titles(_doc(*lines)) == titles


@pytest.mark.parametrize(
    "block",
    [
        ["<pre>", "# draft", "</pre>"],
        ["<?php", "# draft", "?>"],
        ["<!DOCTYPE", "# draft", ">"],
        ["<![CDATA[", "# draft", "]]>"],
    ],
    ids=["t1", "t3", "t4", "t5"],
)
def test_fallback_title_skips_an_html_block(block: list[str]) -> None:
    # No surviving heading: the first line outside every skipped block titles
    # the chunk (#288), and an HTML block's lines are no title either.
    assert _titles(_doc(*block, "Body text")) == ["Body text"]


def test_fence_line_inside_an_html_block_opens_no_fence_span() -> None:
    assert _fenced_spans(_doc("<pre>", BACKTICK, "</pre>", "# B")) == []


# Controls: right before #317 and must stay right.
_CONTROL_CASES = [
    # The end string on the opener's own line closes the block there.
    ("t1-closes-on-its-own-line", ["# A", "<pre>x</pre>", "# B", "b"], ["A", "B"]),
    ("t3-closes-on-its-own-line", ["# A", "<?xml version='1.0'?>", "# B", "b"], ["A", "B"]),
    # `?>` overlaps the opener's `?`, as `<!-->` does for type 2.
    ("t3-degenerate-opener", ["# A", "<?>", "# B", "b"], ["A", "B"]),
    ("t4-closes-on-its-own-line", ["# A", "<!DOCTYPE html>", "# B", "b"], ["A", "B"]),
    ("t5-closes-on-its-own-line", ["# A", "<![CDATA[x]]>", "# B", "b"], ["A", "B"]),
    # Inside an open fence an HTML opener is code.
    (
        "pre-opened-inside-a-fence-is-code",
        ["# A", BACKTICK, "<pre>", BACKTICK, "# B", "b"],
        ["A", "B"],
    ),
    # A `</pre>` inside a comment does not end the comment.
    (
        "pre-closer-does-not-close-a-comment",
        ["# A", "<!--", "</pre>", "# x", "-->", "# B", "b"],
        ["A", "B"],
    ),
    # Not openers: a longer tag name, a 4-space indent (#156), a mid-line
    # tag, `<!` + a digit, and a lowercase `cdata` (type 5 is case-sensitive).
    ("pref-is-not-type-1", ["# A", "<pref", "# B", "b"], ["A", "B"]),
    ("opener-indented-four-spaces", ["# A", "", "    <pre>", "# B", "b", "</pre>"], ["A", "B"]),
    ("opener-mid-line", ["# A", "text <pre>", "# B", "b", "</pre>"], ["A", "B"]),
    ("bang-digit-is-not-type-4", ["# A", "<!1", "# B", "b"], ["A", "B"]),
    ("lowercase-cdata-is-not-type-5", ["# A", "<![cdata[", "# B", "b"], ["A", "B"]),
]


@pytest.mark.parametrize(
    ("lines", "titles"),
    [(lines, titles) for _, lines, titles in _CONTROL_CASES],
    ids=[case_id for case_id, _, _ in _CONTROL_CASES],
)
def test_html_block_controls(lines: list[str], titles: list[str]) -> None:
    assert _titles(_doc(*lines)) == titles


def test_html_block_spans_keep_full_coverage() -> None:
    # The skipped block stays inside its section's chunk: no codepoint lost.
    doc = "# Setup\n\n<pre>\n# install deps\nmake deps\n</pre>\n\n# Usage\nu\n"
    chunks = StructureAwareStrategy().chunk(doc)
    assert "".join(c.text for c in chunks) == doc
    assert "<pre>\n# install deps\nmake deps\n</pre>" in chunks[0].text
