"""`docs/setup.md` describes the pinned corpus the way the corpus is (#246).

The substrate spec said "600–1200 words each" and "Each document has ...
code blocks". Every document was 386–473 words and only two contained a fenced
code block. Nothing tested the prose, and the commit that wrote it is the only
one that has touched the corpus, so it was never true. These tests read both
claims out of the doc rather than restating them, so changing either the doc
or the corpus without the other fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

from chunking_lab import load_corpus

SETUP_MD = Path(__file__).resolve().parents[1] / "docs" / "setup.md"

_WORDS_RE = re.compile(r"\*\*Documents:\*\* (\d+) [^,]*, (\d+)–(\d+) words each")
_CODE_SENTENCE_RE = re.compile(r"Each document has .*?differentiate\.", re.S)


def _setup_text() -> str:
    return SETUP_MD.read_text(encoding="utf-8")


def _has_fenced_code(text: str) -> bool:
    return any(line.lstrip().startswith(("```", "~~~")) for line in text.splitlines())


def test_document_count_and_word_range_match_the_corpus() -> None:
    m = _WORDS_RE.search(_setup_text())
    assert m, "docs/setup.md no longer states 'N ... articles, A–B words each'"
    count, low, high = (int(g) for g in m.groups())
    words = {d.filename: len(d.text.split()) for d in load_corpus()}
    assert len(words) == count
    outside = {name: n for name, n in words.items() if not low <= n <= high}
    assert not outside, f"docs/setup.md says {low}–{high} words each; outside it: {outside}"


def test_named_code_block_documents_are_exactly_the_ones_with_fences() -> None:
    m = _CODE_SENTENCE_RE.search(_setup_text())
    assert m, "docs/setup.md no longer has the 'Each document has ...' sentence"
    named = set(re.findall(r"`([^`]+\.md)`", m.group(0)))
    with_fences = {d.filename for d in load_corpus() if _has_fenced_code(d.text)}
    assert with_fences, "control: the corpus has at least one fenced code block"
    assert named == with_fences, (
        f"docs/setup.md names {sorted(named)} as carrying code blocks; "
        f"the corpus documents with a fence are {sorted(with_fences)}"
    )
