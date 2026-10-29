"""The CJK row covers `．` and the closers cover `〟` `〞` `｠` (#321).

Measured on `main` 40ff507: the Japanese technical-writing period `．`
(U+FF0E FULLWIDTH FULL STOP) was not a terminator, so
`会議は月曜日に開かれた．売上は大幅に伸びた．新製品は好評だった．` was ONE
sentence, and `SemanticBoundaryStrategy(HashEmbedder(), max_chunk_chars=20,
min_chunk_chars=0, distance_threshold=0.0)` cut it into two `size_capped`
chunks mid-word -- the fallback #140 and #240 were written to end. The
double prime quotes `〟` `〞` and the white paren `｠` were not closers, so
`彼は〝はい。〟と言った。` started its second sentence with the orphaned `〟`
(#285's harm).

`．` is also a fullwidth decimal point, so unspaced it ends a sentence only
when no digit follows; those rows are controls.
"""

from __future__ import annotations

import unicodedata

import pytest

from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies.semantic import (
    _CLOSERS,
    _TERMINATORS,
    SemanticBoundaryStrategy,
    _split_sentences_with_offsets,
)

JA_STOP = "会議は月曜日に開かれた．売上は大幅に伸びた．新製品は好評だった．"


def _sentences(text: str) -> list[str]:
    out = _split_sentences_with_offsets(text)
    for sentence, start in out:  # the offset contract stays exact
        assert text[start : start + len(sentence)] == sentence
    return [s for s, _ in out]


def test_the_fullwidth_stop_is_listed() -> None:
    assert "．" in _TERMINATORS


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        (
            "ja-unspaced",
            JA_STOP,
            ["会議は月曜日に開かれた．", "売上は大幅に伸びた．", "新製品は好評だった．"],
        ),
        ("ja-spaced", "会議は終わった． 次に進んだ．", ["会議は終わった．", "次に進んだ．"]),
        ("ja-newline", "会議は終わった．\n次に進んだ．", ["会議は終わった．", "次に進んだ．"]),
        ("ja-then-corner", "彼は「はい．」と言った．", ["彼は「はい．」", "と言った．"]),
        ("ja-doubled", "本当か？．次に進んだ．", ["本当か？．", "次に進んだ．"]),
        ("ja-after-number", "値は１．５です．次に進んだ．", ["値は１．５です．", "次に進んだ．"]),
    ],
)
def test_the_fullwidth_stop_ends_a_sentence(label: str, text: str, expected: list[str]) -> None:
    assert _sentences(text) == expected, label


@pytest.mark.parametrize(
    ("label", "text"),
    [
        # Controls: `．` between digits is a decimal point, not a boundary.
        ("fullwidth-decimal", "値は１．５です"),
        ("ascii-digit-decimal", "値は1．5です"),
        ("version", "バージョン２．０．１を使う"),
    ],
)
def test_a_fullwidth_decimal_point_does_not_split(label: str, text: str) -> None:
    assert _sentences(text) == [text], label


def test_the_japanese_text_chunks_on_sentences_not_mid_word() -> None:
    strategy = SemanticBoundaryStrategy(
        HashEmbedder(), max_chunk_chars=20, min_chunk_chars=0, distance_threshold=0.0
    )
    chunks = strategy.chunk(JA_STOP)
    assert [c.text for c in chunks] == [
        "会議は月曜日に開かれた．",
        "売上は大幅に伸びた．",
        "新製品は好評だった．",
    ]
    assert not any(c.metadata.get("size_capped") for c in chunks)


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        (
            "ja-low-double-prime",
            "彼は〝はい。〟と言った。次に進んだ。",
            ["彼は〝はい。〟", "と言った。", "次に進んだ。"],
        ),
        (
            "ja-double-prime",
            "彼は〝はい。〞と言った。次に進んだ。",
            ["彼は〝はい。〞", "と言った。", "次に進んだ。"],
        ),
        ("white-paren", "注記｟補足。｠次に進んだ。", ["注記｟補足。｠", "次に進んだ。"]),
        ("white-paren-spaced", "注記｟補足。｠ 次に進んだ。", ["注記｟補足。｠", "次に進んだ。"]),
        ("double-prime-after-stop", "彼は〝はい．〟と言った．", ["彼は〝はい．〟", "と言った．"]),
        ("double-prime-ascii", "He said 〝yes.〟 Then left.", ["He said 〝yes.〟", "Then left."]),
    ],
)
def test_the_new_closers_stay_with_their_sentence(
    label: str, text: str, expected: list[str]
) -> None:
    assert _sentences(text) == expected, label


def test_every_closing_mark_in_the_main_cjk_blocks_is_a_closer() -> None:
    # Enumerated from the Unicode catalog, not a hand list: every Pe/Pf in
    # CJK Symbols and Punctuation and in Halfwidth and Fullwidth Forms. The
    # small and vertical presentation forms (U+FE10-FE6F) are left out on
    # purpose, see the `_TERMINATORS` comment.
    blocks = [*range(0x3000, 0x3040), *range(0xFF00, 0xFFF0)]
    closing = {chr(cp) for cp in blocks if unicodedata.category(chr(cp)) in ("Pe", "Pf")}
    assert closing, "the enumeration found nothing"
    assert closing - set(_CLOSERS) == set()
