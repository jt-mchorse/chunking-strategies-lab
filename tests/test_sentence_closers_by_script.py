"""Closing marks beyond the 8 `_CLOSERS` listed split sentences correctly (#285).

`_CLOSERS` (#140) knew `" ” ’ ' ) ] 」 』`. Measured on `main`:

- a German `„Ja.“ Dann`, French/Swiss `«Ja.» Dann` or `‹Ja.› Dann` came back as
  ONE sentence. A closer between terminator and whitespace hid the boundary,
  which is #140's harm 2 for every closer it did not list;
- after an unspaced `。`, the branch-(b) alternative that is meant to "refuse to
  fire when a closer follows" fired before `）` `】` `》`, so the next sentence
  started with the orphaned bracket (`['会议在周一举行（见附录。', '）收入大幅增长。']`).

The locale-dependent quotes (`“ ‘ » « › ‹`) count as closers only in the
spaced branch. In the unspaced CJK branch `“` opens a quote, and that row is
pinned as a control.
"""

from __future__ import annotations

import pytest

from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies.semantic import (
    SemanticBoundaryStrategy,
    _split_sentences_with_offsets,
)


def _sentences(text: str) -> list[str]:
    out = _split_sentences_with_offsets(text)
    for sentence, start in out:  # the offset contract stays exact
        assert text[start : start + len(sentence)] == sentence
    return [s for s, _ in out]


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        # Spaced: a closer between the terminator and the whitespace.
        ("de-low-high-double", "Er sagte „Ja.“ Dann ging er.", ["Er sagte „Ja.“", "Dann ging er."]),
        ("de-low-high-single", "Er sagte ‚Ja.‘ Dann ging er.", ["Er sagte ‚Ja.‘", "Dann ging er."]),
        ("fr-ch-guillemets", "Er sagte «Ja.» Dann ging er.", ["Er sagte «Ja.»", "Dann ging er."]),
        (
            "de-reversed-guillemets",
            "Er sagte »Ja.« Dann ging er.",
            ["Er sagte »Ja.«", "Dann ging er."],
        ),
        ("single-guillemets", "Er sagte ‹Ja.› Dann ging er.", ["Er sagte ‹Ja.›", "Dann ging er."]),
        (
            "ja-fullwidth-paren-spaced",
            "会議は月曜日（付録参照。） 収益は増加した。",
            ["会議は月曜日（付録参照。）", "収益は増加した。"],
        ),
        # Unspaced CJK: the bracket stays on the sentence it closes.
        (
            "zh-fullwidth-paren",
            "会议在周一举行（见附录。）收入大幅增长。",
            ["会议在周一举行（见附录。）", "收入大幅增长。"],
        ),
        ("zh-lenticular", "注意【重要。】收入大幅增长。", ["注意【重要。】", "收入大幅增长。"]),
        ("zh-double-angle", "他读了《红楼梦。》然后睡了。", ["他读了《红楼梦。》", "然后睡了。"]),
        ("zh-angle", "他读了〈序言。〉然后睡了。", ["他读了〈序言。〉", "然后睡了。"]),
    ],
)
def test_closer_stays_with_its_sentence(label: str, text: str, expected: list[str]) -> None:
    assert _sentences(text) == expected, label


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        # Controls: behaviour that was already right and must not move.
        ("en-curly-double", "He said “Yes.” Then he left.", ["He said “Yes.”", "Then he left."]),
        (
            "en-opening-quote-after-space",
            "He said. “Next one.” Done.",
            ["He said.", "“Next one.”", "Done."],
        ),
        ("zh-corner-bracket", "他说「好了。」然后走了。", ["他说「好了。」", "然后走了。"]),
        # `“` / `‘` OPEN a quote after an unspaced `。`: they must stay on the
        # second sentence, which is why they are spaced-branch-only.
        ("zh-opening-curly-double", "他走了。“你好”他说。", ["他走了。", "“你好”他说。"]),
        ("zh-opening-curly-single", "他走了。‘你好’他说。", ["他走了。", "‘你好’他说。"]),
    ],
)
def test_controls_unchanged(label: str, text: str, expected: list[str]) -> None:
    assert _sentences(text) == expected, label


def test_the_german_quote_is_a_semantic_boundary() -> None:
    strategy = SemanticBoundaryStrategy(HashEmbedder(), distance_threshold=0.0, min_chunk_chars=0)
    chunks = strategy.chunk("Er sagte „Ja.“ Dann ging er.")
    assert [c.text.strip() for c in chunks] == ["Er sagte „Ja.“", "Dann ging er."]


# A run of terminators and closers is ONE sentence end (#287). Measured on
# `main` 6c40be3, after #285: a doubled full-width terminator or two closers
# still orphaned a one-character sentence (`['真的吗！', '？', '我不相信。']`,
# `['（他说「好。」', '）然后走了。']`), and the spaced branch accepted only one
# closer, so `("yes.") Then` stayed one sentence.
@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("zh-exclaim-question", "真的吗！？我不相信。", ["真的吗！？", "我不相信。"]),
        ("zh-question-exclaim", "真的吗？！我不相信。", ["真的吗？！", "我不相信。"]),
        ("zh-doubled-period", "会议举行。。收入增长。", ["会议举行。。", "收入增长。"]),
        ("zh-two-closers", "（他说「好。」）然后走了。", ["（他说「好。」）", "然后走了。"]),
        (
            "zh-doubled-terminator-then-closer",
            "他说「真的吗！？」然后走了。",
            ["他说「真的吗！？」", "然后走了。"],
        ),
        # Spaced CJK: branch (b) fired after `！` before branch (a) could see
        # the space after `？`.
        ("zh-doubled-terminator-spaced", "真的吗！？ 我不相信。", ["真的吗！？", "我不相信。"]),
        # `“` opens the next quote after the whole run, as after a lone `。`.
        (
            "zh-doubled-terminator-then-opening-quote",
            "他走了！？“你好”他说。",
            ["他走了！？", "“你好”他说。"],
        ),
        ("en-two-closers", 'He said ("yes.") Then he left.', ['He said ("yes.")', "Then he left."]),
        (
            "en-three-closers",
            "He said (“‘yes.’”) Then he left.",
            ["He said (“‘yes.’”)", "Then he left."],
        ),
    ],
)
def test_a_terminator_closer_run_is_one_sentence_end(
    label: str, text: str, expected: list[str]
) -> None:
    assert _sentences(text) == expected, label


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        # Controls (#287): green on `main` before the fix and must not move.
        ("zh-single-terminator", "会议举行。收入增长。", ["会议举行。", "收入增长。"]),
        ("en-single-closer", 'He said "yes." Then he left.', ['He said "yes."', "Then he left."]),
        ("en-ellipsis", "Wait... Then go.", ["Wait...", "Then go."]),
        ("en-decimal", "Pi is 3.14 today. Fine.", ["Pi is 3.14 today.", "Fine."]),
        ("en-exclaim-question-spaced", "Really?! Yes.", ["Really?!", "Yes."]),
    ],
)
def test_run_controls_unchanged(label: str, text: str, expected: list[str]) -> None:
    assert _sentences(text) == expected, label


def test_a_doubled_terminator_is_not_a_semantic_chunk() -> None:
    strategy = SemanticBoundaryStrategy(HashEmbedder(), distance_threshold=0.0, min_chunk_chars=0)
    chunks = strategy.chunk("真的吗！？我不相信。他走了。")
    assert [c.text for c in chunks] == ["真的吗！？", "我不相信。", "他走了。"]
