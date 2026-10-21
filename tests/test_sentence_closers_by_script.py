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


# #291 read the run as "terminators, THEN closers", so a terminator after a
# closer ended no sentence (#293). Measured on `main` 52ae6c1:
# `['他喊道：“救命！”……没有人回答。']` and `['（他说「好。」）……然后走了。']`, each
# one sentence where #291's own rule ("each branch matches only a WHOLE run")
# gives two. A FULL-WIDTH terminator after the closer (`「好。」。然后`) was
# never affected: it starts a run of its own.
@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        (
            "zh-closer-then-ellipsis",
            "他喊道：“救命！”……没有人回答。",
            ["他喊道：“救命！”……", "没有人回答。"],
        ),
        (
            "zh-two-closers-then-ellipsis",
            "（他说「好。」）……然后走了。",
            ["（他说「好。」）……", "然后走了。"],
        ),
        ("zh-closer-then-one-ellipsis", "他说：“好。”…然后走了。", ["他说：“好。”…", "然后走了。"]),
        (
            "zh-interleaved",
            "他说「真的？！」…」然后走了。",
            ["他说「真的？！」…」", "然后走了。"],
        ),
        # A spaced `“` after a doubled terminator closes the sentence, as it
        # does after a single one (control below); #291's (b) fired before it.
        (
            "zh-doubled-terminator-spaced-quote",
            "他走了！？“ 我不信。",
            ["他走了！？“", "我不信。"],
        ),
    ],
)
def test_a_terminator_after_a_closer_stays_in_the_run(
    label: str, text: str, expected: list[str]
) -> None:
    assert _sentences(text) == expected, label


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        # Controls (#293): green on `main` before the fix and must not move.
        ("zh-single-terminator-spaced-quote", "他走了！“ 我不信。", ["他走了！“", "我不信。"]),
        (
            "zh-closer-then-ellipsis-spaced",
            "他喊道：“救命！”…… 没有人回答。",
            ["他喊道：“救命！”……", "没有人回答。"],
        ),
        ("zh-run-at-end-of-text", "他喊道：“救命！”……", ["他喊道：“救命！”……"]),
        # A long run is still one sentence end (#291 took 3.3 s on this).
        ("zh-long-run", "好" + "！" * 16000, ["好" + "！" * 16000]),
    ],
)
def test_closer_terminator_controls_unchanged(label: str, text: str, expected: list[str]) -> None:
    assert _sentences(text) == expected, label


def test_a_terminator_after_a_closer_is_a_semantic_boundary() -> None:
    strategy = SemanticBoundaryStrategy(HashEmbedder(), distance_threshold=0.0, min_chunk_chars=0)
    chunks = strategy.chunk("他喊道：“救命！”……没有人回答。")
    assert [c.text for c in chunks] == ["他喊道：“救命！”……", "没有人回答。"]


# French typography spaces a closing `»` off the terminator (#299). Python's
# `\s` matches U+00A0 and U+202F, so branch (a) ended the sentence at the `.`
# and the closer opened the next one. Measured on `main` 8e10489:
# `['Il a dit «\xa0Oui.', '» Puis il est parti.']`, the same with U+202F and an
# ASCII space, and `['Il a dit «\xa0Oui.', '»']` at end of text.
@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        (
            "fr-nbsp",
            "Il a dit «\u00a0Oui.\u00a0» Puis il est parti.",
            ["Il a dit «\u00a0Oui.\u00a0»", "Puis il est parti."],
        ),
        (
            "fr-narrow-nbsp",
            "Il a dit «\u202fOui.\u202f» Puis il est parti.",
            ["Il a dit «\u202fOui.\u202f»", "Puis il est parti."],
        ),
        (
            "fr-thin-space",
            "Il a dit «\u2009Oui.\u2009» Puis il est parti.",
            ["Il a dit «\u2009Oui.\u2009»", "Puis il est parti."],
        ),
        (
            "fr-ascii-space",
            "Il a dit « Oui. » Puis il est parti.",
            ["Il a dit « Oui. »", "Puis il est parti."],
        ),
        (
            "fr-single-guillemet",
            "Il a dit ‹\u00a0Oui.\u00a0› Puis.",
            ["Il a dit ‹\u00a0Oui.\u00a0›", "Puis."],
        ),
        ("fr-nbsp-at-end-of-text", "Il a dit «\u00a0Oui.\u00a0»", ["Il a dit «\u00a0Oui.\u00a0»"]),
        # Both guillemets of a quote that opens a sentence: the `«` stays on
        # the second sentence and the `»` closes it.
        (
            "fr-quote-after-boundary",
            "Il est parti. «\u00a0Bonjour.\u00a0» Fin.",
            ["Il est parti.", "«\u00a0Bonjour.\u00a0»", "Fin."],
        ),
        (
            "fr-nbsp-question",
            "Il a demandé «\u00a0Pourquoi\u00a0?\u00a0» Puis il est parti.",
            ["Il a demandé «\u00a0Pourquoi\u00a0?\u00a0»", "Puis il est parti."],
        ),
    ],
)
def test_a_spaced_french_closer_stays_with_its_sentence(
    label: str, text: str, expected: list[str]
) -> None:
    assert _sentences(text) == expected, label


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        # Controls (#299): green on `main` before the fix and must not move.
        (
            "fr-unspaced",
            "Il a dit «Oui.» Puis il est parti.",
            ["Il a dit «Oui.»", "Puis il est parti."],
        ),
        ("fr-no-closer", "Oui. Puis il est parti.", ["Oui.", "Puis il est parti."]),
        ("fr-nbsp-no-closer", "Oui.\u00a0Puis il est parti.", ["Oui.", "Puis il est parti."]),
        # A spaced `»` that OPENS (German) is followed by text: next sentence.
        ("de-spaced-opener", "Er sagte. »Ja.« Dann.", ["Er sagte.", "»Ja.«", "Dann."]),
        # `«` OPENS in French, with a no-break space after it: it must not be
        # pulled back onto the previous sentence.
        (
            "fr-opener-after-boundary",
            "Il est parti. «\u00a0Bonjour Fin.",
            ["Il est parti.", "«\u00a0Bonjour Fin."],
        ),
        # A line break is a boundary, not horizontal space.
        ("fr-newline-before-closer", "Oui.\n» Puis.", ["Oui.", "» Puis."]),
    ],
)
def test_spaced_french_closer_controls_unchanged(
    label: str, text: str, expected: list[str]
) -> None:
    assert _sentences(text) == expected, label


def test_a_spaced_french_closer_is_not_a_semantic_chunk() -> None:
    strategy = SemanticBoundaryStrategy(HashEmbedder(), distance_threshold=0.0, min_chunk_chars=0)
    text = "Il a dit «\u00a0Oui.\u00a0» Puis il est parti."
    chunks = strategy.chunk(text)
    assert [c.text.strip() for c in chunks] == ["Il a dit «\u00a0Oui.\u00a0»", "Puis il est parti."]
    for c in chunks:
        assert text[c.start_offset : c.end_offset] == c.text
