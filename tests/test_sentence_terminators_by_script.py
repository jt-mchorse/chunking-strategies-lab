"""Sentence terminators beyond CJK and Arabic split sentences (#240).

`_TERMINATORS` (#140) covered `.!?…` plus CJK `。！？` and Arabic `؟`. Measured
on `main` (found by a hunt agent, re-run here): three-sentence texts in Hindi
(danda `।`), Urdu (`۔`), Amharic (`።`) and Armenian (`։`) each came back as ONE
sentence, and with `SemanticBoundaryStrategy(HashEmbedder(),
distance_threshold=0.0, min_chunk_chars=0, max_chunk_chars=60)` the Hindi text
was two `size_capped` chunks cut mid-word -- the silent fixed-size fallback #140
was written to end.
"""

from __future__ import annotations

import pytest

from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies.semantic import (
    _TERMINATORS,
    SemanticBoundaryStrategy,
    _split_sentences_with_offsets,
)

SCRIPTS = {
    "hindi": "राम घर गया। सीता बाजार गई। नया उत्पाद अगले महीने आएगा।",
    "urdu": "یہ کتاب ہے۔ وہ گھر گیا۔ بارش ہو رہی ہے۔",
    "amharic": "ሰላም ነው። ዛሬ ሞቃት ነው። ነገ ይዘንባል።",
    "armenian": "Բարեւ։ Այսօր տաք է։ Վաղը անձրեւ կլինի։",
    "myanmar": "မင်္ဂလာပါ။ ဒီနေ့ ပူတယ်။ မနက်ဖြန် မိုးရွာမယ်။",
    "khmer": "សួស្តី។ ថ្ងៃនេះក្តៅ។ ថ្ងៃស្អែកភ្លៀង។",
    "english": "The cat sat. The dog ran. The bird flew.",  # control
}


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_three_sentences_split_into_three(script: str) -> None:
    text = SCRIPTS[script]
    sentences = _split_sentences_with_offsets(text)
    assert len(sentences) == 3, sentences
    for sentence, start in sentences:  # the offset contract stays exact
        assert text[start : start + len(sentence)] == sentence


def test_the_hindi_text_chunks_on_sentences_not_mid_word() -> None:
    strategy = SemanticBoundaryStrategy(
        HashEmbedder(), distance_threshold=0.0, min_chunk_chars=0, max_chunk_chars=60
    )
    chunks = strategy.chunk(SCRIPTS["hindi"])
    assert [c.text.strip() for c in chunks] == [
        "राम घर गया।",
        "सीता बाजार गई।",
        "नया उत्पाद अगले महीने आएगा।",
    ]
    assert not any(c.metadata.get("size_capped") for c in chunks)


@pytest.mark.parametrize("terminator", sorted(set(_TERMINATORS) - set(".")))
def test_every_listed_terminator_ends_a_sentence(terminator: str) -> None:
    text = f"first{terminator} second"
    assert [s for s, _ in _split_sentences_with_offsets(text)] == [f"first{terminator}", "second"]


def test_the_greek_question_mark_is_not_listed() -> None:
    # NFC maps U+037E to ';', which ends no sentence elsewhere.
    assert ";" not in _TERMINATORS
    assert len(_split_sentences_with_offsets("one; two")) == 1


def test_a_decimal_point_still_does_not_split() -> None:
    assert len(_split_sentences_with_offsets("Pi is 3.14 today. Fine.")) == 2
