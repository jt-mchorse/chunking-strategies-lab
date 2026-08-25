"""All five strategies must reject the same `chunk()` inputs the same way (#167).

`Chunk`'s docstring states this module's central invariant: offsets are Unicode
**codepoint** offsets, "NOT byte offsets ... on multibyte text these differ from
byte offsets". `#80` pinned it with `test_offsets_are_codepoint_not_byte_offsets`
— over `str` input only.

`FixedSizeStrategy` and `RecursiveStrategy` accepted `bytes`, and when they did,
they produced byte offsets:

    input: 'Café résumé — 日本語'.encode()    len(bytes) = 28,  len(str) = 17
    fixed      chunk.text is `bytes`,  offsets = [0, 28)
    recursive  chunk.text is `bytes`,  offsets = [0, 28)

So `source_text[start_offset:end_offset] == chunk.text` failed by construction,
and `#80`'s test could not see it because it only ever passes `str`.

Measured over nine inputs before the fix — three different exception types
across five strategies, and two of them accepting `bytes` and `list`:

    case             fixed        recursive    semantic     late           structure
    CONTROL str/str  ok, 1 chunk  ok, 1 chunk  ok, 1 chunk  ok, 1 chunk    ok, 1 chunk
    text=None        ok, 0 chunks ok, 0 chunks ok, 0 chunks ok, 0 chunks   ok, 0 chunks
    text=123         TypeError    TypeError    TypeError    AttributeError TypeError
    text=b'bytes'    ok, 1 chunk  ok, 1 chunk  TypeError    AttributeError TypeError
    text=['a']       ok, 1 chunk  ok, 1 chunk  TypeError    AttributeError TypeError
    doc_id=None/123  ok           ok           ok           ok             ok
"""

from __future__ import annotations

from typing import Any

import pytest

from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies import (
    Chunk,
    FixedSizeStrategy,
    LateChunkingStrategy,
    RecursiveStrategy,
    SemanticBoundaryStrategy,
    StructureAwareStrategy,
)

DOC = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\n# Heading\n\nTheta iota kappa."
MULTIBYTE = "Café résumé — 日本語のテキスト"


def _strategies() -> list[tuple[str, Any]]:
    e = HashEmbedder()
    return [
        ("fixed", FixedSizeStrategy(chunk_chars=60, overlap_chars=10)),
        ("recursive", RecursiveStrategy(chunk_chars=60)),
        (
            "semantic",
            SemanticBoundaryStrategy(embedder=e, distance_threshold=0.4, min_chunk_chars=20),
        ),
        ("late", LateChunkingStrategy(embedder=e, chunk_chars=60, overlap_chars=10)),
        ("structure", StructureAwareStrategy()),
    ]


NAMES = [n for n, _ in _strategies()]

# Inputs that must be rejected, and which argument they are wrong in.
BAD_TEXT: list[tuple[str, Any]] = [
    ("None", None),
    ("int", 123),
    ("bytes", MULTIBYTE.encode("utf-8")),
    ("list of str", ["Alpha", "beta"]),
    ("dict", {"text": "Alpha"}),
]

BAD_DOC_ID: list[tuple[str, Any]] = [
    ("None", None),
    ("int", 123),
    ("bytes", b"d1"),
]


def test_the_table_covers_every_strategy_and_the_shapes_that_mattered() -> None:
    """Anti-vacuous. `bytes` and `list` are the two that were *accepted*; a table
    without them would pass against the pre-fix code for the wrong reason."""
    assert len(NAMES) == 5
    labels = [lbl for lbl, _ in BAD_TEXT]
    assert "bytes" in labels
    assert "list of str" in labels


@pytest.mark.parametrize("name", NAMES)
def test_a_normal_document_still_chunks(name: str) -> None:
    """The control. Without it, a guard that rejected everything would satisfy
    every rejection case below."""
    strategy = dict(_strategies())[name]
    chunks = strategy.chunk(DOC, source_doc_id="d1")
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(isinstance(c.text, str) for c in chunks)


@pytest.mark.parametrize("name", NAMES)
def test_an_empty_document_still_yields_zero_chunks(name: str) -> None:
    """Deliberately unchanged: only the *type* is checked, never the content."""
    strategy = dict(_strategies())[name]
    assert strategy.chunk("", source_doc_id="d1") == []


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(("label", "value"), BAD_TEXT, ids=[b[0] for b in BAD_TEXT])
def test_every_strategy_rejects_non_str_text(name: str, label: str, value: Any) -> None:
    strategy = dict(_strategies())[name]
    with pytest.raises(ValueError, match="text must be a str"):
        strategy.chunk(value, source_doc_id="d1")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(("label", "value"), BAD_DOC_ID, ids=[b[0] for b in BAD_DOC_ID])
def test_every_strategy_rejects_non_str_doc_id(name: str, label: str, value: Any) -> None:
    strategy = dict(_strategies())[name]
    with pytest.raises(ValueError, match="source_doc_id must be a str"):
        strategy.chunk(DOC, source_doc_id=value)


@pytest.mark.parametrize(("label", "value"), BAD_TEXT, ids=[b[0] for b in BAD_TEXT])
def test_all_five_produce_the_same_message(label: str, value: Any) -> None:
    """Differential: not just 'each raises', but 'all five say the same thing'.
    Before the fix there were three different exception types across the five."""
    messages = set()
    for _, strategy in _strategies():
        with pytest.raises(ValueError, match="text must be a str") as exc:
            strategy.chunk(value, source_doc_id="d1")
        messages.add(str(exc.value))
    assert len(messages) == 1, messages
    assert type(value).__name__ in messages.pop()


@pytest.mark.parametrize("name", NAMES)
def test_bytes_input_can_no_longer_produce_byte_offsets(name: str) -> None:
    """The invariant, phrased against `Chunk`'s docstring rather than the guard's
    message — so it keeps meaning something if the message ever changes.

    `MULTIBYTE` is 21 codepoints and 33 UTF-8 bytes, so a byte-offset chunk is
    detectable by its end offset alone.
    """
    strategy = dict(_strategies())[name]
    encoded = MULTIBYTE.encode("utf-8")
    assert len(encoded) > len(MULTIBYTE), "fixture must actually be multibyte"

    with pytest.raises(ValueError, match="text must be a str"):
        strategy.chunk(encoded, source_doc_id="d1")

    # And the str form still satisfies the documented invariant, which is what
    # the rejection is protecting.
    for c in strategy.chunk(MULTIBYTE, source_doc_id="d1"):
        assert MULTIBYTE[c.start_offset : c.end_offset] == c.text
        assert c.end_offset <= len(MULTIBYTE)


@pytest.mark.parametrize("name", NAMES)
def test_an_empty_doc_id_is_still_accepted(name: str) -> None:
    """`""` is a `str`, so it passes — the guard is about type, not emptiness.
    Pinned so a later tightening is a deliberate choice rather than a drift."""
    strategy = dict(_strategies())[name]
    assert len(strategy.chunk(DOC, source_doc_id="")) >= 1
