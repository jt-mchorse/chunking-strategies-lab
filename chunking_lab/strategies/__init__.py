"""Chunking strategies for the lab.

Five implementations, each in its own module, sharing one interface:

    class Strategy(Protocol):
        name: str
        def chunk(self, text: str, **opts) -> list[Chunk]: ...

Strategies (D-004 — each is a standalone module so a reader can copy one
without dragging in siblings):

- `fixed`     — fixed-size sliding window with optional overlap. Baseline.
- `recursive` — split on a hierarchy of separators until each chunk fits.
- `semantic`  — embedding-boundary chunker; splits at adjacent-sentence
                cosine peaks. Uses the repo's pinned Embedder.
- `late`      — late chunking. Returns (chunk, vector) pairs because each
                chunk's vector is derived from the document-level
                embedding, not the chunk text in isolation (D-006).
- `structure` — markdown-heading-aware chunker; one chunk per
                heading-bounded section.

`Chunk` carries `start_offset` / `end_offset` (D-005) so #3's metrics
matrix can attribute retrieved chunks back to source documents without
re-tokenizing. These are Unicode codepoint offsets (Python `str` indices):
`source_text[start_offset:end_offset] == chunk.text`, which differs from a
byte slice on multibyte text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .._fields import require_non_negative_int, require_str


def check_chunk_input(text: object, source_doc_id: object) -> None:
    """Raise unless `chunk()`'s two inputs are the `str`s its signature declares.

    One definition, called at the top of every public chunk-producing method, so
    the strategies agree and a new one gets it by construction (#167, #176).

    That used to read "all five `chunk()` methods". The count was right and the
    population was wrong: there are five strategies but **six** entry points --
    `LateChunkingStrategy.chunk_with_vectors` is the surface D-006 exists for,
    the one its sibling `chunk()` delegates to, and the one
    `metrics._materialize_vectors` deliberately routes late chunking through.
    It did not call this function, so the shipped evaluator took the unguarded
    road for exactly one of the five strategies (#176). The population to
    enumerate is entry points, not classes;
    `tests/test_chunk_input_guard_entry_points.py` now discovers them rather
    than listing them.

    Before this, they did not agree. Measured over nine inputs:

        case             fixed        recursive    semantic     late           structure
        CONTROL str/str  ok, 1 chunk  ok, 1 chunk  ok, 1 chunk  ok, 1 chunk    ok, 1 chunk
        text=123         TypeError    TypeError    TypeError    AttributeError TypeError
        text=b'bytes'    ok, 1 chunk  ok, 1 chunk  TypeError    AttributeError TypeError
        text=['a']       ok, 1 chunk  ok, 1 chunk  TypeError    AttributeError TypeError
        doc_id=None/123  ok           ok           ok           ok             ok

    `fixed` and `recursive` accepting `bytes` is the sharp one, because of what
    they then produce. `Chunk`'s docstring states this module's central
    invariant -- offsets are Unicode CODEPOINT offsets, "NOT byte offsets ... on
    multibyte text these differ from byte offsets" -- and a `bytes` input makes
    them byte offsets:

        input: 'Cafe resume - <CJK>'.encode()   len(bytes) = 28, len(str) = 17
        fixed      chunk.text is `bytes`,  offsets = [0, 28)
        recursive  chunk.text is `bytes`,  offsets = [0, 28)

    So `source_text[start_offset:end_offset] == chunk.text` fails by
    construction, and `#80`'s `test_offsets_are_codepoint_not_byte_offsets`
    cannot see it because that test only ever passes `str`.

    `source_doc_id` was unvalidated on all five -- uniform rather than a
    divergence, but a chunk with `source_doc_id=None` cannot be attributed to a
    document, and the recall metrics key on exactly that field.

    Deliberately unchanged: `text=""` still yields zero chunks everywhere. Only
    the *type* is checked here, not the content.
    """
    if not isinstance(text, str):
        raise ValueError(
            f"text must be a str; got {type(text).__name__}. Offsets are Unicode "
            "codepoint offsets into that str, so a bytes/list input silently "
            "produces offsets of a different kind (#167)."
        )
    if not isinstance(source_doc_id, str):
        raise ValueError(
            f"source_doc_id must be a str; got {type(source_doc_id).__name__}. "
            "It is the key the metrics matrix attributes retrieved chunks by."
        )


@dataclass(frozen=True)
class Chunk:
    """One chunk produced by a strategy."""

    text: str
    # Offsets are Unicode CODEPOINT offsets (Python str indices), NOT byte
    # offsets: strategies populate them via `text[start:end]`, so the invariant
    # is `source_text[start_offset:end_offset] == chunk.text`. On multibyte
    # text these differ from byte offsets — slicing `source.encode()` with them
    # splits characters. See tests/test_strategies.py offset-contract tests.
    start_offset: int  # inclusive codepoint offset into source text
    end_offset: int  # exclusive codepoint offset into source text
    source_doc_id: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Type before range (#180), completing the #29/#31 sweep. That sweep put
        # `isinstance(x, int) and not isinstance(x, bool)` on nine numeric fields
        # across all five strategy classes and never reached this dataclass --
        # whose two numeric fields are the ones the offset invariant above is
        # *about*. Ordering-only, the boundary accepted:
        #
        #   start=True, end=5   -> source[True:5] is 'ello' for a chunk whose
        #                          text is 'Hello'. Silent: the invariant is
        #                          false and nothing raises.
        #   start=0.0, end=5.0  -> accepted here, TypeError at the eventual
        #                          slice, far from this constructor.
        #   start='0'           -> rejected, but by a bare TypeError out of the
        #                          `<` comparison rather than a field-named
        #                          ValueError, so a caller's `except ValueError`
        #                          (the class `check_chunk_input` raises) missed
        #                          it.
        #
        # `bool` is the one that matters most, for the same reason it did in
        # #29/#31: it is the only row that fails *silently*. It also degrades
        # `metrics.evaluate_strategy`'s ranking -- that sort breaks score ties on
        # "the chunk's stable identity (source_doc_id, start/end offsets)" (#68)
        # so recall@k is a pure function of the (score, chunk) set rather than of
        # corpus iteration order, and `True == 1`, so two chunks differing only
        # there tie on all four key components and fall back to insertion order.
        require_non_negative_int("start_offset", self.start_offset)
        require_non_negative_int("end_offset", self.end_offset)
        # The three string fields, for the same reason and by the same road:
        # `check_chunk_input` guards the strategies' *inputs*, and `Chunk` is
        # exported on `__all__` and constructible directly, so nothing guarded
        # the output. `source_doc_id` is the sharp one -- a non-str value is the
        # #176 harm (an attribution key that cannot compare equal to a query's
        # `expected_doc`, giving recall 0.0 next to snippet 1.0) reached without
        # going through a strategy at all.
        require_str("text", self.text)
        require_str("source_doc_id", self.source_doc_id)
        require_str("strategy_name", self.strategy_name)
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be >= start_offset ({self.start_offset})"
            )


@dataclass(frozen=True)
class LateChunk:
    """One late-chunk: chunk plus a vector derived from document-level context (D-006)."""

    chunk: Chunk
    vector: tuple[float, ...]


class Strategy(Protocol):
    """Single-method seam every strategy implements."""

    name: str

    def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
        """Split `text` into chunks. Each chunk's offsets index into `text`."""
        ...


from .fixed import FixedSizeStrategy  # noqa: E402
from .late import LateChunkingStrategy  # noqa: E402
from .recursive import RecursiveStrategy  # noqa: E402
from .semantic import SemanticBoundaryStrategy  # noqa: E402
from .structure import StructureAwareStrategy  # noqa: E402

__all__ = [
    "Chunk",
    "FixedSizeStrategy",
    "LateChunk",
    "LateChunkingStrategy",
    "RecursiveStrategy",
    "SemanticBoundaryStrategy",
    "Strategy",
    "StructureAwareStrategy",
]
