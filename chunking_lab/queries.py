"""Loader for the pinned Q&A test set.

Each line of ``data/queries.jsonl`` is one query record:

    {
      "id": "q01",
      "question": "What parameter controls the candidate list size during HNSW build?",
      "expected_doc": "01_hnsw.md",
      "expected_snippet": "ef_construction"
    }

``expected_snippet`` is verbatim text from the expected document — the
retrieval matrix in issue #3 will check whether each strategy's
retrieved chunks contain the snippet, so strategies that fragment the
relevant passage fail this query.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES_PATH = _REPO_ROOT / "data" / "queries.jsonl"

#: Fields whose value is used to *match* or *key* a record, and so must contain
#: no invisible characters. `question` is deliberately absent: it is only ever
#: embedded, never compared, and directional marks (U+200E / U+200F) are
#: legitimate inside RTL question text.
MATCHED_FIELDS: tuple[str, ...] = ("id", "expected_doc", "expected_snippet")


def find_format_char(value: str) -> tuple[int, str] | None:
    """Return ``(index, codepoint_label)`` of the first Cf character, else None.

    Unicode general category ``Cf`` (format) is the set of characters that are
    invisible when rendered *and* survive ``str.strip()`` -- U+200B ZERO WIDTH
    SPACE, U+FEFF, U+2060 WORD JOINER, U+00AD SOFT HYPHEN, U+180E. The
    emptiness guards in this module use ``not value.strip()`` precisely because
    "a whitespace-only field is as corrupting as an empty one", and that is
    true for every codepoint Python calls whitespace -- U+00A0, U+3000, U+202F
    are all removed. The Cf characters are the ones ``str.isspace()``
    deliberately excludes, so they slipped through (#162).

    Measured, one golden query against a two-document corpus at k=3, changing
    nothing but the golden data::

        clean (control)                recall@3 1.000   snippet@3 1.000
        expected_doc trailing ZWSP     recall@3 0.000   snippet@3 1.000
        expected_doc leading BOM       recall@3 0.000   snippet@3 1.000
        snippet trailing ZWSP          recall@3 1.000   snippet@3 0.000
        snippet = soft hyphen only     recall@3 1.000   snippet@3 0.000

    Every one of those was accepted by ``Query()`` and reported CLEAN by
    ``validate_queries``. A ``0.000`` produced this way is indistinguishable
    from an honest "this strategy retrieved nothing relevant" -- the same
    confusion #160 closed for the *unmeasured* case, reached here through the
    golden data instead of through a missing key.

    The provenance is established rather than hypothetical: #93 fixed a BOM at
    the *start* of this very file, and ``load_corpus`` documents that it
    tolerates one too, both because spreadsheet and Notepad exports emit them.
    A BOM *inside* a field value comes from the same paste.

    ``Cc`` (control) is deliberately not included: ``\n`` is ``Cc`` and is
    legitimate inside an ``expected_snippet`` that spans a line break.
    """
    for i, ch in enumerate(value):
        if unicodedata.category(ch) == "Cf":
            name = unicodedata.name(ch, "unnamed")
            return i, f"U+{ord(ch):04X} {name}"
    return None


def format_char_reason(field_name: str, value: str) -> str | None:
    """Human-facing reason string for a Cf character, or None if clean.

    Names the codepoint and its index rather than echoing the value, because
    the offending character is invisible in the author's editor -- a message
    that just prints the field back looks identical to a correct one.
    """
    hit = find_format_char(value)
    if hit is None:
        return None
    index, label = hit
    return (
        f"{field_name} contains the invisible character {label} at index {index}; "
        "it survives str.strip() and silently breaks the exact/substring match "
        "this field is used for, reporting the metric as 0.000"
    )


@dataclass(frozen=True)
class Query:
    """One question + golden-answer record."""

    id: str
    question: str
    expected_doc: str
    expected_snippet: str

    def __post_init__(self) -> None:
        # `Query` is on the public surface and is constructed directly (in the
        # metrics tests, the matrix script, and by any consumer who builds a
        # query set in code rather than from JSONL), so `load_queries`'
        # `_require_str` is not the only entry point. An unvalidated empty field
        # silently corrupts measurement: an empty `expected_snippet` makes
        # `expected_snippet in chunk.text` True for *every* chunk (`"" in s` is
        # always True), so snippet-hit@k reads a trivial 1.0 for every strategy;
        # an empty `expected_doc` is never a `source_doc_id`, so recall@k reads a
        # trivial 0.0. Fail loud at the dataclass boundary, the same backstop
        # pattern as `FixedSizeStrategy.__post_init__` (#29) and `_cosine` (#66).
        # `load_queries` still validates first with file:lineno context; this is
        # the in-memory invariant for the direct-construction path.
        for name, value in (
            ("id", self.id),
            ("question", self.question),
            ("expected_doc", self.expected_doc),
            ("expected_snippet", self.expected_snippet),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string, got {type(value).__name__}")
            # `not value.strip()`, not `not value`: a whitespace-only field is as
            # corrupting as an empty one. `expected_snippet="   "` makes
            # `"   " in chunk.text` True for any chunk with three consecutive
            # spaces, so snippet-hit@k still reads a trivial 1.0 — the exact #72
            # bypass, reached with whitespace instead of "". `value.strip()` is
            # falsy for both, so the literal-empty case is still rejected.
            if not value.strip():
                raise ValueError(f"{name} must be non-empty and not whitespace-only")
            # The emptiness rule above covers every codepoint `str.isspace()`
            # calls whitespace. It does not cover the Unicode *format* (Cf)
            # characters, which are equally invisible and equally corrupting --
            # see `find_format_char` for the measured table (#162). Scoped to
            # the matched/keyed fields so a legitimate directional mark in an
            # RTL `question` is still accepted.
            if name in MATCHED_FIELDS:
                reason = format_char_reason(name, value)
                if reason is not None:
                    raise ValueError(reason)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty and not whitespace-only")
    return value


def load_queries(path: PathLike[str] | str | None = None) -> list[Query]:
    """Load all queries from a JSONL file. Validates required fields."""
    p = Path(path) if path is not None else DEFAULT_QUERIES_PATH
    if not p.exists():
        raise FileNotFoundError(f"queries file not found: {p}")
    out: list[Query] = []
    seen_ids: set[str] = set()
    # utf-8-sig transparently strips a leading BOM (EF BB BF — the default for
    # Windows Notepad and some spreadsheet exports) and is a no-op for BOM-less
    # UTF-8. `.strip()` below does not remove U+FEFF, so without this the BOM
    # reaches json.loads on line 1 and the whole file fails to load (#93).
    with p.open("r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
            # A line can be well-formed JSON yet not an object — a bare number,
            # string, array, or bool. Without this guard the very next
            # `rec.get(...)` raises `AttributeError`, which escapes the
            # documented `ValueError` contract a caller catches around this
            # loader. The sibling validator (`validate.py`) already reports this
            # exact shape as a `not_an_object` finding; this keeps the two
            # file-walkers in parity on the same malformed input.
            if not isinstance(rec, dict):
                raise ValueError(
                    f"{p}:{lineno}: row must be a JSON object, got {type(rec).__name__}"
                )
            try:
                q = Query(
                    id=_require_str(rec.get("id"), f"{p}:{lineno} 'id'"),
                    question=_require_str(rec.get("question"), f"{p}:{lineno} 'question'"),
                    expected_doc=_require_str(
                        rec.get("expected_doc"), f"{p}:{lineno} 'expected_doc'"
                    ),
                    expected_snippet=_require_str(
                        rec.get("expected_snippet"), f"{p}:{lineno} 'expected_snippet'"
                    ),
                )
            except ValueError:
                raise
            if q.id in seen_ids:
                raise ValueError(f"{p}:{lineno}: duplicate query id {q.id!r}")
            seen_ids.add(q.id)
            out.append(q)
    if not out:
        raise ValueError(f"{p}: queries file is empty")
    return out
