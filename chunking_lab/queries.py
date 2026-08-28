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


def _is_invisible(ch: str) -> bool:
    """True if *ch* is invisible when rendered AND survives ``str.strip()``.

    Two Unicode categories qualify, and the second one is the whole of #171.

    ``Cf`` (format) -- U+200B ZERO WIDTH SPACE, U+FEFF, U+2060 WORD JOINER,
    U+00AD SOFT HYPHEN, U+180E. These are what ``str.isspace()`` deliberately
    excludes, so they slipped past the emptiness guard (#162).

    ``Cc`` (control), **minus the ones Python calls whitespace**. The #162 rule
    excluded ``Cc`` wholesale on one stated reason: "a newline is ``Cc`` and is
    legitimate inside an ``expected_snippet`` that spans a line break". That is
    true, and it justifies excluding *newline* -- ``Cc`` has 65 members and the
    reason reaches at most the ten ``str.isspace()`` already recognises (tab,
    newline, vertical tab, form feed, carriage return, U+001C-U+001F, U+0085).
    The other 55 -- NUL, BEL, BACKSPACE, ESC, DELETE, and the C1 block
    U+0080-U+009F -- are invisible, survive ``.strip()``, and are not legitimate
    inside an id, a filename or a snippet. Measured on the same harness #162
    used (one golden query, a two-document corpus, ``FixedSizeStrategy``, k=3),
    changing nothing but one appended character::

        control (clean)     recall@3 1.000   snippet@3 1.000   validator CLEAN
        U+0000 NUL          recall@3 0.000   snippet@3 0.000   validator CLEAN
        U+0007 BEL          recall@3 0.000   snippet@3 0.000   validator CLEAN
        U+0008 BACKSPACE    recall@3 0.000   snippet@3 0.000   validator CLEAN
        U+001B ESC          recall@3 0.000   snippet@3 0.000   validator CLEAN
        U+007F DELETE       recall@3 0.000   snippet@3 0.000   validator CLEAN
        U+009B CSI          recall@3 0.000   snippet@3 0.000   validator CLEAN

    Byte-for-byte the failure #162 exists to prevent, reached through the
    sibling category. Partitioning ``Cc`` on ``str.isspace()`` rather than
    listing codepoints keeps the rule exactly as wide as its own justification:
    every character the newline argument covers stays legal, and nothing else
    does.

    Provenance, as with #162, is established rather than hypothetical. A NUL run
    inside a field is what a UTF-16 export read as UTF-8 leaves behind, and the
    C1 block is what a CP-1252 round trip produces -- the same spreadsheet-and-
    paste road that put a BOM at the start of this very file in #93.
    """
    category = unicodedata.category(ch)
    if category == "Cf":
        return True
    return category == "Cc" and not ch.isspace()


def find_invisible_char(value: str) -> tuple[int, str] | None:
    """Return ``(index, codepoint_label)`` of the first invisible character, else None.

    Named for what it finds rather than for one of the two categories it covers.
    It was ``find_format_char`` while the rule was ``Cf``-only, and that name is
    part of why the rule stayed ``Cf``-only through #162 -- the finding code it
    has always produced is ``invisible_char_{field}``, so the *code* was the
    accurate half all along (#171). See :func:`_is_invisible` for the rule and
    the measured table.

    Both readers route through here: ``Query.__post_init__`` (strict) and
    ``validate.validate_queries`` (collecting), so the loader and the linter
    cannot disagree about a file -- the parity #162 established.
    """
    for i, ch in enumerate(value):
        if _is_invisible(ch):
            name = unicodedata.name(ch, "unnamed")
            return i, f"U+{ord(ch):04X} {name}"
    return None


def invisible_char_reason(field_name: str, value: str) -> str | None:
    """Human-facing reason string for an invisible character, or None if clean.

    Names the codepoint and its index rather than echoing the value, because
    the offending character is invisible in the author's editor -- a message
    that just prints the field back looks identical to a correct one. The
    control characters #171 adds make that sharper, not weaker: several of them
    (BACKSPACE, ESC, CSI) actively *rewrite* a terminal's rendering of whatever
    is echoed after them.
    """
    hit = find_invisible_char(value)
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
            # characters (#162), nor the *control* (Cc) characters that
            # `str.isspace()` does not recognise (#171) -- both are equally
            # invisible and equally corrupting. See `_is_invisible` for the rule
            # and the measured table. Scoped to the matched/keyed fields so a
            # legitimate directional mark in an RTL `question` is still accepted;
            # `metrics.py` embeds `question` and never compares it, which is why
            # that exclusion still holds.
            if name in MATCHED_FIELDS:
                reason = invisible_char_reason(name, value)
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
