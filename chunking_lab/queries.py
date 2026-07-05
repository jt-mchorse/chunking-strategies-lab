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
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES_PATH = _REPO_ROOT / "data" / "queries.jsonl"


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
