"""Queries-JSONL collecting-mode lint (#37).

``chunking_lab.queries.load_queries`` raises ``ValueError`` on the first
malformed line, missing field, non-string value, duplicate ``id``, or
empty file. An operator who extends the pinned 12-query set with their
own retrieval cases sees one error per ``load_queries`` invocation: fix,
retry, fix, retry.

``validate_queries(path, corpus_dir=None)`` walks the same file in
*collecting* mode and returns every finding in one pass. Same shape as
``eval_harness.dataset.validate_dataset`` (llm-eval-harness#56/#57),
``prompt_regression.validate.validate_snapshots`` (prompt-regression-suite#49/#50),
and ``emb_shootout.validate.validate_corpus`` (embedding-model-shootout#45/#46).

A frozen ``ValidationReport`` carries the result: ``n_rows``,
``n_valid``, a tuple of ``ValidationFinding`` records (each
``(line_no, reason, code)``), and an ``ok`` property that is true iff
the file produced at least one valid row AND zero findings.

When ``corpus_dir`` is provided, each row's ``expected_doc`` is checked
for membership in the *loaded* corpus — the set of ``*.md`` files
``load_corpus`` enumerates, keyed by basename — not merely for existence
on disk. This is the cross-file invariant that catches typo'd doc
references that silently invalidate recall metrics (the run completes,
the number becomes meaningless): a non-``.md`` file, a directory named
``*.md``, or a case-mismatched name resolves on disk but is never a
loaded ``Document.filename``, so recall for that query is permanently 0.

Finding codes (1-indexed line numbers; blank lines silently skipped to
match ``load_queries``):

- ``malformed_json``           — ``json.loads`` raised.
- ``not_an_object``            — parsed JSON not a dict (e.g., bare string).
- ``missing_id`` /
  ``missing_question`` /
  ``missing_expected_doc`` /
  ``missing_expected_snippet`` — required field absent.
- ``non_string_<field>``       — field present but not a string (one per
                                 required field).
- ``empty_<field>``            — field present, string, empty *or
                                 whitespace-only* (one per required field;
                                 #92 — a blank field corrupts metrics the same
                                 way an empty one does).
- ``duplicate_id``             — same ``id`` seen at multiple lines.
- ``expected_doc_not_found``   — *only when corpus_dir is provided* —
                                 ``expected_doc`` is not a loaded corpus
                                 document (a ``*.md`` file matched by name).
- ``empty``                    — file contained zero rows; reported once
                                 with ``line_no=0``.

A missing ``path`` raises ``FileNotFoundError`` so the CLI surfaces it as
exit 2. A missing ``corpus_dir`` (when supplied but doesn't exist) does
the same — same convention as the sister-repo validators.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunking_lab.io_utils import atomic_write_text

REQUIRED_FIELDS: tuple[str, ...] = ("id", "question", "expected_doc", "expected_snippet")


@dataclass(frozen=True)
class ValidationFinding:
    """One row-level issue surfaced by ``validate_queries``.

    Mirrors the harness / emb-shootout shape so JSON consumers can route
    on ``code`` without parsing prose.
    """

    line_no: int
    reason: str
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {"line_no": self.line_no, "reason": self.reason, "code": self.code}


@dataclass(frozen=True)
class ValidationReport:
    """Result of walking a queries JSONL in collecting mode.

    ``ok`` is true iff the file contained at least one valid row AND
    there are zero findings. An empty file is a finding shape
    (``empty`` code, ``line_no=0``) — same convention as the sister
    validators.
    """

    path: str
    n_rows: int
    n_valid: int
    findings: tuple[ValidationFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings and self.n_valid > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_valid": self.n_valid,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate_queries(
    path: str | Path,
    corpus_dir: str | Path | None = None,
) -> ValidationReport:
    """Walk a queries JSONL in collecting mode; see module docstring."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    corpus_root: Path | None = None
    corpus_docs: set[str] | None = None
    if corpus_dir is not None:
        corpus_root = Path(corpus_dir)
        if not corpus_root.exists():
            raise FileNotFoundError(corpus_root)
        # Mirror load_corpus's enumeration EXACTLY — `base.glob("*.md")` keyed by
        # `path.name`, files only — instead of a raw `.exists()` on
        # `corpus_root / expected_doc`. `.exists()` accepts anything on disk under
        # the corpus dir that is never a loaded `Document.filename`: a non-`.md`
        # file (recall silently 0.0 for that query), a directory named `*.md`
        # (load_corpus then crashes on read), or a case-mismatched name on a
        # case-insensitive FS. Comparing against the real document set closes all
        # three — exactly the "typo'd doc reference that invalidates recall" this
        # check exists to catch (#98).
        corpus_docs = {p.name for p in corpus_root.glob("*.md") if p.is_file()}

    findings: list[ValidationFinding] = []
    seen_ids: dict[str, int] = {}
    n_rows = 0
    n_valid = 0

    # utf-8-sig strips a leading BOM, matching load_queries for parity (#93):
    # otherwise a BOM-prefixed file reports a spurious malformed_json finding on
    # line 1 (`.strip()` does not remove U+FEFF).
    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            n_rows += 1
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                findings.append(
                    ValidationFinding(
                        line_no=line_no,
                        reason=f"invalid JSON: {e.msg}",
                        code="malformed_json",
                    )
                )
                continue
            if not isinstance(obj, dict):
                findings.append(
                    ValidationFinding(
                        line_no=line_no,
                        reason=(
                            f"row must be a JSON object with {list(REQUIRED_FIELDS)} fields, "
                            f"got {type(obj).__name__}"
                        ),
                        code="not_an_object",
                    )
                )
                continue

            row_findings = _validate_row(obj, line_no)
            findings.extend(row_findings)

            # The duplicate-id check is independent of the other field checks:
            # a duplicate id is a real, separate finding even when the row also
            # has (say) an empty question, and collecting mode must surface every
            # finding in one pass. Run it whenever the id field is itself valid
            # (present, string, non-empty) — a missing/empty/non-string id is
            # already reported by `_validate_row`, so guarding here avoids a
            # KeyError and avoids registering a junk `seen_ids` entry. A valid id
            # is recorded even when the row has other errors, so a later row
            # reusing it is still flagged.
            # `id_value.strip() != ""`, not `id_value != ""`: a whitespace-only
            # id (`"  "`) is already reported as `empty_id` by `_validate_row`
            # (`value.strip() == ""`, the #92 rule), so it is NOT a valid,
            # registrable id. The old literal-empty guard let it through, which
            # both registered a junk `seen_ids` entry (the exact thing the
            # comment above says it avoids) and emitted a spurious `duplicate_id`
            # finding on a repeat -- flagging as "duplicate" a value already
            # flagged as "empty". Stripping keeps the duplicate check in lockstep
            # with `_validate_row`'s emptiness check (#102).
            row_has_duplicate = False
            id_value = obj.get("id")
            if isinstance(id_value, str) and id_value.strip() != "":
                if id_value in seen_ids:
                    row_has_duplicate = True
                    findings.append(
                        ValidationFinding(
                            line_no=line_no,
                            reason=(
                                f"duplicate id {id_value!r}; first seen at line "
                                f"{seen_ids[id_value]}; query id must be unique within a file"
                            ),
                            code="duplicate_id",
                        )
                    )
                else:
                    seen_ids[id_value] = line_no

            # Only a fully clean row proceeds to the corpus check and n_valid.
            if row_findings or row_has_duplicate:
                continue

            if corpus_docs is not None:
                expected_doc = obj["expected_doc"]
                if expected_doc not in corpus_docs:
                    findings.append(
                        ValidationFinding(
                            line_no=line_no,
                            reason=(
                                f"expected_doc {expected_doc!r} is not a loaded corpus "
                                f"document (*.md file) under {corpus_root}"
                            ),
                            code="expected_doc_not_found",
                        )
                    )
                    # The row is well-formed JSONL-wise but invalid against
                    # the corpus; don't count toward n_valid so the operator
                    # sees a non-zero finding total without ambiguity.
                    continue

            n_valid += 1

    if n_rows == 0 and not findings:
        findings.append(
            ValidationFinding(
                line_no=0,
                reason=f"queries file {path} contains no rows",
                code="empty",
            )
        )

    return ValidationReport(
        path=str(path),
        n_rows=n_rows,
        n_valid=n_valid,
        findings=tuple(findings),
    )


def _validate_row(obj: dict[str, Any], line_no: int) -> list[ValidationFinding]:
    """Schema checks for the four required fields, collected per row.

    Multiple independent problems on the same row (e.g., missing ``id``
    AND empty ``question``) surface together so the operator sees them
    in one pass.
    """
    findings: list[ValidationFinding] = []
    for field in REQUIRED_FIELDS:
        if field not in obj:
            findings.append(
                ValidationFinding(
                    line_no=line_no,
                    reason=f"missing required field {field!r}",
                    code=f"missing_{field}",
                )
            )
            continue
        value = obj[field]
        if not isinstance(value, str):
            findings.append(
                ValidationFinding(
                    line_no=line_no,
                    reason=(f"field {field!r} must be a string, got {type(value).__name__}"),
                    code=f"non_string_{field}",
                )
            )
            continue
        if value.strip() == "":
            # `value.strip()`, not `value == ""`: a whitespace-only field is as
            # corrupting as an empty one (an `expected_snippet` of "   " matches
            # any chunk with three consecutive spaces → trivial snippet-hit@k),
            # so it is flagged under the same `empty_<field>` code. Keeps the
            # linter in lockstep with `Query.__post_init__` / `_require_str`,
            # which reject both (chunking-strategies-lab #92, completing #72).
            findings.append(
                ValidationFinding(
                    line_no=line_no,
                    reason=f"field {field!r} must not be empty or whitespace-only",
                    code=f"empty_{field}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# CLI: python -m chunking_lab.validate
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chunking_lab.validate",
        description=(
            "Lint a queries JSONL; report every malformed row in one pass. "
            "Exit 0 clean / 1 findings / 2 I/O error."
        ),
    )
    parser.add_argument("queries", help="Queries JSONL path (e.g. data/queries.jsonl).")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help=(
            "Optional corpus directory. When given, every row's expected_doc "
            "is checked against this directory; missing docs are reported as "
            "'expected_doc_not_found' findings."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the report as JSON instead of the human-readable summary.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Write the rendered output to this path instead of stdout. Parent dirs "
            "are auto-created via chunking_lab/io_utils.atomic_write_text. Parity "
            "with llm-eval-harness validate --out (#66). Findings still print to "
            "stderr in human-readable mode even when --out is set, so the operator's "
            "diagnostic channel is preserved."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = validate_queries(args.queries, corpus_dir=args.corpus_dir)
    except FileNotFoundError as e:
        sys.stderr.write(f"file not found: {e}\n")
        return 2
    except OSError as e:
        sys.stderr.write(f"failed to read {args.queries}: {e}\n")
        return 2

    if args.as_json:
        rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    else:
        # Findings go to stderr regardless of --out so the operator's diagnostic
        # channel is preserved even when stdout is captured to a file. Parity
        # with llm-eval-harness validate (#66) and the sibling --out subcommands.
        for finding in report.findings:
            line_label = f"line {finding.line_no}" if finding.line_no else "file"
            sys.stderr.write(f"{line_label} [{finding.code}]: {finding.reason}\n")
        status = "ok" if report.ok else "fail"
        rendered = (
            f"{status}: {args.queries} rows={report.n_rows} valid={report.n_valid} "
            f"findings={len(report.findings)}\n"
        )
    if args.out:
        atomic_write_text(args.out, rendered)
    else:
        sys.stdout.write(rendered)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
