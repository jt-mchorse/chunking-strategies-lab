"""A lone surrogate in any query field is refused by the loader and the linter (#216).

`\\ud800` is valid JSON escape syntax, so a row carrying one loaded and
validated clean. Measured on `d11fcb3`: in `expected_snippet` snippet-hit@5
went 1.0 -> 0.0 with no warning (documents are strict UTF-8, so it can never
match); in `question` the run died with `UnicodeEncodeError` from the embedder.
`_is_invisible` covers Cf and Cc; a surrogate is Cs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chunking_lab.queries import Query, load_queries
from chunking_lab.validate import main, validate_queries

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data" / "queries.jsonl"
FIELDS = ["id", "question", "expected_doc", "expected_snippet"]
SURROGATES = ["\ud800", "\udbff", "\udc00", "\udfff"]


def _valid() -> dict[str, str]:
    return json.loads(QUERIES.read_text(encoding="utf-8-sig").splitlines()[0])


def _jsonl_with(tmp_path: Path, field: str, ch: str) -> Path:
    row = _valid()
    row[field] = row[field] + ch
    p = tmp_path / "q.jsonl"
    # json.dumps escapes the surrogate as \\udXXX, exactly the on-disk shape.
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return p


@pytest.mark.parametrize("ch", SURROGATES, ids=[f"U+{ord(c):04X}" for c in SURROGATES])
@pytest.mark.parametrize("field", FIELDS)
def test_query_refuses_a_lone_surrogate_in_every_field(field: str, ch: str) -> None:
    row = _valid()
    row[field] = row[field] + ch
    with pytest.raises(ValueError, match=f"lone surrogate U\\+{ord(ch):04X}"):
        Query(**row)


@pytest.mark.parametrize("field", FIELDS)
def test_the_loader_and_the_linter_agree(field: str, tmp_path: Path) -> None:
    path = _jsonl_with(tmp_path, field, "\ud800")
    with pytest.raises(ValueError, match="lone surrogate"):
        load_queries(path)
    report = validate_queries(path)
    codes = [f.code for f in report.findings]
    assert codes == [f"unencodable_char_{field}"], codes


def test_the_cli_reports_it_as_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([str(_jsonl_with(tmp_path, "expected_snippet", "\ud800")), "--json"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "unencodable_char_expected_snippet" in out


def test_a_directional_mark_in_question_is_still_legal() -> None:
    """`question`'s exemption from the invisible-character rule stands: an RTL
    mark is a character; a surrogate is not."""
    row = _valid()
    row["question"] = "‏" + row["question"]
    Query(**row)


def test_a_real_astral_character_is_not_a_surrogate() -> None:
    """Control: a supplementary-plane character is one codepoint in Python, not
    a surrogate pair, and must stay legal."""
    row = _valid()
    row["question"] = row["question"] + " \U0001f600"
    Query(**row)


def test_the_shipped_queries_are_clean() -> None:
    assert list(validate_queries(QUERIES).findings) == []
