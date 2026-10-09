"""`docs/architecture.md`'s finding-code paragraph matches what the validator emits (#252).

The doc said "Seventeen finding codes" -- right after #88, which corrected the
count by hand -- and then #162/#171 (`invisible_char_<field>`) and #216
(`unencodable_char_<field>`) added seven more without the doc changing,
because nothing tied the two together. The validator emitted 24.

The population is discovered, not listed: every `code=` template is read out
of `chunking_lab/validate.py`'s source, and a crafted file has to trigger each
one. A new code family therefore fails here twice over -- untriggered by the
fixture, and absent from the doc -- rather than shipping silently as the last
two did.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.validate import REQUIRED_FIELDS, validate_queries  # noqa: E402

_VALIDATE_SRC = _REPO_ROOT / "chunking_lab" / "validate.py"
_ARCH_DOC = _REPO_ROOT / "docs" / "architecture.md"

_NUMBER_WORDS = {
    word: n
    for n, word in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
for _n, _unit in enumerate(
    ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"], start=1
):
    _NUMBER_WORDS[f"twenty-{_unit}"] = 20 + _n

_FIELD_SUFFIX = re.compile(rf"_({'|'.join(map(re.escape, REQUIRED_FIELDS))})$")


def _template(code: str) -> str:
    """`missing_expected_doc` -> `missing_<field>`; a fieldless code is itself.

    `duplicate_id` ends in a field name and is not a per-field family, so the
    fieldless codes the source writes literally are kept whole.
    """
    if code in {t for t in _source_templates() if "<field>" not in t}:
        return code
    return _FIELD_SUFFIX.sub("_<field>", code)


def _source_templates() -> set[str]:
    templates = set(re.findall(r'code=f?"([a-z_{}]+)"', _VALIDATE_SRC.read_text("utf-8")))
    return {t.replace("{field}", "<field>") for t in templates}


def _emitted_codes(tmp_path: Path) -> set[str]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "real.md").write_text("# Real\n\nbody\n", encoding="utf-8")
    zw, lone = "​", "\ud800"
    rows = [
        "{bad",
        "[1]",
        json.dumps({}),
        json.dumps({"id": 1, "question": 2, "expected_doc": 3, "expected_snippet": 4}),
        json.dumps({"id": " ", "question": " ", "expected_doc": " ", "expected_snippet": " "}),
        json.dumps(
            {
                "id": "a" + zw,
                "question": "q",
                "expected_doc": "d" + zw,
                "expected_snippet": "s" + zw,
            }
        ),
        json.dumps(
            {
                "id": "b" + lone,
                "question": "q" + lone,
                "expected_doc": "d" + lone,
                "expected_snippet": "s" + lone,
            }
        ),
        json.dumps(
            {"id": "c", "question": "q", "expected_doc": "nope.md", "expected_snippet": "s"}
        ),
        json.dumps(
            {"id": "c", "question": "q", "expected_doc": "real.md", "expected_snippet": "s"}
        ),
    ]
    rows_path = tmp_path / "q.jsonl"
    rows_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    codes = {f.code for f in validate_queries(rows_path, corpus_dir=corpus).findings}
    return codes | {f.code for f in validate_queries(empty_path).findings}


def _doc_paragraph() -> str:
    text = _ARCH_DOC.read_text("utf-8")
    start = text.index("**Pre-flight validator (#37).**")
    return text[start : text.index("\n\n", start)]


def test_the_fixture_triggers_every_code_template_in_the_source(tmp_path: Path) -> None:
    """Anti-vacuity: the emitted population covers every `code=` the source can write."""
    source = _source_templates()
    assert len(source) >= 10, source  # the scan found the call sites at all
    assert {_template(c) for c in _emitted_codes(tmp_path)} == source


def test_the_doc_states_the_emitted_count(tmp_path: Path) -> None:
    emitted = _emitted_codes(tmp_path)
    match = re.search(r"\b([A-Za-z-]+|\d+) finding codes\b", _doc_paragraph())
    assert match, "the validator paragraph no longer states a finding-code count"
    word = match.group(1).lower()
    stated = int(word) if word.isdigit() else _NUMBER_WORDS[word]
    assert stated == len(emitted), (
        f"docs/architecture.md says {match.group(1)!r} finding codes; the validator emits "
        f"{len(emitted)}: {sorted(emitted)}"
    )


def test_the_doc_names_every_emitted_code_family(tmp_path: Path) -> None:
    named = set(re.findall(r"`([a-z_]+(?:<field>)?)`", _doc_paragraph()))
    missing = sorted({_template(c) for c in _emitted_codes(tmp_path)} - named)
    assert not missing, f"docs/architecture.md never names: {missing}"
