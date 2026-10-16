"""A bad dataset is an `::error::` line at exit 2 in `run_matrix.py` (#275).

#149 put the script on the 0 / 1 / 2 contract for its flags and its write path.
The two data loads were left outside `_fail()`, so on main a bad row in
`data/queries.jsonl` or a non-UTF-8 corpus document escaped as a traceback at
exit 1, the code `test_no_operator_input_failure_exits_one` says is reserved.

The loaders read module-level default paths, so each test points them at a temp
copy of `data/` and never touches the committed files.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab import corpus as corpus_mod  # noqa: E402
from chunking_lab import queries as queries_mod  # noqa: E402
from scripts.run_matrix import main  # noqa: E402


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "data"
    shutil.copytree(_REPO_ROOT / "data", copy)
    monkeypatch.setattr(corpus_mod, "DEFAULT_CORPUS_DIR", copy / "corpus")
    monkeypatch.setattr(queries_mod, "DEFAULT_QUERIES_PATH", copy / "queries.jsonl")
    return copy


def _run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, Path]:
    out = tmp_path / "results"
    rc = main(["--strategy", "fixed-size", "--results-dir", str(out)])
    return rc, capsys.readouterr().err, out


def _break_row(d: Path) -> None:
    with (d / "queries.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "q99", "question": "x"}\n')


def _non_utf8_query(d: Path) -> None:
    with (d / "queries.jsonl").open("ab") as f:
        f.write(
            b'{"id": "q99", "question": "caf\xe9?", "expected_doc": "a", "expected_snippet": "b"}\n'
        )


def _non_utf8_document(d: Path) -> None:
    (d / "corpus" / "99_bad.md").write_bytes(b"caf\xe9\n")


def _missing_queries(d: Path) -> None:
    (d / "queries.jsonl").unlink()


def _empty_corpus(d: Path) -> None:
    for p in (d / "corpus").iterdir():
        p.unlink()


CASES = {
    "bad-row": (_break_row, "queries.jsonl:13 'expected_doc' must be a string"),
    "non-utf8-query": (_non_utf8_query, "can't decode byte 0xe9"),
    "non-utf8-document": (_non_utf8_document, "corpus document"),
    "missing-queries": (_missing_queries, "queries file not found"),
    "empty-corpus": (_empty_corpus, "no markdown documents"),
}


@pytest.mark.parametrize(("breakage", "needle"), list(CASES.values()), ids=list(CASES))
def test_a_bad_dataset_exits_two_with_one_error_line(
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    breakage: Callable[[Path], None],
    needle: str,
) -> None:
    breakage(data)
    rc, err, out = _run(tmp_path, capsys)
    assert rc == 2
    lines = err.strip().splitlines()
    assert len(lines) == 1, err
    assert lines[0].startswith("::error::failed to load the dataset: ")
    assert needle in lines[0]
    assert "python -m chunking_lab.validate" in lines[0]
    assert not out.exists()


def test_the_copied_dataset_still_runs_clean(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Control: the fixture's copy is a working dataset, so each failure above
    # is the breakage and not the redirection.
    rc, err, out = _run(tmp_path, capsys)
    assert rc == 0, err
    assert [p.name for p in out.iterdir() if p.suffix == ".json"]
