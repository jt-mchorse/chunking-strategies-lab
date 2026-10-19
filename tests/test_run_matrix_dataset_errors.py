"""A bad dataset is an `::error::` line at exit 2 in `run_matrix.py` (#275).

#149 put the script on the 0 / 1 / 2 contract for its flags and its write path.
The two data loads were left outside `_fail()`, so on main a bad row in
`data/queries.jsonl` or a non-UTF-8 corpus document escaped as a traceback at
exit 1, the code `test_no_operator_input_failure_exits_one` says is reserved.

The loaders read module-level default paths, so each test points them at a temp
copy of `data/` and never touches the committed files.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
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


def _append_row(d: Path, field: str, ch: str) -> None:
    # `json.dumps` escapes the character, so the file stays valid UTF-8 and the
    # row reaches `Query.__post_init__` (#281).
    row = {"id": "q99", "question": "x?", "expected_doc": "01_hnsw.md", "expected_snippet": "HNSW"}
    row[field] += ch
    with (d / "queries.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _invisible_char_row(d: Path) -> None:
    _append_row(d, "expected_doc", "\u200b")


def _lone_surrogate_row(d: Path) -> None:
    _append_row(d, "question", "\ud800")


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
    "invisible-char-row": (_invisible_char_row, "queries.jsonl:13: expected_doc contains"),
    "lone-surrogate-row": (_lone_surrogate_row, "queries.jsonl:13: question contains"),
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


# --- the hint is runnable as printed and checks what failed (#283) ----------

_HINT_MARK = "(check it with: "


def _validate_argv(d: Path) -> list[str]:
    """The validator run that checks everything `run_matrix` loads from *d*."""
    return [
        "python",
        "-m",
        "chunking_lab.validate",
        str(d / "queries.jsonl"),
        "--corpus-dir",
        str(d / "corpus"),
    ]


def _hint_argv(err: str) -> list[str]:
    """The command the `::error::` line tells the operator to run, as argv."""
    line = err.strip().splitlines()[-1]
    assert _HINT_MARK in line, line
    return shlex.split(line.rsplit(_HINT_MARK, 1)[1].removesuffix(")"))


def _run_validator(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # `python` is this interpreter; the cwd is NOT the repo root, so a hint that
    # only works from there (a relative `data/...`) fails here.
    assert argv[0] == "python", argv
    env = dict(os.environ, PYTHONPATH=str(_REPO_ROOT))
    return subprocess.run(
        [sys.executable, *argv[1:]], cwd=cwd, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(("breakage", "needle"), list(CASES.values()), ids=list(CASES))
def test_the_printed_hint_runs_and_refuses_the_same_dataset(
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    breakage: Callable[[Path], None],
    needle: str,
) -> None:
    breakage(data)
    rc, err, _ = _run(tmp_path, capsys)
    assert rc == 2
    argv = _hint_argv(err)
    assert argv == _validate_argv(data)
    proc = _run_validator(argv, tmp_path)
    # 1 = findings, 2 = could not read; 0 would call the refused dataset ok.
    # An argparse usage error is also 2, so require that the validator ran.
    assert proc.returncode in (1, 2), proc
    assert "usage:" not in proc.stderr, proc.stderr


def test_the_validator_run_passes_the_clean_copy(data: Path, tmp_path: Path) -> None:
    # Control: the command the hint must equal is exit 0 on the unbroken copy,
    # so each refusal above is the breakage and not the command.
    proc = _run_validator(_validate_argv(data), tmp_path)
    assert proc.returncode == 0, proc
    assert proc.stdout.startswith("ok: "), proc.stdout
