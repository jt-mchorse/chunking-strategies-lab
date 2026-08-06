"""`scripts/run_matrix.py` honours the repo's 0 / 1 / 2 exit contract (#149).

`docs/architecture.md` states that contract, and #124–#127 delivered it on
`chunking_lab.validate`. This script — the one the README leads with, and the
one that actually writes five JSONs and a summary — never received it, so every
operator-input mistake escaped as a raw traceback at exit 1:

- `--ks ''` / `abc` / `1,-3` / `0`
- `--embedder minilm` without the `sbert` extra (whose *one* guard was dead
  code: `MiniLMEmbedder` is unconditionally importable, so the `except
  ImportError` around the import could never fire, and `SystemExit(<str>)`
  exits 1 anyway)
- an unwritable `--results-dir`

Every test here pins the **message**, not just the code. `2` has several causes
in this script, so a code-only assertion would pass for the wrong reason — the
trap embedding-model-shootout#112 hit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_matrix import main  # noqa: E402


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    rc = main(argv)
    return rc, capsys.readouterr().err


@pytest.mark.parametrize(
    ("ks", "expected"),
    [
        ("", "ks must be non-empty"),
        ("abc", "comma-separated list of integers"),
        ("1,-3", "every k in ks must be positive"),
        ("0", "every k in ks must be positive"),
        ("1,0,5", "every k in ks must be positive"),
        ("3.5", "comma-separated list of integers"),
    ],
)
def test_bad_ks_exits_two_with_an_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], ks: str, expected: str
) -> None:
    rc, err = _run(capsys, ["--ks", ks, "--strategy", "fixed-size", "--results-dir", str(tmp_path)])
    assert rc == 2, f"--ks {ks!r} should exit 2"
    assert err.startswith("::error::"), err
    assert expected in err
    assert "Traceback" not in err
    assert list(tmp_path.iterdir()) == [], "a rejected run must not write artifacts"


def test_bad_ks_is_rejected_before_any_work_happens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--ks` is pure string parsing, so the operator learns the flag is wrong
    before waiting on an embedder build, a corpus load and five evaluations."""
    import scripts.run_matrix as rm

    def _boom(*_a, **_k):  # pragma: no cover - must never be reached
        raise AssertionError("corpus was loaded despite an invalid --ks")

    monkeypatch.setattr(rm, "load_corpus", _boom)
    rc, err = _run(capsys, ["--ks", "0", "--results-dir", str(tmp_path)])
    assert rc == 2
    assert "every k in ks must be positive" in err


def test_ks_rules_are_the_library_s_own_not_a_second_copy() -> None:
    """The CLI pre-flights `--ks` through `metrics.validate_ks` — the same
    function `evaluate_strategy` calls — so the two can't drift apart. A second
    copy of the rule in the CLI would just be a second thing to keep in sync.
    """
    import inspect

    import scripts.run_matrix as rm
    from chunking_lab.metrics import validate_ks

    assert rm.validate_ks is validate_ks
    src = inspect.getsource(rm.main)
    assert "validate_ks(ks)" in src
    # The rules themselves must not be restated in the CLI.
    assert "must be non-empty" not in src
    assert "must be positive" not in src


def test_minilm_without_the_extra_exits_two_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dead-guard case. `MiniLMEmbedder` imports fine without `sbert`; the
    ImportError comes from its *constructor*, one line past the old `try`.
    Simulated here so the test runs identically with or without the extra
    installed.
    """
    import chunking_lab.embedder as emb_mod

    class _NeedsExtra:
        def __init__(self, *_a, **_k):
            raise ImportError(
                "MiniLMEmbedder requires the 'sbert' extra. "
                "Install with: pip install 'chunking-strategies-lab[sbert]'"
            )

    monkeypatch.setattr(emb_mod, "MiniLMEmbedder", _NeedsExtra, raising=False)
    rc, err = _run(
        capsys,
        ["--embedder", "minilm", "--strategy", "fixed-size", "--results-dir", str(tmp_path)],
    )
    assert rc == 2, "the sbert-extra failure must exit 2, not 1"
    assert "requires the `[sbert]` extra" in err
    assert "Traceback" not in err


def test_the_import_guard_alone_would_not_have_caught_it() -> None:
    """Pins *why* the old guard was dead, so a future refactor can't quietly
    put the `try` back around the import only: `MiniLMEmbedder` is importable
    without the extra, by design.
    """
    from chunking_lab.embedder import MiniLMEmbedder

    assert MiniLMEmbedder is not None


def test_unwritable_results_dir_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path component that is a file — the same shape #126 closed for
    `validate.py --out`, on the script that writes six artifacts."""
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory\n", encoding="utf-8")
    rc, err = _run(capsys, ["--strategy", "fixed-size", "--results-dir", str(blocker / "sub")])
    assert rc == 2
    assert "failed to create results dir" in err
    assert "Traceback" not in err


def test_a_valid_run_still_exits_zero(tmp_path: Path) -> None:
    """The contract only tightened on failure paths; the happy path is
    untouched and still writes its artifact."""
    rc = main(["--strategy", "fixed-size", "--results-dir", str(tmp_path)])
    assert rc == 0
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".json"]


def test_no_operator_input_failure_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sweep: exit 1 is reserved (it collides with a "findings" code in the
    sibling validators), so no bad-flag combination may return it. Catches a
    future flag that reopens the gap this issue closed.
    """
    blocker = tmp_path / "f"
    blocker.write_text("x\n", encoding="utf-8")
    bad_invocations = [
        ["--ks", "", "--results-dir", str(tmp_path / "a")],
        ["--ks", "abc", "--results-dir", str(tmp_path / "b")],
        ["--ks", "-1", "--results-dir", str(tmp_path / "c")],
        ["--results-dir", str(blocker / "sub")],
    ]
    for argv in bad_invocations:
        rc, _ = _run(capsys, [*argv, "--strategy", "fixed-size"])
        assert rc == 2, f"{argv} returned {rc}; exit 1 is reserved"
