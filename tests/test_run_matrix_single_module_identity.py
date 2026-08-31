"""One file must be one module (#165, D-014).

`scripts/run_matrix.py` was imported by the suite under two different names —
`run_matrix` (after a `sys.path` insert) and `scripts.run_matrix`. Python
treats those as unrelated modules: the file is executed twice, gets two
`__dict__`s, and a `monkeypatch.setattr` on one is invisible to the other.

Measured before the fix, in one interpreter:

    same module object?                                 False
    run_matrix.__name__ = run_matrix                    file: run_matrix.py
    scripts.run_matrix.__name__ = scripts.run_matrix    file: run_matrix.py
    patch on scripts.run_matrix visible via run_matrix? False

Nothing was failing, because each group of tests happened to patch and call the
same copy. That is a property of which tests exist, not one anyone enforced —
so it is pinned here. It is also exactly what mypy's "Source file found twice
under different module names" was reporting, which is why the gate could not
start; that error was a true finding, not a layout quirk to configure away.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _REPO_ROOT / "tests"
_SCRIPT = _REPO_ROOT / "scripts" / "run_matrix.py"

#: The one spelling. `scripts.run_matrix` rather than bare `run_matrix`
#: because it needs no `sys.path` mutation and it is what the majority of the
#: suite already used.
_CANONICAL = "scripts.run_matrix"


def _imported_names_of_run_matrix(path: Path) -> set[str]:
    """Every module name this test file reaches `run_matrix` by."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"run_matrix", _CANONICAL}:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"run_matrix", _CANONICAL}:
                    found.add(alias.name)
    return found


def test_the_suite_imports_run_matrix_by_exactly_one_name() -> None:
    by_name: dict[str, list[str]] = {}
    for path in sorted(_TESTS.glob("test_*.py")):
        for name in _imported_names_of_run_matrix(path):
            by_name.setdefault(name, []).append(path.name)

    # Anti-vacuous: an AST walk that matched nothing would satisfy a subset
    # check. Several test files really do import it.
    assert by_name, "no test file imports run_matrix — this guard found nothing to check"
    assert set(by_name) == {_CANONICAL}, (
        f"run_matrix is imported under more than one module name: {by_name}. "
        "Python makes each name a separate module object, so a monkeypatch on "
        "one cannot reach the other."
    )


def _mutates_sys_path_with_scripts(path: Path) -> bool:
    """True when the file *calls* `sys.path.insert/append` mentioning "scripts".

    Matched structurally rather than by substring. A substring search finds its
    own needle in this file's source and reports the guard as the offender —
    which is what the first draft did. Walking the AST looks for a real call,
    so the equivalent code that appears inside a string literal below (handed
    to a subprocess on purpose) is correctly not a match.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"insert", "append"}:
            continue
        target = node.func.value
        is_sys_path = (
            isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        )
        if is_sys_path and "scripts" in ast.dump(node):
            return True
    return False


def test_no_test_file_puts_the_scripts_dir_on_sys_path() -> None:
    """The mechanism that made the bare `run_matrix` name importable at all.

    Leaving it behind would let the duplicate come back without any import
    statement changing, since a later `import run_matrix` anywhere would then
    resolve.
    """
    offenders = [
        path.name
        for path in sorted(_TESTS.glob("test_*.py"))
        if _mutates_sys_path_with_scripts(path)
    ]
    assert offenders == [], f"these files re-add scripts/ to sys.path: {offenders}"


def test_importing_it_both_ways_really_does_produce_two_modules() -> None:
    """Pins *why* the rule above exists, rather than asserting the rule twice.

    Run in a subprocess so this test cannot leave a second copy of the module
    in this session's `sys.modules` — which would be the very hazard it
    documents.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "sys.path.insert(0, r'{root}');"
            "sys.path.insert(0, r'{scripts}');"
            "import run_matrix, scripts.run_matrix as s;"
            "print(run_matrix is s)".format(root=_REPO_ROOT, scripts=_REPO_ROOT / "scripts"),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        "the two import spellings resolved to the same module object; if Python "
        "ever unified them, the guard above is no longer load-bearing and this "
        "test's rationale needs rewriting"
    )


@pytest.mark.parametrize(
    ("argv", "label"),
    [
        (["scripts/run_matrix.py", "--help"], "direct-path"),
        (["-m", "scripts.run_matrix", "--help"], "dash-m"),
    ],
)
def test_both_invocation_styles_still_work(argv: list[str], label: str) -> None:
    """#165 asks for this explicitly: the resolution must not break either one."""
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=_REPO_ROOT)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}{proc.stderr}"
    assert "--results-dir" in proc.stdout, f"{label} did not print the real usage"


def test_scripts_has_no_init_py() -> None:
    """The resolution deliberately taken, recorded as an assertion.

    Adding `scripts/__init__.py` was the other option #165 listed. It would not
    have removed the duplicate — only the import normalization does that — and
    it changes how `python scripts/run_matrix.py` resolves. If a later change
    wants it, this test is the place that argument gets re-made.
    """
    assert not (_REPO_ROOT / "scripts" / "__init__.py").exists()
    assert _SCRIPT.is_file()
