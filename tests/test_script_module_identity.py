"""One file must be one module, for every script directory (#165 / D-014, #174).

`scripts/run_matrix.py` was imported by the suite under two different names —
`run_matrix` (after a `sys.path` insert) and `scripts.run_matrix`. Python
treats those as unrelated modules: the file is executed twice, gets two
`__dict__`s, and a `monkeypatch.setattr` on one is invisible to the other.

Measured before that fix, in one interpreter:

    same module object?                                 False
    run_matrix.__name__ = run_matrix                    file: run_matrix.py
    scripts.run_matrix.__name__ = scripts.run_matrix    file: run_matrix.py
    patch on scripts.run_matrix visible via run_matrix? False

Nothing was failing, because each group of tests happened to patch and call the
same copy. That is a property of which tests exist, not one anyone enforced —
so it is pinned here. It is also exactly what mypy's "Source file found twice
under different module names" was reporting, which is why the gate could not
start; that error was a true finding, not a layout quirk to configure away.

**#174 widened this file, and the reason is the point.** As written it named
one module and one directory: `_CANONICAL = "scripts.run_matrix"`, and a
sys.path guard matching the literal string `"scripts"`. `notebooks/` is a
second non-package directory of runnable Python, `notebooks/_build_notebook.py`
was imported bare from two test files after a `sys.path` insert, and this guard
could not see any of it — the same class, one directory over, left standing by
a fix whose own scope read like a survey. The directories and their modules are
now *discovered* from what git tracks, so a third one is covered on the day it
is added rather than on the day someone remembers this file.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _REPO_ROOT / "tests"

#: The importable package and the test package are not "script directories" —
#: they are already packages, and importing them by one name is not in
#: question. Everything else git tracks Python under is.
_PACKAGE_DIRS = {"chunking_lab", "tests"}


def _tracked_python_dirs() -> list[str]:
    """Top-level directories git tracks `.py` files under, minus the packages.

    Discovered rather than listed. A hand-written list is how this guard came
    to cover `scripts/` and not `notebooks/` in the first place. Reading from
    git rather than the filesystem keeps a local `test_venv/` or a build
    artifact out of it.
    """
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    tops = {Path(rel).parts[0] for rel in out if len(Path(rel).parts) > 1}
    return sorted(tops - _PACKAGE_DIRS)


SCRIPT_DIRS = _tracked_python_dirs()

#: `{module_stem: "<dir>.<stem>"}` — the one canonical spelling per module.
#: Package-qualified, because it needs no `sys.path` mutation and it is what
#: the majority of the suite already used.
CANONICAL: dict[str, str] = {
    path.stem: f"{d}.{path.stem}"
    for d in SCRIPT_DIRS
    for path in sorted((_REPO_ROOT / d).glob("*.py"))
}


def test_the_discovery_found_both_known_script_directories() -> None:
    """Anti-vacuous arm for everything below.

    A `_tracked_python_dirs` that returned `[]` — after a `git ls-files` flag
    change, or a layout move — would make every assertion in this file
    vacuously true. Both directories that exist today are named explicitly, so
    losing one is loud.
    """
    assert set(SCRIPT_DIRS) >= {"scripts", "notebooks"}, SCRIPT_DIRS
    assert {"run_matrix", "_build_notebook"} <= set(CANONICAL), CANONICAL


def _names_reaching_a_script_module(path: Path) -> set[str]:
    """Every module name *path* reaches a script module by, raw spelling kept.

    `run_matrix` and `scripts.run_matrix` must stay distinguishable — that
    distinction is the whole point.
    """
    found: set[str] = set()
    bare = set(CANONICAL)
    qualified = set(CANONICAL.values())
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in SCRIPT_DIRS:
                # `from notebooks import _build_notebook` reaches it by the
                # qualified name even though `node.module` is the directory.
                found.update(f"{node.module}.{a.name}" for a in node.names if a.name in bare)
            elif node.module in bare | qualified:
                found.add(node.module)
            elif node.module is not None:
                # `from notebooks._build_notebook import build_notebook`
                head = node.module.rsplit(".", 1)[0]
                if node.module.count(".") == 1 and head in SCRIPT_DIRS:
                    found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in bare | qualified:
                    found.add(alias.name)
    return found


def test_the_suite_imports_each_script_module_by_exactly_one_name() -> None:
    by_module: dict[str, set[str]] = {}
    for path in sorted(_TESTS.glob("test_*.py")):
        for name in _names_reaching_a_script_module(path):
            by_module.setdefault(name.rsplit(".", 1)[-1], set()).add(name)

    # Anti-vacuous: an AST walk that matched nothing would satisfy a subset
    # check. Several test files really do import these.
    assert by_module, "no test file imports any script module — this guard found nothing"
    assert {"run_matrix", "_build_notebook"} <= set(by_module), sorted(by_module)

    offenders = {stem: sorted(names) for stem, names in by_module.items() if len(names) > 1}
    assert offenders == {}, (
        f"these modules are imported under more than one name: {offenders}. "
        "Python makes each name a separate module object, so a monkeypatch on "
        "one cannot reach the other."
    )
    wrong = {stem: sorted(names) for stem, names in by_module.items() if names != {CANONICAL[stem]}}
    assert wrong == {}, f"these modules are imported by a non-canonical name: {wrong}"


def _mutates_sys_path_with_a_script_dir(path: Path) -> str | None:
    """The script directory *path* **calls** `sys.path.insert/append` with, if any.

    Matched structurally rather than by substring. A substring search finds its
    own needle in this file's source and reports the guard as the offender —
    which is what the first draft did. Walking the AST looks for a real call,
    so the equivalent code inside the string literal handed to a subprocess
    below is correctly not a match.
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
        if not is_sys_path:
            continue
        dumped = ast.dump(node)
        for d in SCRIPT_DIRS:
            if f"'{d}'" in dumped or f'"{d}"' in dumped:
                return d
    return None


def test_no_test_file_puts_a_script_dir_on_sys_path() -> None:
    """The mechanism that makes a bare name importable at all.

    Leaving it behind lets the duplicate come back without any import statement
    changing, since a later bare `import <stem>` anywhere would then resolve.
    Inserting the **repo root** is fine and is not matched here — that is what
    makes the canonical `<dir>.<stem>` spelling work.
    """
    offenders = {
        path.name: d
        for path in sorted(_TESTS.glob("test_*.py"))
        if (d := _mutates_sys_path_with_a_script_dir(path)) is not None
    }
    assert offenders == {}, f"these files re-add a script dir to sys.path: {offenders}"


@pytest.mark.parametrize("stem", sorted(CANONICAL))
def test_importing_it_both_ways_really_does_produce_two_modules(stem: str) -> None:
    """Pins *why* the rules above exist, rather than asserting the rule twice.

    Run in a subprocess so this test cannot leave a second copy of the module
    in this session's `sys.modules` — which would be the very hazard it
    documents.
    """
    qualified = CANONICAL[stem]
    directory = _REPO_ROOT / qualified.split(".", 1)[0]
    code = (
        "import sys;"
        f"sys.path.insert(0, r'{_REPO_ROOT}');"
        f"sys.path.insert(0, r'{directory}');"
        f"import {stem}; import {qualified} as q;"
        f"print({stem} is q)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        "the two import spellings resolved to the same module object; if Python "
        "ever unified them, the guards above are no longer load-bearing and this "
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
    """#165 asks for this explicitly: the resolution must not break either one.

    Only `run_matrix` is exercised: it is the one with a CLI.
    `notebooks/_build_notebook.py` has no argument parser — it *writes the
    notebook* when run, so calling it with `--help` would silently regenerate a
    committed artifact rather than print usage.
    """
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=_REPO_ROOT)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}{proc.stderr}"
    assert "--results-dir" in proc.stdout, f"{label} did not print the real usage"


@pytest.mark.parametrize("directory", SCRIPT_DIRS)
def test_script_dirs_have_no_init_py(directory: str) -> None:
    """The resolution deliberately taken, recorded as an assertion.

    Adding an `__init__.py` was the other option #165 listed. It would not have
    removed the duplicate — only the import normalization does that — and it
    changes how `python scripts/run_matrix.py` resolves. If a later change
    wants it, this test is the place that argument gets re-made.
    """
    assert not (_REPO_ROOT / directory / "__init__.py").exists()
    assert list((_REPO_ROOT / directory).glob("*.py")), f"{directory} has no modules"
