"""Type-checking gate for ``chunking_lab`` (#164, D-013).

The in-repo half of the contract: it runs the configured ``mypy`` gate over the
package and asserts it exits clean, so an annotation that drifts out of shape
fails a *test* — not only the (separately wired) CI ``mypy`` step.

Why this repo needs it, stated differently from the two siblings that already
have one. ``llm-eval-harness`` (D-016) and ``llm-cost-optimizer`` (D-014) both
justify their gate by shipping a ``py.typed`` marker, so their annotations are
visible to downstream type-checkers and could silently drift. ``chunking_lab``
ships no such marker. The justification here is the **latent-green** one: the
annotations exist, nothing machine-checked them, and #164 is the proof — two
``no-redef`` errors sat in ``strategies/recursive.py`` unnoticed, on a repo
whose CI was green the entire time, because no gate ran.

``mypy`` is invoked with **no arguments** so it reads exactly the
``[tool.mypy]`` block in ``pyproject.toml``. That keeps three things in
lockstep that would otherwise drift apart: this test, the CI step, and a
developer's bare ``mypy`` in a terminal. A test that passed its own file list
would be testing a scope nothing else uses.

Skipped (not failed) when mypy isn't importable, so a minimal environment
without the ``dev`` extra can still run the rest of the suite; CI installs
``.[dev]`` so the gate is always exercised there.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mypy_reports_no_issues() -> None:
    pytest.importorskip("mypy", reason="mypy not installed (dev extra); CI installs it")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "mypy gate failed — an annotation in chunking_lab drifted from the "
        "code. Output:\n" + proc.stdout + proc.stderr
    )


def test_mypy_config_covers_the_package_and_scripts() -> None:
    """`files` must name both roots, so a bare `mypy` checks both.

    Without a `files` key, a bare `mypy` exits 2 with "no files or directories
    to check" — which `test_mypy_reports_no_issues` would report as a failure,
    but a future edit setting `files = []` would make it *pass vacuously*.

    `scripts` joined the scope in #165 (D-014). It had been excluded because
    `mypy chunking_lab scripts` could not start at all, so nobody knew whether
    it was clean; it wasn't, by one dead `type: ignore`.
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy_cfg = config["tool"]["mypy"]
    assert mypy_cfg["files"] == ["chunking_lab", "scripts"]
    for root in mypy_cfg["files"]:
        assert (_REPO_ROOT / root).is_dir(), f"{root} in mypy files= does not exist"


def test_the_package_base_config_that_lets_scripts_be_checked_is_present() -> None:
    """The two keys that make `scripts/run_matrix.py` map to one module name.

    Without them mypy resolves that file as both `run_matrix` and
    `scripts.run_matrix` and stops with "Source file found twice", checking
    *nothing* — including `chunking_lab`. Dropping either key silently reverts
    the whole gate to that state, which exits non-zero rather than passing, but
    the failure would read as an unrelated config error rather than as this.
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy_cfg = config["tool"]["mypy"]
    assert mypy_cfg["explicit_package_bases"] is True
    assert mypy_cfg["mypy_path"] == "."


def test_the_gate_actually_checks_source_files() -> None:
    """Anti-vacuous: a clean exit means nothing if nothing was checked.

    `mypy` prints "Success: no issues found in N source files"; assert N > 1 so
    a config that narrowed to a single stub file, or to none, is caught.
    """
    pytest.importorskip("mypy", reason="mypy not installed (dev extra); CI installs it")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    import re

    match = re.search(r"no issues found in (\d+) source file", proc.stdout)
    assert match is not None, f"unexpected mypy output: {proc.stdout!r}"
    assert int(match.group(1)) > 1, (
        f"mypy checked only {match.group(1)} source file(s) — the gate is "
        "effectively vacuous. Check `files` in [tool.mypy]."
    )


def test_no_blanket_ignore_missing_imports() -> None:
    """A blanket `ignore_missing_imports` would silence a *typo'd* import too.

    The optional `sentence_transformers` dependency is handled by a per-module
    override instead, which is narrow enough to leave real mistakes visible.
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy_cfg = config["tool"]["mypy"]
    assert "ignore_missing_imports" not in mypy_cfg, (
        "a top-level ignore_missing_imports silences typo'd imports as well as "
        "optional ones; use a [[tool.mypy.overrides]] block per module"
    )
    overrides = mypy_cfg.get("overrides", [])
    modules = {o.get("module") for o in overrides}
    assert "sentence_transformers.*" in modules


def test_warn_unused_ignores_is_on_so_a_stale_ignore_cannot_linger() -> None:
    """This is what let the inline `# type: ignore[import-not-found]` in
    `embedder.py` be removed rather than kept: with the flag on, that ignore
    would itself become an error the moment someone installs the `[sbert]`
    extra, so an inline ignore is a time bomb where an override is not."""
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["mypy"]["warn_unused_ignores"] is True
    embedder = (_REPO_ROOT / "chunking_lab" / "embedder.py").read_text(encoding="utf-8")
    assert "type: ignore[import-not-found]" not in embedder


def test_ci_lint_job_runs_the_gate() -> None:
    """The test and the CI step must both exist — either alone is a gate that
    can be bypassed by running the other."""
    yaml = pytest.importorskip("yaml", reason="pyyaml not installed (dev extra)")
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    runs = [
        step.get("run", "") for step in workflow["jobs"]["lint"]["steps"] if isinstance(step, dict)
    ]
    assert any(r.strip() == "mypy" for r in runs), (
        f"the CI lint job does not run `mypy`; steps were {runs!r}"
    )
