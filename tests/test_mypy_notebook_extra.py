"""The mypy gate survives the `[notebook]` extra (#208).

With `[notebook]` installed, mypy followed matplotlib's stubs into numpy's,
which use Python 3.12's `type` statement, and stopped on a syntax error under
`python_version = "3.11"` -- so the README's own notebook path turned
`test_mypy_clean.py` red. CI never installs `[notebook]`, so no CI run can see
this; the overrides are pinned here instead. `follow_imports_for_stubs` is the
load-bearing half: numpy ships `.pyi` stubs, and `follow_imports = "skip"`
alone was measured to change nothing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_CONFIG = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())


@pytest.mark.parametrize("module", ["numpy.*", "matplotlib.*"])
def test_the_notebook_stack_is_not_followed_into_its_stubs(module: str) -> None:
    overrides = [o for o in _CONFIG["tool"]["mypy"]["overrides"] if o["module"] == module]
    assert len(overrides) == 1, f"expected exactly one override for {module}"
    (override,) = overrides
    assert override.get("follow_imports") == "skip"
    assert override.get("follow_imports_for_stubs") is True
