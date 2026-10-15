"""Every install hint in the repo is a command that runs (#266).

The repo settled on `pip install -e '.[<extra>]'` everywhere it tells an
operator how to get an optional extra -- README, `docs/setup.md`, the notebook,
`results/summary.md`, `run_matrix.py`'s exit-2 message. Three sites spelled it
differently, and each of those spellings failed as printed:

- `MiniLMEmbedder`'s ImportError named the distribution
  (`chunking-strategies-lab[sbert]`). It is not on PyPI, so pip answered "No
  matching distribution found" -- and `run_matrix.py --embedder minilm` printed
  that hint in parentheses after the right one, two commands on one line.
- `scripts/capture_demo.sh` and README's Demo section printed `.[sbert]`
  unquoted. On zsh, the macOS default shell, `[sbert]` is a glob character
  class, so the command stopped at "no matches found".

The population is discovered, not listed: every tracked text file is scanned
for `pip install`, and the extras are read from `pyproject.toml`, so a new hint
or a renamed extra is checked without editing this file.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: Append-only session logs quote history, including the forms this module
#: refuses; they are records, not instructions. This file is excluded because
#: its docstring names the broken spellings.
_EXEMPT = {
    "MEMORY/full_history_ai.md",
    "MEMORY/full_history_human.md",
    "tests/test_install_hints.py",
}

#: A command, not a mention: prose like "a stall during `pip install`" is not
#: an instruction anyone copies, so the match needs an argument after it.
_HINT = re.compile(r"pip install(?P<rest> [^\n]*)")


def _extras() -> set[str]:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(pyproject["project"]["optional-dependencies"])


def _tracked_text_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout
    files = []
    for rel in out.splitlines():
        if rel in _EXEMPT:
            continue
        try:
            (_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue
        files.append(rel)
    return files


def _hints() -> list[tuple[str, int, str]]:
    found = []
    for rel in _tracked_text_files():
        text = (_ROOT / rel).read_text(encoding="utf-8")
        for m in _HINT.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            found.append((rel, line_no, m.group("rest")))
    return found


def test_the_scan_sees_the_hints_it_exists_to_check() -> None:
    """Anti-vacuity: the README alone carries several `[sbert]` hints, and the
    two sites #266 fixed must be inside the scanned population."""
    files = {rel for rel, _, _ in _hints()}
    assert {"README.md", "scripts/capture_demo.sh", "chunking_lab/embedder.py"} <= files
    assert len(_hints()) >= 10


def test_every_install_hint_is_a_quoted_editable_install_of_a_real_extra() -> None:
    extras = _extras()
    allowed = re.compile(r" -e '\.\[(?P<extra>[a-z0-9_-]+)\]'")
    bad = []
    for rel, line_no, rest in _hints():
        m = allowed.match(rest)
        if m is None or m.group("extra") not in extras:
            bad.append(f"{rel}:{line_no}: pip install{rest[:60]}")
    assert not bad, (
        "install hints must read `pip install -e '.[<extra>]'` with <extra> one of "
        f"{sorted(extras)}: the package is not on PyPI, and an unquoted `[...]` is a "
        "glob that zsh refuses. Offenders:\n  " + "\n  ".join(bad)
    )


def test_minilm_import_error_names_the_installable_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime text, not only the source: what a library caller sees when
    the extra is missing. `None` in `sys.modules` makes the import raise
    `ImportError` whether or not the extra is installed here."""
    from chunking_lab.embedder import MiniLMEmbedder

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError) as exc:
        MiniLMEmbedder()
    message = str(exc.value)
    assert "pip install -e '.[sbert]'" in message
    assert "chunking-strategies-lab[" not in message
