"""Lock: ruff's scope stays limited to Python source (#142).

ruff 0.16.1 extended `ruff format` to Python code blocks *inside Markdown*.
CI installs ruff unpinned (`pip install -e '.[dev]'`), so the tool's scope
widened overnight and `ruff format --check .` began failing on committed
Markdown that had never been in scope — with nothing in this repo changed.
Six portfolio repos broke the same way on the same day.

`[tool.ruff] extend-exclude` re-states the scope the config always meant:
this repo's lint contract is "format Python source", and prose is not Python
source. This test keeps it that way. A future "tidy up pyproject" that drops
the entry would silently hand a formatter write access to committed
Markdown — which in chunking-strategies-lab means a *pinned benchmark corpus
document*, where a rewritten code block shifts every canonical metric.

Deliberately asserts on the config rather than shelling out to ruff: the
point is that the intent is recorded and can't be dropped by accident, and
the assertion must hold on any ruff version (including ones predating the
Markdown feature, which is exactly the version skew that let this land).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _ruff_config() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]


def test_markdown_is_excluded_from_ruff() -> None:
    excluded = _ruff_config().get("extend-exclude", [])
    assert "*.md" in excluded, (
        "ruff must not format Python code blocks inside Markdown. Restore "
        '`extend-exclude = ["*.md"]` under [tool.ruff] in pyproject.toml — '
        "without it, ruff >=0.16.1 rewrites committed prose and fixtures."
    )


def test_committed_markdown_exists_to_protect() -> None:
    # Guards against the exclusion quietly becoming a no-op: if the repo ever
    # stops carrying Markdown, the lock above passes vacuously and the next
    # person has no signal that it still matters.
    root = _PYPROJECT.parent
    assert (root / "README.md").is_file()


def test_pinned_corpus_documents_are_markdown_and_protected() -> None:
    # chunking-specific and the sharpest version of the harm: the corpus the
    # canonical fixtures are computed over is Markdown. `data/corpus/
    # 05_async_pipelines.md` is one of the two files ruff 0.16.1 wanted to
    # rewrite. Reformatting a code block inside it changes the text the
    # chunkers run over, shifting `n_chunks_total` and every recall/snippet
    # metric in `results/canonical__*.json` — a lint tool silently editing a
    # benchmark input.
    corpus = _PYPROJECT.parent / "data" / "corpus"
    docs = sorted(corpus.glob("*.md"))
    assert docs, "pinned corpus documents are missing"
    assert "*.md" in _ruff_config().get("extend-exclude", [])
