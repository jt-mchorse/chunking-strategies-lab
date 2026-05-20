"""Public-surface tests for ``chunking_lab/__init__.py``.

``chunking_lab`` re-exports 15 names from four submodules (``corpus``,
``embedder``, ``queries``, ``strategies``) and declares them in
``__all__`` + ``__version__``. The fifth submodule (``metrics``) is
intentionally accessed via dotted path, not re-exported. Every other
test in the suite imports submodules directly (``from chunking_lab
.strategies.fixed import FixedSizeStrategy``), so silent renames or
accidental ``__all__`` drops in ``__init__.py`` don't fail any test —
but they break the README's two quoted ``from chunking_lab import (…)``
snippets (lines 74 and 94) and any downstream importer.

These four standalone + 4 parametrized tests lock the surface:

1. ``__version__`` is set to a semver-ish string.
2. Every name in ``__all__`` is bound on the package and non-None.
3. ``__all__`` agrees with the actual top-level relative ``from .X
   import …`` names (filter on ``level >= 1``).
4. Six names quoted across the README's two quickstart import snippets
   resolve at the top level: ``HashEmbedder``, ``FixedSizeStrategy``,
   ``SemanticBoundaryStrategy``, ``load_corpus``, ``load_queries``,
   ``CANONICAL_EMBEDDING_MODEL``.
5. One anchor per *re-exported* submodule (4 anchors). ``metrics`` is
   deliberately excluded — it's dotted-path-only by design, and
   re-exporting it would expand the public surface without an explicit
   decision.

Sixth strike of the portfolio-wide public-surface hygiene pattern.
Orthogonal to the existing snapshot tests (`test_summary_snapshot.py`
locks the matrix output; this test locks the Python surface the
README's prose depends on).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import chunking_lab

_INIT_PATH = Path(chunking_lab.__file__)
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")

# Union of names quoted across the README's two `from chunking_lab
# import …` snippets (lines 74 and 94 in README.md). If any of these
# six names drops off the top level, every reader who copy-pastes a
# quickstart hits an ImportError.
README_QUICKSTART_NAMES = (
    "HashEmbedder",
    "FixedSizeStrategy",
    "SemanticBoundaryStrategy",
    "load_corpus",
    "load_queries",
    "CANONICAL_EMBEDDING_MODEL",
)

# Anchor names that prove each *re-exported* submodule survived.
# ``metrics`` is intentionally excluded — it's accessed via dotted path
# only, by design.
SUBMODULE_ANCHORS = {
    "corpus": "load_corpus",
    "embedder": "HashEmbedder",
    "queries": "load_queries",
    "strategies": "FixedSizeStrategy",
}


def _parse_init_relative_imports() -> set[str]:
    """Return the set of names imported into ``__init__.py`` via
    top-level relative ``from .X import (...)`` blocks."""
    tree = ast.parse(_INIT_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_version_is_set_to_semver_ish_string() -> None:
    """``__version__`` is published; downstream importers and PyPI
    builds rely on it."""
    assert hasattr(chunking_lab, "__version__"), (
        "chunking_lab.__version__ is missing — packaging tools and "
        "downstream `chunking_lab.__version__` lookups will break."
    )
    version = chunking_lab.__version__
    assert isinstance(version, str), (
        f"chunking_lab.__version__ should be a string, got {type(version).__name__}: {version!r}."
    )
    assert version, "chunking_lab.__version__ is an empty string."
    assert _SEMVER_PATTERN.match(version), (
        f"chunking_lab.__version__ = {version!r} doesn't look like "
        f"semver (expected MAJOR.MINOR.PATCH[-prerelease][+build])."
    )


def test_all_names_are_bound_and_non_none() -> None:
    """Every name in ``__all__`` must be importable and non-None."""
    missing: list[str] = []
    none_valued: list[str] = []
    for name in chunking_lab.__all__:
        if not hasattr(chunking_lab, name):
            missing.append(name)
            continue
        if getattr(chunking_lab, name) is None:
            none_valued.append(name)
    assert not missing, (
        f"chunking_lab.__all__ advertises names that are not bound on "
        f"the package: {missing}. The most likely cause is a re-import "
        f"line was deleted from __init__.py but __all__ wasn't updated."
    )
    assert not none_valued, (
        f"chunking_lab.__all__ entries bound to None: {none_valued}. "
        f"A re-import probably resolved to a missing submodule attribute."
    )


def test_all_matches_actual_top_level_imports() -> None:
    """``__all__`` should equal the set of top-level relative re-exports."""
    advertised = set(chunking_lab.__all__)
    imported = _parse_init_relative_imports()
    only_imported = imported - advertised
    only_advertised = advertised - imported
    assert not only_imported, (
        f"Names imported into chunking_lab/__init__.py but missing from "
        f"__all__: {sorted(only_imported)}. Add them to __all__ or stop "
        f"importing them at the top level."
    )
    assert not only_advertised, (
        f"Names in chunking_lab.__all__ but not imported at the top of "
        f"__init__.py: {sorted(only_advertised)}. Add the import or "
        f"remove the __all__ entry."
    )


def test_readme_quickstart_imports_resolve() -> None:
    """README's two quickstart import snippets must keep working.

    The README literally quotes (lines 74 and 94)::

        from chunking_lab import (
            HashEmbedder, FixedSizeStrategy, SemanticBoundaryStrategy, load_corpus,
        )

        from chunking_lab import load_corpus, load_queries, CANONICAL_EMBEDDING_MODEL

    If any of those six names disappears from the top-level surface,
    every reader who copy-pastes a quickstart hits an ImportError.
    """
    missing = [n for n in README_QUICKSTART_NAMES if not hasattr(chunking_lab, n)]
    assert not missing, (
        f"chunking_lab is missing README-quoted names: {missing}. The "
        f"README's quickstart imports them directly — either restore "
        f"the exports or update the README."
    )


@pytest.mark.parametrize(
    ("submodule", "anchor"),
    sorted(SUBMODULE_ANCHORS.items()),
    ids=sorted(SUBMODULE_ANCHORS.keys()),
)
def test_submodule_anchor_re_exported(submodule: str, anchor: str) -> None:
    """One anchor per *re-exported* submodule survives at the top level.

    ``metrics`` is intentionally NOT in this map — it's accessed via
    dotted path and re-exporting it at the top level would expand the
    public surface without an explicit decision.
    """
    assert hasattr(chunking_lab, anchor), (
        f"`{anchor}` from `chunking_lab.{submodule}` is no longer "
        f"re-exported at the top level. Did `{submodule}` move or get "
        f"renamed? Update `chunking_lab/__init__.py` to re-export from "
        f"the new path."
    )
