"""Public-surface tests for ``chunking_lab/__init__.py``.

``chunking_lab`` re-exports 15 names from four submodules (``corpus``,
``embedder``, ``queries``, ``strategies``) and declares them in
``__all__`` + ``__version__``. The fifth submodule (``metrics``) is
intentionally accessed via dotted path, not re-exported. Every other
test in the suite imports submodules directly (``from chunking_lab
.strategies.fixed import FixedSizeStrategy``), so silent renames or
accidental ``__all__`` drops in ``__init__.py`` don't fail any test —
but they break the quoted ``from chunking_lab import (…)`` snippets in
``README.md`` and ``docs/setup.md``, and any downstream importer.

These five standalone + 4 parametrized tests lock the surface:

1. ``__version__`` is set to a semver-ish string.
2. Every name in ``__all__`` is bound on the package and non-None.
3. ``__all__`` agrees with the actual top-level relative ``from .X
   import …`` names (filter on ``level >= 1``).
4. Every name the quickstart snippets import resolves at the top level
   — **discovered** from ``README.md`` and ``docs/setup.md`` rather than
   transcribed beside the test (#194), with an anti-vacuity arm pinning
   that the discovery found a snippet in each document.
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

# The documents a reader copy-pastes a quickstart out of. Both carry a
# `from chunking_lab import …` snippet, and the claim "a reader who
# copy-pastes a quickstart hits an ImportError" covers both equally --
# `docs/setup.md` is a quickstart by exactly that definition (#194).
#
# A *list of documents* rather than a list of *names*: the names are
# discovered from these files below. The tuple this replaces was six names
# transcribed once, pinned to "lines 74 and 94", and by the time #194 read
# it the snippets were at 95 and 115 and a third one had appeared in
# `docs/setup.md` outside the corpus entirely. A check that can only fail
# when someone edits the check is not a lock on the thing it names.
_REPO_ROOT = _INIT_PATH.parent.parent
QUICKSTART_DOCS = ("README.md", "docs/setup.md")

# `from chunking_lab import ...`, in both spellings the docs use: the
# parenthesised multi-line block and the bare single-line form. The body is
# handed to `ast` rather than split on commas, so a trailing comma, a line
# continuation or an `as` alias parse the way Python parses them.
_IMPORT_BLOCK = re.compile(
    r"^from\s+chunking_lab\s+import\s+(\([^)]*\)|[^\n(]+)$",
    re.MULTILINE,
)

# The names the docs quoted when #194 was written. Not the source of truth --
# the documents are -- but the *control* on the discovery: if the regex or the
# parse regresses to finding fewer names, this set says so instead of the
# whole check quietly passing on an empty result.
KNOWN_QUICKSTART_NAMES = frozenset(
    {
        "HashEmbedder",
        "FixedSizeStrategy",
        "SemanticBoundaryStrategy",
        "load_corpus",
        "load_queries",
        "CANONICAL_EMBEDDING_MODEL",
    }
)


def _discover_quickstart_imports() -> dict[str, set[str]]:
    """Map each quickstart doc to the names its snippets import from the package.

    Returns one entry per document that contains at least one snippet, so the
    caller can assert on the *corpus* (how many documents, how many snippets)
    and not only on the union of names -- which is what makes the vacuity arm
    below able to distinguish "found nothing" from "found everything".
    """
    found: dict[str, set[str]] = {}
    for rel in QUICKSTART_DOCS:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        names: set[str] = set()
        for match in _IMPORT_BLOCK.finditer(text):
            stmt = f"from chunking_lab import {match.group(1)}"
            for node in ast.parse(stmt).body:
                assert isinstance(node, ast.ImportFrom)
                for alias in node.names:
                    # The *imported* name is what has to exist on the package;
                    # an `as` alias renames it only in the reader's script.
                    names.add(alias.name)
        if names:
            found[rel] = names
    return found


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


def test_the_discovery_finds_every_quickstart_snippet() -> None:
    """Anti-vacuity: the corpus is non-empty, in every document, before anything
    is asserted about it.

    This arm exists because the failure mode this change is most likely to
    introduce is a regex that matches **zero** blocks — after which every
    downstream assertion is over an empty set and passes for free. It pins the
    corpus (two documents, at least three snippets' worth of names) rather than
    the result, so it fails loudly on a discovery that has stopped discovering.

    Green on the pre-#194 tree too, in the sense that matters: the property it
    pins — that both documents carry a snippet — was already true. It is
    precisely the property the old six-name tuple did not walk.
    """
    found = _discover_quickstart_imports()
    assert set(found) == set(QUICKSTART_DOCS), (
        f"the quickstart-import discovery found snippets in {sorted(found)}, "
        f"expected all of {list(QUICKSTART_DOCS)}. Either a document lost its "
        f"`from chunking_lab import …` snippet, or `_IMPORT_BLOCK` stopped "
        f"matching the spelling it uses."
    )
    union = set().union(*found.values())
    assert union >= KNOWN_QUICKSTART_NAMES, (
        f"the discovery is missing names it found when #194 was written: "
        f"{sorted(KNOWN_QUICKSTART_NAMES - union)}. The documents are the "
        f"source of truth, but a *shrinking* result means the parser "
        f"regressed, not that the docs did."
    )


def test_quickstart_imports_resolve() -> None:
    """Every name the quickstarts import must exist on the package (#194).

    The names come from the documents, not from a list beside them. The tuple
    this replaces was transcribed once and pointed at "lines 74 and 94"; by the
    time it was read the snippets had moved to 95 and 115 and `docs/setup.md`
    had grown a third one outside the corpus. Three failures it could not see:
    a new document, a new name in an existing snippet, and its own drift.

    `test_all_matches_actual_top_level_imports` already parses `__init__.py`
    with `ast` for the other half of this same question; this is that technique
    applied to the documents.
    """
    found = _discover_quickstart_imports()
    missing = {
        rel: sorted(n for n in names if not hasattr(chunking_lab, n))
        for rel, names in found.items()
    }
    missing = {rel: names for rel, names in missing.items() if names}
    assert not missing, (
        f"documents import names that are not on the top-level surface: "
        f"{missing}. A reader copy-pasting the snippet gets an ImportError — "
        f"either restore the exports or fix the snippet."
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
