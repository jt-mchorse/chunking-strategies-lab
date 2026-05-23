"""README snapshot lock: catch drift between README.md and the actual
shipped surface of the repo.

Sister to the README locks across the portfolio:
``llm-eval-harness`` ``test_readme_defaults_snapshot.py``,
``llm-cost-optimizer`` ``test_readme_defaults_snapshot.py`` +
``test_readme_paths_resolve.py``, ``embedding-model-shootout``
``test_readme_snapshot.py``, ``python-async-llm-pipelines``
``test_readme_kwarg_consistency.py``,
``rag-production-kit``
``test_readme_what_this_is_lists_shipped_layers.py``,
``vector-search-at-scale`` ``test_readme_snapshot.py``,
``prompt-regression-suite`` ``test_readme_defaults_snapshot.py``.

Three invariants pinned:

1. **Banned-phrase absence.** Pre-shipping drift shapes (``this pr``,
   ``pending``, ``to-be-filed``, ``(unfiled)``) are absent (case-
   insensitive). These are the same shapes the portfolio-wide hygiene
   sweep is catching across all twelve repos.

2. **Active-decision-range upper bound.** README's
   ``See docs/architecture.md ... (D-002…D-NNN)`` claim must cite the
   highest active (non-superseded) ``D-NNN`` in
   ``MEMORY/core_decisions_ai.md``. So a future decision that lands
   without README being updated fails the test loud.

3. **Path-token reachability.** Every backtick-quoted token starting
   with one of ``RESOLVABLE_PREFIXES`` resolves on disk. Glob ``*``,
   ``<...>``, and ``{...}`` shapes skipped as templates.

Hard-pin tests lock ``BANNED_PHRASES`` and ``RESOLVABLE_PREFIXES``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
DECISIONS = REPO_ROOT / "MEMORY" / "core_decisions_ai.md"


BANNED_PHRASES = (
    "this pr",
    "pending downstream",
    "(unfiled)",
    "to-be-filed",
)


RESOLVABLE_PREFIXES = (
    "chunking_lab/",
    "scripts/",
    "data/",
    "docs/",
    "results/",
    "notebooks/",
    "tests/",
    ".github/",
)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def max_active_decision_id() -> int:
    """Highest non-superseded ``D-NNN`` in ``MEMORY/core_decisions_ai.md``.

    Returns the integer (e.g. ``11`` if D-011 is the latest active
    decision). The README's architecture-doc-summary line cites
    ``D-002…D-NNN`` and this is the NNN that must be current.
    """
    text = DECISIONS.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=- id:)", text)
    best = 0
    for block in blocks:
        id_match = re.search(r"- id:\s*D-(\d+)", block)
        if not id_match:
            continue
        sup_match = re.search(r"superseded_by:\s*(\S+)", block)
        is_active = (sup_match is None) or (sup_match.group(1).strip().lower() == "null")
        if is_active:
            n = int(id_match.group(1))
            if n > best:
                best = n
    return best


def _extract_backtick_paths(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"`([^`\n]+)`", text):
        token = match.group(1).strip()
        for prefix in RESOLVABLE_PREFIXES:
            if token.startswith(prefix):
                while token and token[-1] in ".,;:":
                    token = token[:-1]
                token = re.sub(r"\(\)$", "", token)
                # Skip placeholder shapes: `<var>` / `{a,b}` / glob `*`.
                if "<" in token or "{" in token or "*" in token:
                    break
                if token:
                    found.add(token)
                break
    return found


def test_readme_exists() -> None:
    assert README.exists(), f"missing {README}"


def test_no_banned_phrases(readme_text: str) -> None:
    lowered = readme_text.lower()
    hits = [p for p in BANNED_PHRASES if p in lowered]
    assert not hits, (
        "README.md contains drift phrases:\n"
        + "\n".join(f"  - {p!r}" for p in hits)
        + "\n(these phrases describe a pre-shipping state; the README is "
        "a steady-state product description, not a PR description)"
    )


def test_decision_range_cites_latest_active(readme_text: str, max_active_decision_id: int) -> None:
    """The README's architecture-doc-summary claims the design decisions
    span a range like ``D-002…D-011``. The upper bound must equal the
    highest active D-NNN in MEMORY.
    """
    # Match the unicode ellipsis form (`D-002…D-NNN`) used in the README;
    # also tolerate the ASCII three-dot form for forward compatibility.
    pattern = re.compile(r"D-0*2\s*(?:…|\.\.\.)\s*D-0*(\d+)")
    matches = pattern.findall(readme_text)
    assert matches, (
        "README.md must cite the active-decision range as "
        "`D-002…D-NNN` somewhere (the architecture-doc-summary "
        "paragraph by convention). Not found."
    )
    cited = max(int(m) for m in matches)
    assert cited == max_active_decision_id, (
        f"README.md cites decision range up to D-{cited:03d}, but the "
        f"highest active D-NNN in MEMORY/core_decisions_ai.md is "
        f"D-{max_active_decision_id:03d}. Update the README's "
        f"architecture-doc-summary to D-002…D-{max_active_decision_id:03d}."
    )


def test_backtick_paths_resolve_on_disk(readme_text: str) -> None:
    tokens = _extract_backtick_paths(readme_text)
    unresolved = sorted(t for t in tokens if not (REPO_ROOT / t).exists())
    assert not unresolved, (
        "README.md quotes paths that don't exist on disk:\n"
        + "\n".join(f"  - `{t}`" for t in unresolved)
        + "\n(regenerate the README or fix the typo)"
    )


def test_banned_phrases_hard_pin_set() -> None:
    assert BANNED_PHRASES == (
        "this pr",
        "pending downstream",
        "(unfiled)",
        "to-be-filed",
    )


def test_resolvable_prefixes_hard_pin_set() -> None:
    assert RESOLVABLE_PREFIXES == (
        "chunking_lab/",
        "scripts/",
        "data/",
        "docs/",
        "results/",
        "notebooks/",
        "tests/",
        ".github/",
    )
