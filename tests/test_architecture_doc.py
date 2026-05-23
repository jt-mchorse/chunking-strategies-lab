"""Architecture-doc lock: catch drift between `docs/architecture.md` and
the actual shipped surface of the repo.

Sister to the architecture-doc locks shipped this same week in
``embedding-model-shootout`` (PR #20), ``vector-search-at-scale``
(PR #22), ``llm-eval-harness`` (PR #30), ``prompt-regression-suite``
(PR #25), ``llm-cost-optimizer`` (PR #28), and ``rag-production-kit``
(PR #30), plus the JS variants in ``mcp-server-cookbook``,
``nextjs-streaming-ai-patterns``, and ``ai-app-integration-tests``.

This repo's doc annotates surfaces with ``D-NNN`` core-decision
references; it uses no ``(#NN)`` issue annotations (compare:
``rag-production-kit`` PR #30 which uses both, and
``embedding-model-shootout`` PR #20 which uses ``(#NN)`` only). The
coverage axis is therefore D-NNN-only.

Three invariants pinned:

1. **Path-token reachability.** Every backtick-quoted token starting with
   one of ``RESOLVABLE_PREFIXES`` resolves on disk. Operator-supplied
   future artifacts allow-listed in ``OPERATOR_SUPPLIED_PATHS``.

2. **Active-decision coverage.** Every non-superseded ``D-NNN >= 2`` in
   ``MEMORY/core_decisions_ai.md`` is referenced at least once. D-001 is
   the scope baseline and intentionally excluded.

3. **Banned-phrase absence.** Phrases that characterized pre-fix drift
   elsewhere in the portfolio.

Hard-pin tests lock each constant so a future loose edit can't silently
weaken the guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "architecture.md"
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
    "notebooks/",
    "data/",
    "results/",
    "docs/",
    "tests/",
    ".github/",
)


OPERATOR_SUPPLIED_PATHS: tuple[str, ...] = ()


MIN_ACTIVE_DECISION_ID = 2


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def active_decisions() -> tuple[int, ...]:
    text = DECISIONS.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=- id:)", text)
    active: list[int] = []
    for block in blocks:
        id_match = re.search(r"- id:\s*D-(\d+)", block)
        if not id_match:
            continue
        sup_match = re.search(r"superseded_by:\s*(\S+)", block)
        is_active = (sup_match is None) or (sup_match.group(1).strip().lower() == "null")
        if is_active:
            n = int(id_match.group(1))
            if n >= MIN_ACTIVE_DECISION_ID:
                active.append(n)
    return tuple(sorted(active))


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
                # These are templates, not literal paths.
                if "<" in token or "{" in token or "*" in token:
                    break
                if token:
                    found.add(token)
                break
    return found


def _resolves_on_disk(token: str) -> bool:
    return (REPO_ROOT / token).exists()


def test_doc_exists() -> None:
    assert DOC.exists(), f"missing {DOC}"


def test_decisions_file_exists() -> None:
    assert DECISIONS.exists(), f"missing {DECISIONS}"


def test_backtick_paths_resolve_on_disk(doc_text: str) -> None:
    tokens = _extract_backtick_paths(doc_text)
    operator_set = set(OPERATOR_SUPPLIED_PATHS)
    unresolved = sorted(t for t in tokens if not _resolves_on_disk(t) and t not in operator_set)
    assert not unresolved, (
        "docs/architecture.md quotes paths that don't exist on disk:\n"
        + "\n".join(f"  - `{t}`" for t in unresolved)
        + "\n(regenerate the doc to match the current layout, fix the typo, "
        "or — if this is an operator-supplied future artifact — add it to "
        "OPERATOR_SUPPLIED_PATHS in tests/test_architecture_doc.py)"
    )


def test_operator_supplied_paths_actually_absent() -> None:
    landed = [p for p in OPERATOR_SUPPLIED_PATHS if (REPO_ROOT / p).exists()]
    assert not landed, (
        "These paths are listed as operator-supplied in "
        "tests/test_architecture_doc.py but exist on disk:\n"
        + "\n".join(f"  - `{p}`" for p in landed)
        + "\n(drop them from OPERATOR_SUPPLIED_PATHS so the resolvability "
        "check covers them as literal paths)"
    )


def test_every_active_decision_referenced(doc_text: str, active_decisions: tuple[int, ...]) -> None:
    referenced = {int(m.group(1)) for m in re.finditer(r"\bD-0*(\d+)\b", doc_text)}
    missing = sorted(set(active_decisions) - referenced)
    assert not missing, (
        "docs/architecture.md doesn't reference these active "
        "(non-superseded) core decisions even once:\n"
        + "\n".join(f"  - D-{n:03d}" for n in missing)
        + "\n(every shipped layer / posture in MEMORY/core_decisions_ai.md "
        "should be annotated in the doc where the relevant code lives; "
        "add a `D-NNN` reference to the relevant bullet)"
    )


def test_no_banned_phrases(doc_text: str) -> None:
    lowered = doc_text.lower()
    hits = [p for p in BANNED_PHRASES if p in lowered]
    assert not hits, (
        "docs/architecture.md contains drift phrases:\n"
        + "\n".join(f"  - {p!r}" for p in hits)
        + "\n(these phrases describe a pre-shipping state; the doc is a "
        "steady-state reference, not a PR description)"
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
        "notebooks/",
        "data/",
        "results/",
        "docs/",
        "tests/",
        ".github/",
    )


def test_min_active_decision_id_hard_pin() -> None:
    assert MIN_ACTIVE_DECISION_ID == 2


def test_operator_supplied_paths_hard_pin_set() -> None:
    assert OPERATOR_SUPPLIED_PATHS == ()
