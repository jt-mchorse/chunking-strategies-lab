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

Four invariants pinned:

1. **Path-token reachability.** Every backtick-quoted token starting with
   one of ``RESOLVABLE_PREFIXES`` resolves on disk. Operator-supplied
   future artifacts allow-listed in ``OPERATOR_SUPPLIED_PATHS``.

2. **Symbol-reference resolution** (portfolio-ops #55). Every symbol the
   doc *names* -- a ``<submodule>.<symbol>`` attribute reference or a
   multi-word CamelCase public type -- resolves to a real attribute of the
   ``chunking_lab`` package, one of its submodules, the ``strategies``
   subpackage, or the Python ``builtins``. Catches the drift class #55
   catalogued portfolio-wide (a doc naming a nonexistent type such as
   llm-cost-optimizer's ``BatchAPIBackend`` or embedding-model-shootout's
   ``compute_frontier`` sails through CI green). Propagates the
   embedding-model-shootout #71 / python-async #70 / llm-eval-harness #140
   precedents, adapted to the citation styles this doc actually uses.

3. **Active-decision coverage.** Every non-superseded ``D-NNN >= 2`` in
   ``MEMORY/core_decisions_ai.md`` is referenced at least once. D-001 is
   the scope baseline and intentionally excluded.

4. **Banned-phrase absence.** Phrases that characterized pre-fix drift
   elsewhere in the portfolio.

Hard-pin tests lock each constant so a future loose edit can't silently
weaken the guard.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from chunking_lab.corpus import Document
from chunking_lab.queries import Query
from chunking_lab.strategies.fixed import FixedSizeStrategy

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


# Symbol-resolution lock (portfolio-ops #55). Subpackages under `chunking_lab/`
# whose attributes count as resolvable doc symbols, alongside the top-level
# package and its `*.py` submodules. Pinned by `test_symbol_subpackages_hard_pin_set`
# so a new subpackage is an intentional widening, not an accidental one.
_PKG = "chunking_lab"
_PKG_DIR = REPO_ROOT / _PKG
_SUBPACKAGES = ("strategies",)


# File-suffix tokens that look like a `<name>.<attr>` symbol reference but are
# really filenames (`validate.py`, `corpus.json`). Excluded from the dotted-symbol
# resolution check so a filename isn't mistaken for a submodule attribute.
# Hard-pinned by `test_symbol_skip_extensions_hard_pin_set`.
SYMBOL_SKIP_EXTENSIONS = ("py", "sqlite", "json", "md", "txt", "yaml", "yml", "sh", "toml")


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


def _package_symbol_resolves(name: str) -> bool:
    """True if ``name`` is an attribute of the ``chunking_lab`` package, any of
    its ``*.py`` submodules, the ``strategies`` subpackage, or the Python
    ``builtins``.

    Submodule + subpackage coverage is load-bearing: ``MiniLMEmbedder`` lives in
    ``chunking_lab.embedder`` and ``RetrievalRun`` in ``chunking_lab.metrics``,
    neither re-exported at package level -- a surface-only check would
    false-positive on both. Builtins are included because the doc legitimately
    names ``ValueError`` / ``KeyboardInterrupt`` in its error-handling narrative;
    resolving them via the ``builtins`` module keeps the exemption principled and
    self-maintaining (a future ``TypeError`` mention just works) rather than a
    hand-maintained allow-list that rots.
    """
    import builtins
    import importlib

    if hasattr(builtins, name):
        return True
    pkg = importlib.import_module(_PKG)
    if hasattr(pkg, name):
        return True
    module_names = [f"{_PKG}.{p.stem}" for p in _PKG_DIR.glob("*.py") if p.stem != "__init__"]
    module_names += [f"{_PKG}.{sub}" for sub in _SUBPACKAGES]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, name):
            return True
    return False


def _extract_symbol_refs(text: str) -> tuple[set[str], set[str]]:
    """Split backtick-quoted tokens into the two symbol-citation styles the doc
    uses, so the resolver only checks genuine symbol claims (not prose, field
    names, or file paths). Returns ``(dotted, camel)``.

    - ``dotted``: ``<submodule>.<symbol>`` where ``<submodule>`` is a real
      ``chunking_lab/*.py`` module stem and the token is NOT a filename
      (dropped via ``SYMBOL_SKIP_EXTENSIONS``). e.g. ``metrics.RetrievalRun``.
      Package-qualified paths (``chunking_lab.validate``) and stdlib refs
      (``os.replace``, ``pytest.importorskip``) are skipped: their prefix is not
      a submodule stem.
    - ``camel``: a *multi-word* CamelCase identifier -- an internal
      lowercase->uppercase boundary, e.g. ``LateChunkingStrategy``,
      ``HashEmbedder``. Single-word capitalized tokens (``Chunk``, ``Strategy``,
      ``Embedder``) are deliberately excluded: they collide with prose words and
      would false-positive. Bare snake_case is not locked (collides with the
      finding-code / field names the doc quotes).
    """
    submodules = {p.stem for p in _PKG_DIR.glob("*.py") if p.stem != "__init__"}
    dotted: set[str] = set()
    camel: set[str] = set()
    for match in re.finditer(r"`([^`\n]+)`", text):
        token = match.group(1).strip()
        token = re.sub(r"\(\)$", "", token)
        while token and token[-1] in ".,;:":
            token = token[:-1]
        dotted_match = re.fullmatch(r"([a-z_]+)\.([A-Za-z_][A-Za-z0-9_]*)", token)
        if dotted_match:
            module, attr = dotted_match.group(1), dotted_match.group(2)
            if module in submodules and attr not in SYMBOL_SKIP_EXTENSIONS:
                dotted.add(token)
            continue
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*", token) and re.search(
            r"[a-z][A-Z]", token
        ):
            camel.add(token)
    return dotted, camel


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


def test_doc_symbol_refs_resolve(doc_text: str) -> None:
    """Every symbol the doc names resolves to a real attribute (portfolio-ops #55).

    ``test_backtick_paths_resolve_on_disk`` validates slash-path tokens only; a
    *symbol* reference -- a ``<submodule>.<symbol>`` attribute or a multi-word
    CamelCase public type -- was unguarded. That is exactly the drift class #55
    catalogued (a doc naming a nonexistent ``BatchAPIBackend`` /
    ``compute_frontier`` stays CI-green). Inverse-verified by
    ``test_symbol_resolver_flags_injected_drift``.
    """
    import importlib

    dotted, camel = _extract_symbol_refs(doc_text)
    assert dotted or camel, (
        "expected at least one symbol reference (`<submodule>.<symbol>` or a "
        "multi-word CamelCase type) in docs/architecture.md -- the resolver "
        "would otherwise be vacuously green"
    )

    unresolved: list[str] = []
    for token in sorted(dotted):
        module_name, _, symbol = token.rpartition(".")
        try:
            module = importlib.import_module(f"{_PKG}.{module_name}")
        except ModuleNotFoundError:
            unresolved.append(f"{token} (module {_PKG}.{module_name} not importable)")
            continue
        if not hasattr(module, symbol):
            unresolved.append(token)
    for token in sorted(camel):
        if not _package_symbol_resolves(token):
            unresolved.append(f"{token} (not a chunking_lab symbol or a builtin)")

    assert not unresolved, (
        "docs/architecture.md names symbols that don't exist in the package:\n"
        + "\n".join(f"  - {u}" for u in unresolved)
        + "\n(fix the doc to match the shipped symbol, or update the rename that "
        "orphaned it)"
    )


def test_symbol_resolver_flags_injected_drift() -> None:
    """Inverse safety net: a nonexistent CamelCase type in doc text is flagged.

    Guards against a vacuously-green resolver -- if a refactor ever neutered
    extraction or resolution, this fails. Mirrors the #55 drift shape (a doc
    naming a symbol that never existed) while a real symbol in the same string
    still resolves.
    """
    fake = "The `NonexistentChunkingStrategy` produces a `RetrievalRun`."
    dotted, camel = _extract_symbol_refs(fake)
    assert "NonexistentChunkingStrategy" in camel
    assert "RetrievalRun" in camel
    unresolved = sorted(t for t in camel if not _package_symbol_resolves(t))
    assert unresolved == ["NonexistentChunkingStrategy"]


def test_symbol_skip_extensions_hard_pin_set() -> None:
    assert SYMBOL_SKIP_EXTENSIONS == (
        "py",
        "sqlite",
        "json",
        "md",
        "txt",
        "yaml",
        "yml",
        "sh",
        "toml",
    )


def test_symbol_subpackages_hard_pin_set() -> None:
    assert _SUBPACKAGES == ("strategies",)


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


# ---------------------------------------------------------------------------
# Diagram field-label lock (#122).
#
# A few mermaid node labels annotate a dataclass type with a parenthesized,
# comma-separated field list, e.g. ``Query (id, question, expected_doc,
# expected_snippet)`` or ``FixedSizeStrategy<br/>(chunk_chars, overlap_chars)``.
# Those labels are NOT backtick-quoted and their tokens are bare snake_case, so
# neither ``test_backtick_paths_resolve_on_disk`` (slash paths) nor
# ``test_doc_symbol_refs_resolve`` (dotted / CamelCase symbols) catches a wrong
# field name in them. That is exactly how ``Query (id, text, ...)`` (real field
# is ``question``) and ``FixedSizeStrategy (chunk_chars, overlap)`` (real field
# is ``overlap_chars``) drifted through CI green. Tie each such label's tokens
# to the live ``dataclasses.fields(...)`` of the type it names.


# Doc-named type -> the dataclass whose field names its diagram label must
# match. Only types whose label is a pure comma-separated field list belong
# here; the descriptive labels (``Chunk[] (text + start/end offsets)``,
# ``LateChunk[] (text + offsets + vector)``) are prose, not field lists, and are
# intentionally excluded. Hard-pinned by
# ``test_diagram_field_label_types_hard_pin_set``.
DIAGRAM_FIELD_LABEL_TYPES: dict[str, type] = {
    "Query": Query,
    "Document": Document,
    "FixedSizeStrategy": FixedSizeStrategy,
}

_IDENT_RE = re.compile(r"[a-z_][a-z0-9_]*")


def _diagram_field_lists(text: str, type_name: str) -> list[list[str]]:
    """Every parenthesized comma-separated identifier list the doc attaches to
    ``type_name`` in a mermaid node label.

    Matches ``type_name`` as a whole word (so ``Query`` does not match inside
    ``QueryResult``) optionally followed by a ``<br/>`` line break or spaces,
    then a ``(...)`` group. Only groups whose entire content is a comma-list of
    snake_case identifiers are returned -- descriptive labels such as
    ``(separator hierarchy)`` are skipped so the resolver checks genuine field
    claims only.
    """
    pattern = re.compile(re.escape(type_name) + r"\b(?:<br/>|\s)*\(([^)]*)\)")
    lists: list[list[str]] = []
    for m in pattern.finditer(text):
        tokens = [t.strip() for t in m.group(1).split(",")]
        if tokens and all(_IDENT_RE.fullmatch(t) for t in tokens):
            lists.append(tokens)
    return lists


def test_diagram_node_label_fields_are_real_dataclass_fields(doc_text: str) -> None:
    """Every field named in a pinned type's mermaid node label is a real
    ``dataclasses.fields(...)`` name of that type (#122).

    Catches the field-label drift class the path/symbol locks miss: an
    un-backticked bare-identifier field list inside a node label, e.g.
    ``Query (id, text, ...)`` where the real field is ``question``.
    """
    problems: list[str] = []
    for type_name, cls in DIAGRAM_FIELD_LABEL_TYPES.items():
        real = {f.name for f in dataclasses.fields(cls)}
        lists = _diagram_field_lists(doc_text, type_name)
        assert lists, (
            f"expected at least one `{type_name} (field, ...)` node label in "
            "docs/architecture.md -- the field-label lock would be vacuous for "
            f"{type_name} otherwise"
        )
        for tokens in lists:
            for tok in tokens:
                if tok not in real:
                    problems.append(
                        f"{type_name} label names `{tok}`, not a field of "
                        f"{cls.__module__}.{cls.__qualname__} "
                        f"(real fields: {', '.join(sorted(real))})"
                    )
    assert not problems, (
        "docs/architecture.md mermaid node labels name fields that don't exist "
        "on the dataclass they annotate:\n"
        + "\n".join(f"  - {p}" for p in problems)
        + "\n(fix the label to match the shipped dataclass field name)"
    )


def test_diagram_field_label_resolver_flags_injected_drift() -> None:
    """Inverse safety net: a wrong field name in a node label is flagged while a
    correct sibling token still resolves (guards a vacuously-green resolver).

    Mirrors the #122 drift shape -- ``text`` in place of the real ``question``
    field -- so a future refactor that neutered extraction or resolution fails.
    """
    real = {f.name for f in dataclasses.fields(Query)}
    good = _diagram_field_lists('QL --> Q["Query (id, question)"]', "Query")
    bad = _diagram_field_lists('QL --> Q["Query (id, text)"]', "Query")
    assert good == [["id", "question"]]
    assert bad == [["id", "text"]]
    assert all(t in real for t in good[0])
    assert "text" not in real  # the exact drift this lock rules out
    # Whole-word match: `Query` must not match inside `QueryResult`.
    assert _diagram_field_lists('R["QueryResult (query_id, expected_doc)"]', "Query") == []


def test_diagram_field_label_types_hard_pin_set() -> None:
    assert tuple(DIAGRAM_FIELD_LABEL_TYPES) == ("Query", "Document", "FixedSizeStrategy")
