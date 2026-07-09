"""Snapshot test for `results/summary.md`.

`scripts/run_matrix.py` writes 5 strategy JSONs and a single `summary.md`
markdown table from the in-memory `RetrievalRun` list. The committed JSONs
carry `wall_clock_ms` baked in, so feeding them back through the renderer
produces a deterministic markdown — but no existing test enforces the
committed `summary.md` matches what `_render_summary` would produce from
the JSONs today.

This module is the missing piece. Pattern parallels the snapshot tests
landed today in `llm-cost-optimizer` (docs/savings.{json,md} + README),
`prompt-regression-suite` (docs/regression_demo.html), and
`rag-production-kit` (README rewriter table).

When the snapshot fails, the regen path is:

    python scripts/run_matrix.py

…then `git diff results/summary.md` before committing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.metrics import RetrievalRun  # noqa: E402
from scripts.run_matrix import _render_summary  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"
SUMMARY_MD = RESULTS_DIR / "summary.md"

# Strategy order in `_build_strategies` in `scripts/run_matrix.py`. The
# renderer iterates in input order, so the snapshot needs to feed the
# JSONs back in the same order they were originally produced.
STRATEGY_ORDER = (
    "fixed-size",
    "recursive",
    "semantic",
    "late-chunking",
    "structure-aware",
)

REGEN_HINT = (
    "Regenerate the summary:\n"
    "  python scripts/run_matrix.py\n"
    "Then inspect with `git diff results/summary.md` before committing."
)


def _load_run_from_json(path: Path) -> RetrievalRun:
    """Reconstruct a `RetrievalRun` from a committed result JSON.

    Thin wrapper around `RetrievalRun.from_json` (#47); kept as a
    helper so the snapshot tests' import surface doesn't change.
    The classmethod rebuilds `per_query` from the on-disk shape;
    the snapshot renderer doesn't read it but having it populated
    is harmless and round-trips byte-for-byte through `to_json`.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RetrievalRun.from_json(payload)


def _committed_run_jsons() -> list[Path]:
    """All committed `results/*.json` files, sorted by glob — the script
    writes one per strategy with a timestamp prefix, so sorting is stable
    within a single matrix run.
    """
    return sorted(RESULTS_DIR.glob("*.json"))


def _runs_in_strategy_order() -> list[RetrievalRun]:
    by_name = {p.stem.split("__", 1)[1]: p for p in _committed_run_jsons()}
    missing = set(STRATEGY_ORDER) - by_name.keys()
    assert not missing, (
        f"Missing committed result JSONs for strategies: {sorted(missing)}.\n"
        f"Found: {sorted(by_name)}.\n{REGEN_HINT}"
    )
    return [_load_run_from_json(by_name[name]) for name in STRATEGY_ORDER]


def test_committed_summary_md_matches_render_from_committed_results() -> None:
    """`_render_summary` over the committed JSONs must equal `results/summary.md`."""
    runs = _runs_in_strategy_order()
    rendered = _render_summary(runs, runs[0].embedder_model)
    committed = SUMMARY_MD.read_text(encoding="utf-8")
    assert rendered == committed, (
        f"results/summary.md is out of sync with `_render_summary` over the "
        f"committed result JSONs.\n{REGEN_HINT}"
    )


def test_committed_summary_md_strategy_set_is_complete() -> None:
    """Every strategy in `STRATEGY_ORDER` must have a committed JSON, and
    no extras. Guards against silently adding or dropping a strategy.
    """
    found_names = {p.stem.split("__", 1)[1] for p in _committed_run_jsons()}
    expected_names = set(STRATEGY_ORDER)
    extra = found_names - expected_names
    missing = expected_names - found_names
    assert not missing, (
        f"Committed strategy JSONs are missing {sorted(missing)}.\n"
        f"Found: {sorted(found_names)}.\n"
        f"Expected: {sorted(expected_names)}.\n"
        "If the strategy lineup changed intentionally, update STRATEGY_ORDER "
        f"in this file and run_matrix._build_strategies together.\n{REGEN_HINT}"
    )
    assert not extra, (
        f"Committed strategy JSONs include unexpected names {sorted(extra)}.\n"
        f"Found: {sorted(found_names)}.\n"
        f"Expected: {sorted(expected_names)}.\n"
        "If the strategy lineup changed intentionally, update STRATEGY_ORDER "
        f"in this file and run_matrix._build_strategies together.\n{REGEN_HINT}"
    )


@pytest.mark.parametrize("strategy", STRATEGY_ORDER)
def test_each_strategy_json_loads_into_a_retrieval_run(strategy: str) -> None:
    """Every committed strategy JSON must round-trip through the loader.

    Catches schema regressions where a new required field is added to
    `RetrievalRun` but the existing JSONs aren't migrated.
    """
    matches = [p for p in _committed_run_jsons() if p.stem.endswith(f"__{strategy}")]
    assert len(matches) == 1, (
        f"Expected exactly one committed JSON for {strategy!r}; found {len(matches)}."
    )
    run = _load_run_from_json(matches[0])
    assert run.strategy_name == strategy
    # `recall_at_k` and `snippet_hit_at_k` must carry the k values the
    # renderer expects (1, 3, 5). If a future bench drops one, the
    # summary renderer's `.get(k, 0)` would silently emit 0 for it —
    # this assertion makes the desync loud at load time.
    for k in (1, 3, 5):
        assert k in run.recall_at_k, f"{strategy}: recall@{k} missing"
        assert k in run.snippet_hit_at_k, f"{strategy}: snippet-hit@{k} missing"


def test_render_summary_escapes_pipe_in_strategy_name_so_columns_dont_break() -> None:
    # #100 (sibling to rag-kit comment #130, llm-eval-harness #134,
    # embedding-model-shootout #79): `strategy_name` is the one free-form GFM
    # table cell (every other is a formatted number). It reaches
    # `_render_summary` pipe-free from the five shipped strategies, but a BYO
    # `Strategy` whose `name` carries a `|`, or a `RetrievalRun` loaded from
    # external JSON via `from_json`, can inject one. GFM splits table cells on
    # unescaped pipes, so a piped name adds a spurious column and corrupts the
    # summary table. The fix escapes `|` -> `\|`; the invariant is that the
    # data row's unescaped-pipe count equals the header's. Fails pre-fix
    # (the piped row carried 11 unescaped pipes vs the header's 10).
    import re

    run = RetrievalRun(
        strategy_name="fixed|256",
        embedder_model="HashEmbedder",
        dataset_version="v1",
        n_queries=3,
        n_chunks_total=10,
        recall_at_k={1: 0.5, 3: 0.6, 5: 0.7},
        snippet_hit_at_k={1: 0.4, 3: 0.5, 5: 0.6},
        per_query=(),
        wall_clock_ms=12.0,
    )
    md = _render_summary([run], "HashEmbedder")
    lines = md.splitlines()
    header_line = next(line for line in lines if line.startswith("| strategy "))
    row_line = next(line for line in lines if "fixed" in line and "recall" not in line)

    def unescaped_pipes(s: str) -> int:
        # A `\|` renders as a literal pipe and does NOT split the cell; only a
        # bare, unescaped `|` is a column delimiter.
        return len(re.findall(r"(?<!\\)\|", s))

    assert unescaped_pipes(row_line) == unescaped_pipes(header_line)
    # The literal pipe is preserved (escaped), not dropped.
    assert "fixed\\|256" in row_line


class _NamedStubEmbedder:
    """A deterministic, dep-free embedder whose reported `model_name` differs
    from its Python class name — exactly the shape of `MiniLMEmbedder`
    (`model_name = "sentence-transformers/all-MiniLM-L6-v2"`), which is what
    exposes #116. `HashEmbedder` has no `model_name`, so it can't."""

    model_name = "stub-embedder/v1"

    def embed(self, text: str) -> list[float]:
        # Constant non-zero vector: cosine is well-defined and finite; the exact
        # values don't matter for the header/name invariant under test.
        return [1.0, 0.0, 0.0, 0.0]


def test_run_matrix_summary_header_matches_persisted_embedder_model(tmp_path, monkeypatch) -> None:
    """#116: the `summary.md` embedder header must be rendered from the SAME
    canonical name source (`_embedder_model_name`) that `evaluate_strategy`
    persists into every `RetrievalRun.embedder_model` — not `type().__name__`.

    Pre-fix the runner rendered the header from `type(embedder).__name__`
    (`_NamedStubEmbedder`) while the JSONs carried `embedder_model` =
    `stub-embedder/v1`, so an honest `--canonical-out --embedder minilm` refresh
    wrote a `summary.md` that permanently failed
    `test_committed_summary_md_matches_render_from_committed_results`. Drive the
    real `main()` end-to-end (into a tmp results dir, so the committed fixtures
    are untouched) and assert the written header agrees with the persisted JSON.
    """
    import scripts.run_matrix as rm
    from chunking_lab.metrics import _embedder_model_name

    stub = _NamedStubEmbedder()
    monkeypatch.setattr(rm, "_build_embedder", lambda _name: stub)

    rc = rm.main(["--embedder", "minilm", "--canonical-out", "--results-dir", str(tmp_path)])
    assert rc == 0

    summary_header = (tmp_path / "summary.md").read_text().splitlines()[2]
    persisted = json.loads((tmp_path / "canonical__fixed-size.json").read_text())["embedder_model"]

    # The persisted model name is the canonical `model_name`, not the class name.
    assert persisted == _embedder_model_name(stub) == "stub-embedder/v1"
    # Pre-fix this was `_embedder_: `_NamedStubEmbedder`` and this assertion failed.
    assert f"`{persisted}`" in summary_header
    assert "_NamedStubEmbedder" not in summary_header
