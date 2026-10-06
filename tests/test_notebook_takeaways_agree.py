"""The notebook's takeaways agree with the committed run and the code (#230).

Two bullets said the opposite of what the run shows: that "the *embedding*
step is what scales with chunk count, not the chunking decision itself" (the
semantic strategy embeds every sentence to decide its boundaries -- 115 calls,
more than its 86 chunk embeddings) and that snippet-hit is "≪ recall, always"
(at k=1 two committed strategies score the same on both). These arms derive the
replacement claims rather than restating them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chunking_lab.corpus import load_corpus
from chunking_lab.embedder import HashEmbedder
from chunking_lab.queries import load_queries
from scripts.run_matrix import _build_strategies

ROOT = Path(__file__).resolve().parent.parent


def _takeaways() -> str:
    # The builder imports `nbformat`, which CI does not install (D-009), so
    # only the prose arms need it; the property arms below run everywhere.
    pytest.importorskip("nbformat")
    from notebooks._build_notebook import _TAKEAWAYS  # noqa: PLC0415

    return _TAKEAWAYS


class _Counting:
    def __init__(self) -> None:
        self.inner = HashEmbedder()
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self.inner.embed(text)

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def _chunking_embeds() -> dict[str, tuple[int, int]]:
    """strategy name -> (embed calls inside chunk(), chunks produced)."""
    docs = load_corpus()
    out: dict[str, tuple[int, int]] = {}
    emb = _Counting()
    for strategy in _build_strategies(emb):
        emb.calls = 0
        chunks = [c for d in docs for c in strategy.chunk(d.text, source_doc_id=d.filename)]
        out[strategy.name] = (emb.calls, len(chunks))
    return out


def test_the_semantic_bullet_quotes_the_counts_the_code_produces() -> None:
    counts = _chunking_embeds()
    sentence_embeds, n_chunks = counts["semantic"]
    bullet = next(b for b in _takeaways().split("\n- ") if b.startswith("**Semantic chunking"))
    assert f"{sentence_embeds} embed calls" in bullet
    assert f"against {n_chunks} chunk embeddings" in bullet
    assert sentence_embeds > n_chunks, (
        "the claim is that deciding costs more embeds than chunking yields"
    )


def test_the_strategies_the_bullet_says_embed_nothing_while_chunking_do_not() -> None:
    counts = _chunking_embeds()
    for name in ("fixed-size", "recursive", "structure-aware"):
        assert counts[name][0] == 0, name


def test_snippet_hit_never_exceeds_recall_in_any_committed_run() -> None:
    runs = sorted((ROOT / "results").glob("canonical__*.json"))
    assert runs
    for path in runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        for k, recall in run["recall_at_k"].items():
            assert run["snippet_hit_at_k"][k] <= recall, (path.name, k)


def test_why_it_holds_each_snippet_is_only_in_its_expected_document() -> None:
    docs = {d.filename: d.text for d in load_corpus()}
    for q in load_queries():
        assert [f for f, text in docs.items() if q.expected_snippet in text] == [q.expected_doc], (
            q.id
        )


def test_no_bullet_claims_the_gap_is_always_wide() -> None:
    takeaways = _takeaways()
    assert not re.search(r"≪\s*recall\*\*,?\s*always", takeaways)
    assert "not the chunking decision itself" not in takeaways
