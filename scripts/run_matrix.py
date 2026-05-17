"""Run the retrieval metrics matrix across all 5 strategies + write results/.

Output layout (one JSON per strategy + one markdown summary):

  results/
    20260516T042000__fixed.json
    20260516T042000__recursive.json
    20260516T042000__semantic.json
    20260516T042000__late.json
    20260516T042000__structure.json
    summary.md

Per-strategy JSON is the `RetrievalRun.to_json()` shape. The markdown
summary aggregates recall@k and snippet-hit@k across strategies.

The embedder defaults to `HashEmbedder` (dep-free, hermetic, CI-safe).
Real quality numbers — which strategy actually wins on a given corpus —
require the operator to install the `[sbert]` extra and run with
`--embedder minilm`. Per the no-fabricated-benchmarks rule, the
markdown summary the script writes includes both the embedder name and
a one-line disclosure of which mode produced these numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.corpus import load_corpus  # noqa: E402
from chunking_lab.embedder import CANONICAL_EMBEDDING_MODEL, Embedder, HashEmbedder  # noqa: E402
from chunking_lab.metrics import RetrievalRun, evaluate_strategy  # noqa: E402
from chunking_lab.queries import load_queries  # noqa: E402
from chunking_lab.strategies import (  # noqa: E402
    FixedSizeStrategy,
    LateChunkingStrategy,
    RecursiveStrategy,
    SemanticBoundaryStrategy,
    StructureAwareStrategy,
)


def _build_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "minilm":
        # Imported lazily so the script works on a fresh CI clone that
        # doesn't have `sbert` installed; the operator opts in.
        try:
            from chunking_lab.embedder import MiniLMEmbedder  # type: ignore[attr-defined]
        except ImportError as e:  # pragma: no cover - lazy import path
            raise SystemExit(
                "::error::--embedder minilm requires the `[sbert]` extra: pip install -e '.[sbert]'"
            ) from e
        return MiniLMEmbedder(model_name=CANONICAL_EMBEDDING_MODEL)
    raise ValueError(f"unknown embedder: {name}")  # pragma: no cover - argparse rejects


def _build_strategies(embedder: Embedder):
    # LateChunkingStrategy gets the same embedder so its blended vectors
    # live in the same space as the query embedding — see `metrics.py`
    # `_materialize_vectors` for the constraint.
    return [
        FixedSizeStrategy(chunk_chars=600, overlap_chars=80),
        RecursiveStrategy(chunk_chars=600),
        SemanticBoundaryStrategy(embedder=embedder),
        LateChunkingStrategy(embedder=embedder, chunk_chars=600, overlap_chars=80),
        StructureAwareStrategy(),
    ]


def _render_summary(runs: list[RetrievalRun], embedder_name: str) -> str:
    lines: list[str] = []
    lines.append("# Chunking strategies — retrieval metrics matrix")
    lines.append("")
    lines.append(f"_embedder_: `{embedder_name}` · _n_queries_: {runs[0].n_queries if runs else 0}")
    lines.append("")
    if embedder_name == "HashEmbedder":
        lines.append(
            "> **Note.** HashEmbedder is the dep-free CI embedder; its vectors are "
            "effectively random per text. Absolute recall numbers below reflect "
            "the runner working, **not** the strategies' real retrieval quality. "
            "Run with `--embedder minilm` (after `pip install -e '.[sbert]'`) "
            "for honest numbers."
        )
        lines.append("")
    lines.append(
        "| strategy | n_chunks | recall@1 | recall@3 | recall@5 | snippet-hit@1 | snippet-hit@3 | snippet-hit@5 | wall-clock (ms) |"
    )
    lines.append(
        "| -------- | -------: | -------: | -------: | -------: | ------------: | ------------: | ------------: | --------------: |"
    )
    for r in runs:
        lines.append(
            f"| {r.strategy_name} | {r.n_chunks_total} | "
            f"{r.recall_at_k.get(1, 0):.3f} | "
            f"{r.recall_at_k.get(3, 0):.3f} | "
            f"{r.recall_at_k.get(5, 0):.3f} | "
            f"{r.snippet_hit_at_k.get(1, 0):.3f} | "
            f"{r.snippet_hit_at_k.get(3, 0):.3f} | "
            f"{r.snippet_hit_at_k.get(5, 0):.3f} | "
            f"{r.wall_clock_ms:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["hash", "minilm"],
        default="hash",
        help="Embedder to use. `hash` is dep-free; `minilm` requires the [sbert] extra.",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Where to write per-strategy JSON files and summary.md.",
    )
    p.add_argument(
        "--ks",
        default="1,3,5",
        help="Comma-separated k values for recall@k and snippet-hit@k.",
    )
    p.add_argument(
        "--dataset-version",
        default="v0",
        help="Tag stored in each JSON so consumers can join runs across versions.",
    )
    args = p.parse_args(argv)

    embedder = _build_embedder(args.embedder)
    strategies = _build_strategies(embedder)
    corpus = load_corpus()
    queries = load_queries()
    ks = tuple(int(k) for k in args.ks.split(",") if k.strip())

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")

    runs: list[RetrievalRun] = []
    for strat in strategies:
        run = evaluate_strategy(
            strat,
            corpus,
            queries,
            embedder,
            ks=ks,
            dataset_version=args.dataset_version,
        )
        path = results_dir / f"{stamp}__{run.strategy_name}.json"
        path.write_text(json.dumps(run.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"{run.strategy_name:24} n_chunks={run.n_chunks_total:4d} "
            f"recall@5={run.recall_at_k.get(5, 0):.3f} "
            f"snippet-hit@5={run.snippet_hit_at_k.get(5, 0):.3f} "
            f"wall_clock={run.wall_clock_ms:.0f}ms  →  {path}"
        )
        runs.append(run)

    summary_path = results_dir / "summary.md"
    summary_path.write_text(_render_summary(runs, type(embedder).__name__), encoding="utf-8")
    print(f"\nsummary wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
