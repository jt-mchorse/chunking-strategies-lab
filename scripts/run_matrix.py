"""Run the retrieval metrics matrix across all 5 strategies + write results/.

Output layout (one JSON per strategy + one markdown summary). Filenames
default to a `YYYYMMDDTHHMMSS` timestamp prefix — those files are
gitignored regen scratch. Pass `--canonical-out` to write
`canonical__<strategy>.json` instead, which is the tracked fixture set
that tests/test_summary_snapshot.py locks:

  results/
    canonical__fixed-size.json
    canonical__recursive.json
    canonical__semantic.json
    canonical__late-chunking.json
    canonical__structure-aware.json
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
from chunking_lab.io_utils import atomic_write_text  # noqa: E402
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
    # Derive the recall@k / snippet-hit@k columns from the k values actually
    # present in the runs (set by `--ks`). Hardcoding 1/3/5 made the renderer
    # ignore a non-default `--ks` — every `.get(1/3/5, 0)` missed and the table
    # showed 0.000 for cells whose JSONs held real values (#76). The canonical
    # `--ks 1,3,5` renders byte-identically (same headers + separators), so the
    # summary snapshot is unchanged.
    ks = sorted(runs[0].recall_at_k) if runs else [1, 3, 5]
    recall_headers = " | ".join(f"recall@{k}" for k in ks)
    snippet_headers = " | ".join(f"snippet-hit@{k}" for k in ks)
    recall_seps = " | ".join("-------:" for _ in ks)
    snippet_seps = " | ".join("------------:" for _ in ks)
    lines.append(
        f"| strategy | n_chunks | {recall_headers} | {snippet_headers} | wall-clock (ms) |"
    )
    lines.append(f"| -------- | -------: | {recall_seps} | {snippet_seps} | --------------: |")
    for r in runs:
        recall_cells = " | ".join(f"{r.recall_at_k.get(k, 0):.3f}" for k in ks)
        snippet_cells = " | ".join(f"{r.snippet_hit_at_k.get(k, 0):.3f}" for k in ks)
        # `strategy_name` is the one free-form cell (every other is a formatted
        # number). It reaches here pipe-free from the five shipped strategies,
        # but a BYO `Strategy` whose `name` carries a `|`, or a `RetrievalRun`
        # loaded from external JSON via `from_json`, can inject one. GFM splits
        # table cells on unescaped pipes, so an unescaped `|` adds a spurious
        # column and corrupts the summary table's alignment. Escape `|` -> `\|`
        # (GitHub renders `\|` as a literal pipe, contributing zero column
        # delimiters) — same fix as comment `_row_to_md` (rag-kit #130),
        # `calibration.render_report` (llm-eval-harness #134), and
        # `aggregate_markdown` (embedding-model-shootout #79); applied here (#100).
        strategy_name = r.strategy_name.replace("|", "\\|")
        lines.append(
            f"| {strategy_name} | {r.n_chunks_total} | "
            f"{recall_cells} | {snippet_cells} | {r.wall_clock_ms:.0f} |"
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
    p.add_argument(
        "--canonical-out",
        action="store_true",
        help=(
            "Write per-strategy JSONs as canonical__<strategy>.json instead of "
            "timestamped filenames. Use to refresh the committed snapshot fixtures "
            "that tests/test_summary_snapshot.py locks. Default is timestamped "
            "(gitignored regen scratch)."
        ),
    )
    p.add_argument(
        "--strategy",
        choices=("fixed-size", "recursive", "semantic", "late-chunking", "structure-aware"),
        default=None,
        help=(
            "Evaluate only this strategy (default: all five). When set, no summary.md "
            "is written — a single-row summary would invalidate the snapshot lock and "
            "be misleading next to the canonical aggregate."
        ),
    )
    args = p.parse_args(argv)

    embedder = _build_embedder(args.embedder)
    strategies = _build_strategies(embedder)
    if args.strategy is not None:
        strategies = [s for s in strategies if s.name == args.strategy]
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
        prefix = "canonical" if args.canonical_out else stamp
        path = results_dir / f"{prefix}__{run.strategy_name}.json"
        atomic_write_text(path, json.dumps(run.to_json(), indent=2, sort_keys=True))
        # Report the largest computed k (most informative). Hardcoding 5 showed
        # 0.000 when `--ks` omitted 5; `max(ks)` is 5 for the default --ks 1,3,5
        # so this line is unchanged on the canonical path (#76).
        top_k = max(ks)
        print(
            f"{run.strategy_name:24} n_chunks={run.n_chunks_total:4d} "
            f"recall@{top_k}={run.recall_at_k.get(top_k, 0):.3f} "
            f"snippet-hit@{top_k}={run.snippet_hit_at_k.get(top_k, 0):.3f} "
            f"wall_clock={run.wall_clock_ms:.0f}ms  →  {path}"
        )
        runs.append(run)

    # summary.md is the tracked canonical aggregate. When --strategy
    # filters the run, a partial summary would be misleading next to
    # the canonical (and would invalidate the snapshot lock under
    # --canonical-out). Skip the summary entirely in that case — the
    # iterative dev workflow doesn't need it.
    if args.strategy is not None:
        print("\n(no summary written: --strategy filter is set)")
        return 0

    # summary.md is the tracked canonical fixture; only --canonical-out
    # overwrites it. Default runs emit a sibling timestamped summary so
    # the regen scratch is self-contained and can't desync the snapshot
    # test from the committed canonical set.
    if args.canonical_out:
        summary_path = results_dir / "summary.md"
    else:
        summary_path = results_dir / f"{stamp}__summary.md"
    atomic_write_text(summary_path, _render_summary(runs, type(embedder).__name__))
    print(f"\nsummary wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
