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
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.corpus import load_corpus  # noqa: E402
from chunking_lab.embedder import CANONICAL_EMBEDDING_MODEL, Embedder, HashEmbedder  # noqa: E402
from chunking_lab.io_utils import atomic_write_text  # noqa: E402
from chunking_lab.metrics import (  # noqa: E402
    RetrievalRun,
    _embedder_model_name,
    evaluate_strategy,
    validate_ks,
)
from chunking_lab.queries import load_queries  # noqa: E402
from chunking_lab.strategies import (  # noqa: E402
    FixedSizeStrategy,
    LateChunkingStrategy,
    RecursiveStrategy,
    SemanticBoundaryStrategy,
    StructureAwareStrategy,
)


def _fail(message: str) -> int:
    """Print a clean ``::error::`` line to stderr and return exit code 2.

    `docs/architecture.md` states the repo's "0 / 1 / 2" contract, and #124–#127
    delivered it on `chunking_lab.validate`. This script — the one the README
    leads with, and the one that actually writes five JSONs and a summary — never
    got it, so every operator-input mistake escaped as a raw traceback at exit 1
    (#149).
    """
    sys.stderr.write(f"::error::{message}\n")
    return 2


def _build_embedder(name: str) -> Embedder:
    """Build the requested embedder.

    Raises ``ImportError`` when ``minilm`` is asked for without the ``sbert``
    extra; ``main`` translates that to the exit-2 contract.

    The import stays lazy, but the ``except ImportError`` that used to wrap it
    was **dead code**: ``MiniLMEmbedder`` is unconditionally importable — it
    lazy-imports ``sentence_transformers`` inside its own ``__init__`` precisely
    so "the package still imports cleanly without the extra" — so the import
    here cannot raise. The real ``ImportError`` comes one line later, from the
    *constructor*, outside the old ``try``, which is why the friendly message
    the author wrote never reached anyone (its ``# pragma: no cover`` was the
    tell: nothing exercised it because nothing could). And ``SystemExit(<str>)``
    exits **1**, not 2, so even a firing guard returned the wrong code while
    carrying the ``::error::`` marker this repo pairs with exit 2. Handled in
    ``main`` now, where both the import-time and construction-time cases land in
    the same arm (#149).
    """
    if name == "hash":
        return HashEmbedder()
    if name == "minilm":
        # Imported lazily so the script works on a fresh CI clone that
        # doesn't have `sbert` installed; the operator opts in.
        from chunking_lab.embedder import MiniLMEmbedder  # type: ignore[attr-defined]

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
    # `embedder_name` is free-form `RetrievalRun.embedder_model`, loaded verbatim
    # via `from_json` (no charset/newline restriction — the same external/hand-
    # edited-result-file reachability #100/#130 cite for the `strategy_name` row
    # cell). A `\r`/`\n` in it splits this header across two physical lines and
    # breaks the surrounding inline-code span, corrupting the front-page
    # docs/benchmarks.md. Collapse `[\r\n]+` -> a single space, the row-delimiter
    # sibling of the #130 fix at the one free-form cell in this function that fix
    # missed. Unlike the `strategy_name` GFM *table* cell, this cell renders
    # INSIDE an inline-code span (`` `{embedder_name}` ``) where backslash-escapes
    # are literal, so the pipe-escape half of `md_table_cell` is intentionally NOT
    # applied here (a `\|` would render a visible backslash). But that same
    # code-span context makes the BACKTICK the live threat: an embedder_model
    # carrying a `` ` `` (same external/from_json reachability as the newline)
    # prematurely closes the span, splitting `` `a`b`c` `` into two code spans and
    # leaking the middle out as prose — #133 collapsed the newline but not this.
    # Neutralize backticks to a straight quote so the identifier stays one span.
    safe_embedder = re.sub(r"[\r\n]+", " ", embedder_name).replace("`", "'")
    lines.append(f"_embedder_: `{safe_embedder}` · _n_queries_: {runs[0].n_queries if runs else 0}")
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
        #
        # A `\n`/`\r` in the same cell is the sibling corruption: a GFM row is a
        # single physical line, so an embedded newline splits one result across
        # two lines and breaks every row after it. The pipe-escape closed the
        # column-delimiter class at this site but left the row-delimiter class
        # open; collapse `[\r\n]+` -> a single space (same external-input
        # reachability as the pipe: `from_json` accepts an arbitrary
        # `strategy_name`, or a BYO Strategy name). Portfolio `md_table_cell`
        # pattern, newline sibling of embedding-model-shootout #105.
        strategy_name = r.strategy_name.replace("|", "\\|")
        strategy_name = re.sub(r"[\r\n]+", " ", strategy_name)
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

    # `--ks` first: it is pure string parsing, so the operator learns the flag is
    # wrong before waiting on an embedder build, a corpus load, and five
    # evaluations. `int()` raises on a non-numeric element, and the empty /
    # non-positive rules come from `metrics.validate_ks` — the *same* function
    # `evaluate_strategy` calls, not a second copy of the rule in the CLI.
    try:
        ks = tuple(int(k) for k in args.ks.split(",") if k.strip())
    except ValueError as e:
        return _fail(f"--ks must be a comma-separated list of integers; got {args.ks!r} ({e})")
    try:
        validate_ks(ks)
    except ValueError as e:
        return _fail(f"--ks {args.ks!r}: {e}")

    try:
        embedder = _build_embedder(args.embedder)
    except ImportError as e:
        # Covers both the lazy import and `MiniLMEmbedder.__init__`'s own guard;
        # the latter is where this actually raises. See `_build_embedder` (#149).
        return _fail(
            f"--embedder minilm requires the `[sbert]` extra: pip install -e '.[sbert]'  ({e})"
        )
    strategies = _build_strategies(embedder)
    if args.strategy is not None:
        strategies = [s for s in strategies if s.name == args.strategy]
    corpus = load_corpus()
    queries = load_queries()

    # The output directory is operator input too: a read-only filesystem, a
    # permission-denied path, or a path component that is a file makes `mkdir`
    # raise `NotADirectoryError`/`PermissionError`, which escaped as a raw
    # traceback at exit 1. Write-seam sibling of the #126 guard on
    # `validate.py --out`, on the script that writes five JSONs and a summary.
    results_dir = Path(args.results_dir)
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _fail(f"failed to create results dir {results_dir}: {e}")
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
        try:
            atomic_write_text(path, json.dumps(run.to_json(), indent=2, sort_keys=True))
        except OSError as e:
            return _fail(f"failed to write {path}: {e}")
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
    # Render the header from the SAME canonical name source that
    # `evaluate_strategy` persists into every `RetrievalRun.embedder_model`
    # (`_embedder_model_name`, per D-011) — NOT `type(embedder).__name__`. The
    # two agree only for `HashEmbedder` (no `model_name` → class-name fallback),
    # so the canonical hash path is byte-identical and the line-89 disclaimer
    # gate still fires. They diverge for any `model_name`-bearing embedder
    # (e.g. `MiniLMEmbedder` → `sentence-transformers/all-MiniLM-L6-v2`): with
    # the old class-name source, an honest `--canonical-out --embedder minilm`
    # refresh wrote a `summary.md` whose header disagreed with the committed
    # JSONs, permanently failing `test_summary_snapshot`'s re-render (which
    # reads `runs[0].embedder_model`). Using `_embedder_model_name(embedder)`
    # keeps the summary self-consistent with the fixtures and is empty-runs-safe.
    try:
        atomic_write_text(summary_path, _render_summary(runs, _embedder_model_name(embedder)))
    except OSError as e:
        return _fail(f"failed to write {summary_path}: {e}")
    print(f"\nsummary wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
