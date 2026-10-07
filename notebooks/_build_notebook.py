"""Build `notebooks/comparison.ipynb` programmatically.

The notebook itself is JSON; building it from a Python script keeps the
source readable and the cell list reviewable as code. Run this once and
commit the resulting `.ipynb`:

    python notebooks/_build_notebook.py

Then execute the notebook in place to populate outputs:

    jupyter nbconvert --to notebook --inplace --execute notebooks/comparison.ipynb
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


def _markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def _code(source: str) -> nbformat.NotebookNode:
    # No trailing newline: ruff >= 0.16 formats notebook cells and strips it, so
    # a rebuilt notebook carrying one failed CI's `ruff format --check` while
    # the snapshot lock (which rstrips) stayed green.
    return nbformat.v4.new_code_cell(source.rstrip("\n"))


_INTRO = dedent(
    """\
    # chunking-strategies-lab — comparison notebook (#4)

    Loads every per-strategy result JSON under `results/`, picks the **most recent run per strategy** (by filename timestamp), and renders three charts:

    1. **Recall@k** across strategies — the IR metric.
    2. **Snippet-hit@k** across strategies — the answer-faithfulness proxy ([D-008](../MEMORY/core_decisions_human.md#d-008)).
    3. **Wall-clock latency** per strategy — chunk + embed + retrieve total.

    Honest takeaways at the bottom. Re-running this notebook from a fresh `results/` directory produces the same shape; numbers move when (a) the corpus + query substrate changes ([D-002](../MEMORY/core_decisions_human.md#d-002)) or (b) the embedder changes (`hash` → `minilm` via `pip install -e '.[sbert]'` then `python scripts/run_matrix.py --embedder minilm`).
    """
)


_LOAD_CELL = dedent(
    '''\
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Notebook lives in `notebooks/`; results live in `../results/`.
    REPO_ROOT = Path.cwd() if Path.cwd().name != "notebooks" else Path.cwd().parent
    RESULTS_DIR = REPO_ROOT / "results"


    def _wall_label(ms: float) -> str:
        """A wall-clock value as `run_matrix._wall_clock_cell` publishes it (#210, D-016).

        `0.0` is D-009's "not measured" default for a pre-D-009 JSON, so it is
        the em dash, never `0ms` -- zero is impossible elapsed time, and the
        flattering artefact: it wins any "which is fastest" read. A positive
        value that `.0f` collapses to zero gets `.3g` instead.
        """
        if ms == 0.0:
            return "—"
        rendered = f"{ms:.0f}"
        if float(rendered) == 0.0:
            rendered = f"{ms:.3g}"
        return f"{rendered}ms"


    def _rate_label(value: float) -> str:
        """A recall / snippet-hit rate as `run_matrix._render_rate` publishes it (#232).

        Never a fabricated zero (#198) and never a fabricated one (#226): a
        non-zero rate that `.3f` rounds to `0.000`, or a non-1.0 rate it rounds
        to `1.000`, is widened until it no longer reads as the extreme. This
        cell printed both rates with a bare `.3f`, so one miss in 2001 queries
        printed `1.000`, byte-identical to a perfect run.
        """
        rendered = f"{value:.3f}"
        if value != 0.0 and float(rendered) == 0.0:
            return f"{value:.3g}"
        if value != 1.0 and float(rendered) == 1.0:
            for places in range(4, 18):
                wide = f"{value:.{places}f}"
                if float(wide) != 1.0:
                    return wide
            return repr(value)
        return rendered


    def _stamp_rank(stamp: str) -> tuple[int, str]:
        """Recency key so a fresh run beats the committed `canonical` baseline.

        Timestamped runs use a digit prefix (`YYYYMMDDThhmmss`); the committed
        fixtures use the literal `canonical`. Plain `>` would rank `canonical`
        above every digit-prefixed stamp ('c' > '0'-'9'), so a fresh operator
        run alongside the canonical files would be silently ignored. Rank
        `canonical` lowest instead; any timestamped run supersedes it.
        """
        return (0, "") if stamp == "canonical" else (1, stamp)


    def _load_latest_per_strategy(results_dir: Path) -> list[dict]:
        """Pick the most recent JSON per `strategy_name`."""
        files = sorted(results_dir.glob("*.json"))
        latest_by_strategy: dict[str, dict] = {}
        latest_stamp: dict[str, str] = {}
        for p in files:
            stamp = p.name.split("__")[0]
            payload = json.loads(p.read_text(encoding="utf-8"))
            name = payload["strategy_name"]
            if name not in latest_stamp or _stamp_rank(stamp) > _stamp_rank(latest_stamp[name]):
                latest_by_strategy[name] = payload
                latest_stamp[name] = stamp
        canonical = ["fixed-size", "recursive", "semantic", "late-chunking", "structure-aware"]
        return [latest_by_strategy[n] for n in canonical if n in latest_by_strategy]


    def _one_or_mixed(values: list) -> str:
        """The single value every run shares, or `MIXED (a, b)` when they differ (#221).

        The loader keeps the newest file per strategy, so one fresh `--embedder
        minilm` run can sit beside four canonical HashEmbedder files (#211's
        population). Titling the charts with `runs[0]`'s value then labelled
        every bar with an embedder that produced only one of them.
        """
        distinct = sorted({str(v) for v in values})
        if not distinct:
            return "?"
        return distinct[0] if len(distinct) == 1 else f"MIXED ({', '.join(distinct)})"


    runs = _load_latest_per_strategy(RESULTS_DIR)
    embedder = _one_or_mixed([r["embedder_model"] for r in runs])
    n_queries = _one_or_mixed([r["n_queries"] for r in runs])
    mixed = embedder.startswith("MIXED") or n_queries.startswith("MIXED")
    print(f"Loaded {len(runs)} strategy runs · embedder={embedder} · n_queries={n_queries}")
    if mixed:
        print("  WARNING: these runs are not comparable as one chart; per-run values below.")
    for r in runs:
        name = r["strategy_name"]
        n_chunks = r["n_chunks_total"]
        # Report the largest k actually present, not a hardcoded 5 — a run from a
        # non-default `--ks` (a supported run_matrix.py flag) may omit 5. Mirrors
        # run_matrix.py's `max(ks)` summary line; `kmax` is 5 on the default --ks.
        kmax = max(int(k) for k in r["recall_at_k"])
        recall_top = float(r["recall_at_k"][str(kmax)])
        snippet_top = float(r["snippet_hit_at_k"][str(kmax)])
        wall = _wall_label(float(r.get("wall_clock_ms", 0.0)))
        per_run = f"  embedder={r['embedder_model']}  n_queries={r['n_queries']}" if mixed else ""
        print(
            f"  {name:18} chunks={n_chunks:3d}  recall@{kmax}={_rate_label(recall_top)}  snippet-hit@{kmax}={_rate_label(snippet_top)}  wall={wall}{per_run}"
        )
    '''
)


_CAVEAT = dedent(
    """\
    > **Embedder caveat.** When the loaded runs above use `HashEmbedder`, the *quality* axes (recall, snippet-hit) reflect the runner producing structurally-valid output, not the strategies' real retrieval quality — `HashEmbedder` vectors are effectively random per text. The *latency* axis is real either way (chunking + cosine retrieval is real work). Re-run with `--embedder minilm` (after `pip install -e '.[sbert]'`) for honest quality numbers.
    """
)


_RECALL_CELL = dedent(
    """\
    import numpy as np

    # Derive k values from the loaded runs instead of hardcoding 1/3/5, so a
    # non-default `--ks` (a supported run_matrix.py flag) renders the k's it
    # actually produced rather than crashing on a missing key (#82). The UNION
    # across runs, as run_matrix.py's summary has taken since #160: the loader
    # picks the newest file per strategy, so one fresh `--ks 1,10` run sits
    # beside four canonical `--ks 1,3,5` files, and `runs[0]`'s keys indexed
    # into the others raised KeyError (#210). Equals [1, 3, 5] on default.
    ks = sorted({int(k) for r in runs for k in r["recall_at_k"]}) if runs else [1, 3, 5]
    strategies = [r["strategy_name"] for r in runs]
    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for i, k in enumerate(ks):
        # A k this run did not measure is NaN, which draws no bar: absent, never
        # a fabricated 0 (the #160 rule for a missing cell).
        vals = [float(r["recall_at_k"].get(str(k), np.nan)) for r in runs]
        # Center the grouped bars around each tick for any len(ks); == (i - 1) for
        # the default 3 k's, so the canonical chart is unchanged.
        ax.bar(x + (i - (len(ks) - 1) / 2) * width, vals, width, label=f"recall@{k}")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=12)
    ax.set_ylabel("Recall (proportion of queries)")
    ax.set_title(f"Recall@k by strategy · embedder={embedder} · n_queries={n_queries}")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()
    """
)


_SNIPPET_INTRO = dedent(
    """\
    ## Chart 2 — Snippet-hit@k across strategies

    `snippet_hit@k` is the answer-faithfulness proxy this lab uses (D-008): the expected answer snippet is present as a substring in at least one of the top-k retrieved chunks. Structural rather than semantic, but cheap, hermetic, and exactly the metric that catches strategies that *fragment* the relevant passage across chunk boundaries.
    """
)


_SNIPPET_CELL = dedent(
    """\
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for i, k in enumerate(ks):
        vals = [float(r["snippet_hit_at_k"].get(str(k), np.nan)) for r in runs]
        ax.bar(x + (i - (len(ks) - 1) / 2) * width, vals, width, label=f"snippet-hit@{k}")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=12)
    ax.set_ylabel("Snippet-hit (proportion of queries)")
    ax.set_title(f"Snippet-hit@k by strategy · embedder={embedder} · n_queries={n_queries}")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()
    """
)


_LATENCY_INTRO = dedent(
    """\
    ## Chart 3 — Wall-clock latency per strategy

    Total milliseconds for `chunk + embed + retrieve` over the full query set, measured by `evaluate_strategy` (D-009). This is the latency a downstream consumer sees if they swap in this strategy on this corpus + embedder.
    """
)


_LATENCY_CELL = dedent(
    """\
    latencies = [float(r.get("wall_clock_ms", 0.0)) for r in runs]
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(strategies, latencies, color="#1f77b4")
    for xi, val in enumerate(latencies):
        # `_wall_label` from the load cell: a defaulted 0.0 is labelled "—"
        # rather than "0ms" (#210, D-016).
        ax.text(xi, val, _wall_label(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wall-clock (ms)")
    ax.set_title(f"Latency by strategy · embedder={embedder} · n_queries={n_queries}")
    ax.set_xticklabels(strategies, rotation=12)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()
    """
)


_TAKEAWAYS = dedent(
    """\
    ## Takeaways

    These are the takeaways the *latest committed run* supports — re-run `python scripts/run_matrix.py` (or `--embedder minilm` for real quality numbers) and the bullets below should be regenerated by hand from the new charts.

    - **Semantic chunking pays twice: to decide its boundaries, then for its chunks.** It produces ~3× more chunks than the others on this corpus (86 vs 28-30) and takes ~4× their wall-clock, more than linear. Its boundaries come from embedding every sentence -- 115 embed calls on this corpus before any chunk is embedded for retrieval, against 86 chunk embeddings after -- so the chunking decision itself is a large share of that cost. The fixed-size, recursive and structure-aware strategies embed nothing while chunking; for them the embedding step is the whole cost.
    - **Snippet-hit never exceeds recall, and the gap is the signal.** Every expected snippet occurs only in its expected document, so a snippet hit is always a recall hit too. The reverse fails often: recall@5 says "the right document is in the top 5" while snippet-hit@5 says "but the answer span isn't in any one of those chunks" (fixed-size: 0.917 vs 0.333). That gap is the rate at which a strategy fragments the answer across chunk boundaries. It is not always wide: at k=1, late-chunking scores 0.083 on both and structure-aware 0.000 on both.
    - **HashEmbedder caveat applies to quality only.** When `embedder` reads `HashEmbedder` above, all *quality* claims are about plumbing, not strategy. Latency numbers are real either way — chunking and cosine ranking are real work.
    - **Real quality comparison needs MiniLM.** `pip install -e '.[sbert]'` then `python scripts/run_matrix.py --embedder minilm` regenerates `results/` against the canonical embedder (D-002, D-003); re-running this notebook then shows recall and snippet-hit numbers reflecting real strategy differences. The infrastructure is ready; the operator decides when to spend the model-download budget.
    """
)


def build_notebook() -> nbformat.NotebookNode:
    """Return the notebook node `main()` writes; pure, no disk I/O.

    Factored out so `tests/test_build_notebook_snapshot.py` can compare the
    committed `comparison.ipynb` against the live build script without
    shelling out or writing a tempfile.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}

    nb.cells.append(_markdown(_INTRO))
    nb.cells.append(_code(_LOAD_CELL))
    nb.cells.append(_markdown(_CAVEAT))
    nb.cells.append(_markdown("## Chart 1 — Recall@k across strategies"))
    nb.cells.append(_code(_RECALL_CELL))
    nb.cells.append(_markdown(_SNIPPET_INTRO))
    nb.cells.append(_code(_SNIPPET_CELL))
    nb.cells.append(_markdown(_LATENCY_INTRO))
    nb.cells.append(_code(_LATENCY_CELL))
    nb.cells.append(_markdown(_TAKEAWAYS))
    return nb


def main() -> None:
    nb = build_notebook()
    out = Path(__file__).parent / "comparison.ipynb"
    nbformat.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
