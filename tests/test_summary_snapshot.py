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

#: The command that refreshes the committed fixtures. `--canonical-out` is not
#: optional: without it `run_matrix` writes `results/<timestamp>__summary.md` and
#: timestamped per-strategy JSONs, so the hint named a command that never touched
#: `results/summary.md` -- the very file the failing assertion is about -- and
#: dropped five gitignored scratch files into `results/` that turned five other
#: tests in this module red (#190). The flag's own `--help` says what this should
#: have said: "Use to refresh the committed snapshot fixtures that
#: tests/test_summary_snapshot.py locks."
#:
#: `test_the_regen_hint_is_a_command_run_matrix_accepts` parses the command back
#: out of this string and feeds it to `run_matrix`'s own parser, so the hint
#: cannot drift from the flag a second time.
REGEN_COMMAND = "python scripts/run_matrix.py --canonical-out"

REGEN_HINT = (
    "Regenerate the committed fixtures:\n"
    f"  {REGEN_COMMAND}\n"
    "Then inspect with `git diff results/` before committing."
)

#: What ".gitignore" means by "committed": `results/*` is ignored except
#: `summary.md` and `canonical__*.json`. `_committed_run_jsons` globbed `*.json`
#: instead, which is a different set (#190).
_CANONICAL_GLOB = "canonical__*.json"

#: The one field in a committed result JSON that cannot be reproduced, because it
#: is a wall-clock measurement of the machine that ran it. Every other field is
#: bit-for-bit deterministic -- `HashEmbedder` is deterministic and two fresh
#: runs plus the committed fixtures agree exactly (measured, #190). Named as a
#: set and pinned by `test_only_wall_clock_is_excluded_from_the_pipeline_lock`,
#: so adding a second exclusion is a deliberate act rather than a quiet widening.
_HOST_DEPENDENT_FIELDS = frozenset({"wall_clock_ms"})


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
    """The committed `results/canonical__*.json` files, sorted.

    `canonical__`, not `*.json` (#190). The old glob matched every JSON in
    `results/`, and its docstring described the *timestamped* naming because it
    was written before `--canonical-out` existed and was never narrowed when the
    canonical convention arrived. `.gitignore` already states the set this
    function is named after::

        results/*
        !results/summary.md
        !results/canonical__*.json

    so a function called `_committed_run_jsons` was reading files that are not
    committed and never will be. Running the regen command the failing assertion
    itself printed left five gitignored scratch JSONs in `results/` and turned
    five tests in this module red, reporting them as "committed".
    """
    return sorted(RESULTS_DIR.glob(_CANONICAL_GLOB))


def _runs_in_strategy_order() -> list[RetrievalRun]:
    # A dict comprehension over a sorted list resolves a duplicate strategy key
    # by whichever sorts LATER, silently. That is why the summary assertion
    # stayed green while the per-strategy count assertion went red in #190: a
    # timestamp prefix starts with a digit (0x30-0x39) and `canonical__` starts
    # with `c` (0x63), so the canonical file happened to sort last and happened
    # to win. The lock's correctness rested on that byte ordering, nothing stated
    # it, and an inversion would have compared `summary.md` against SCRATCH.
    # Narrowing the glob above makes a duplicate unreachable through the script;
    # this makes it loud if it ever becomes reachable another way.
    paths_by_name: dict[str, list[Path]] = {}
    for path in _committed_run_jsons():
        paths_by_name.setdefault(path.stem.split("__", 1)[1], []).append(path)
    duplicates = {name: [p.name for p in ps] for name, ps in paths_by_name.items() if len(ps) > 1}
    assert not duplicates, (
        f"more than one committed JSON per strategy: {duplicates}. The snapshot "
        f"would otherwise silently compare against whichever filename sorts "
        f"last.\n{REGEN_HINT}"
    )
    by_name = {name: ps[0] for name, ps in paths_by_name.items()}
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


def test_render_summary_collapses_newline_in_strategy_name_so_row_stays_one_line() -> None:
    # Newline sibling of #100 at the same site (mirrors embedding-model-shootout
    # #105): a GFM row is a single physical line, so a `\n`/`\r` in `strategy_name`
    # (reachable via `RetrievalRun.from_json` on an external result file, or a BYO
    # Strategy name — the same input the pipe-escape guards) splits one result
    # across two lines and breaks every row after it. The fix collapses `[\r\n]+`
    # -> a single space so the row stays on one line.
    run = RetrievalRun(
        strategy_name="ev\nil\r\nx",
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
    row_lines = [line for line in md.splitlines() if line.startswith("|")]
    # header + separator + exactly one data row — no extra physical line from the
    # embedded newlines.
    assert len(row_lines) == 3, f"newline split the row: {row_lines}"
    data_row = row_lines[2]
    assert "ev il x" in data_row
    assert "\n" not in data_row
    assert "\r" not in data_row


def test_render_summary_collapses_newline_in_embedder_name_header_so_it_stays_one_line() -> None:
    # Sibling of #100/#130 in the SAME function, one cell over: `_render_summary`
    # interpolates the free-form `embedder_name` (`RetrievalRun.embedder_model`,
    # loaded verbatim via `from_json` on an external/hand-edited result file, or a
    # BYO embedder's `model_name`) into the `_embedder_:` header line. A `\n`/`\r`
    # there splits the header across two physical lines and breaks the surrounding
    # inline-code span, corrupting the front-page benchmarks doc — the row-
    # delimiter class the #130 fix closed for the `strategy_name` row cell but
    # never applied to this header cell. (Pipe-escape is intentionally NOT applied
    # here: the cell is inside an inline-code span where `\|` renders literally.)
    run = RetrievalRun(
        strategy_name="fixed",
        embedder_model="ev\nil\r\nmodel",
        dataset_version="v1",
        n_queries=3,
        n_chunks_total=10,
        recall_at_k={1: 0.5, 3: 0.6, 5: 0.7},
        snippet_hit_at_k={1: 0.4, 3: 0.5, 5: 0.6},
        per_query=(),
        wall_clock_ms=12.0,
    )
    md = _render_summary([run], run.embedder_model)
    header_line = next(line for line in md.splitlines() if line.startswith("_embedder_:"))
    # The whole embedder name stays on the one header line — no CR/LF leaked
    # through to split it.
    assert "ev il model" in header_line
    assert "\n" not in header_line
    assert "\r" not in header_line


def test_render_summary_neutralizes_backtick_in_embedder_name_header_so_span_stays_one() -> None:
    # Sibling of #133 in the SAME cell: `_render_summary` wraps the free-form
    # `embedder_name` in an inline-code span (`` `{embedder_name}` ``). #133
    # collapsed the newline, but a BACKTICK in the name (same external/from_json
    # reachability) prematurely CLOSES the span — `` `a`b`c` `` splits into two
    # code spans and leaks the middle out as prose, corrupting the front-page
    # benchmarks doc. The backtick must be neutralized so the identifier renders
    # as a single inline-code span.
    run = RetrievalRun(
        strategy_name="fixed",
        embedder_model="team/model`v2`beta",
        dataset_version="v1",
        n_queries=3,
        n_chunks_total=10,
        recall_at_k={1: 0.5, 3: 0.6, 5: 0.7},
        snippet_hit_at_k={1: 0.4, 3: 0.5, 5: 0.6},
        per_query=(),
        wall_clock_ms=12.0,
    )
    md = _render_summary([run], run.embedder_model)
    header_line = next(line for line in md.splitlines() if line.startswith("_embedder_:"))
    # Exactly one opening and one closing backtick delimit the code span — no
    # stray backtick from the model name survives to split it.
    assert header_line.count("`") == 2
    # The identifier's parts all remain inside that single span (backticks
    # neutralized to straight quotes), none leaked out as prose.
    assert "team/model'v2'beta" in header_line


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


# ---------------------------------------------------------------------------
# The lock on the MEASUREMENT, not on the renderer (#190)
# ---------------------------------------------------------------------------
#
# Everything above compares `summary.md` to the committed JSONs, and the JSONs to
# the loader. Nothing asked whether those JSONs reflect what the code produces, so
# a chunking strategy whose recall regressed would leave both the fixtures and the
# published table exactly as committed and this module exactly as green. The
# published table is the repo's headline result.
#
# The module docstring explains the original choice -- "The committed JSONs carry
# `wall_clock_ms` baked in, so feeding them back through the renderer produces a
# deterministic markdown" -- and that reason is true of `wall_clock_ms` and of
# nothing else. Measured (#190): two fresh runs and the committed fixtures agree
# bit-for-bit on every other field, and the whole five-strategy matrix takes 0.2s.


def _run_matrix_into(out_dir: Path) -> dict[str, dict]:
    """Run the real five-strategy matrix and return `{strategy: json}`."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_matrix.py"),
            "--results-dir",
            str(out_dir),
            "--canonical-out",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"run_matrix exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    produced = {
        p.stem.split("__", 1)[1]: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(out_dir.glob(_CANONICAL_GLOB))
    }
    assert set(produced) == set(STRATEGY_ORDER), (
        f"a fresh run produced {sorted(produced)}, expected {sorted(STRATEGY_ORDER)}"
    )
    return produced


def _comparable(blob: dict) -> dict:
    return {k: v for k, v in blob.items() if k not in _HOST_DEPENDENT_FIELDS}


@pytest.mark.parametrize("strategy", STRATEGY_ORDER)
def test_committed_json_matches_a_fresh_pipeline_run(strategy: str, tmp_path: Path) -> None:
    """The committed fixture must equal what the pipeline produces today.

    This is the assertion that can see a strategy regression.
    `test_committed_summary_md_matches_render_from_committed_results` cannot: it
    re-renders the committed JSONs and compares the result to the markdown beside
    them, so a regression that changed recall would leave both stale and it green.
    """
    produced = _run_matrix_into(tmp_path)
    committed = json.loads(
        (RESULTS_DIR / f"canonical__{strategy}.json").read_text(encoding="utf-8")
    )
    assert _comparable(produced[strategy]) == _comparable(committed), (
        f"the committed fixture for {strategy!r} no longer matches what "
        f"scripts/run_matrix.py produces. If the change is intended, regenerate; "
        f"if it is not, a strategy regressed.\n{REGEN_HINT}"
    )


def test_only_wall_clock_is_excluded_from_the_pipeline_lock(tmp_path: Path) -> None:
    """The exclusion must stay exactly one field, discovered from the data.

    A lock whose exclusion list can grow quietly stops being a lock. This
    discovers the JSON's own key set and asserts the excluded part of it is
    precisely `wall_clock_ms`, so a second host-dependent field has to be added
    here on purpose -- and asserts the *included* part is non-trivial, so the
    comparison above cannot be narrowed to nothing.
    """
    produced = _run_matrix_into(tmp_path)
    keys = set(produced[STRATEGY_ORDER[0]])
    assert {"wall_clock_ms"} == _HOST_DEPENDENT_FIELDS, _HOST_DEPENDENT_FIELDS
    assert keys >= _HOST_DEPENDENT_FIELDS, (
        f"the excluded field is not in the JSON at all; exclusion is dead: "
        f"{_HOST_DEPENDENT_FIELDS - keys}"
    )
    compared = keys - _HOST_DEPENDENT_FIELDS
    assert {
        "recall_at_k",
        "snippet_hit_at_k",
        "n_chunks_total",
        "n_queries",
        "per_query",
        "strategy_name",
        "embedder_model",
        "dataset_version",
    } <= compared, (
        f"the pipeline lock stopped comparing a field that carries a result: "
        f"compared={sorted(compared)}"
    )


def test_the_pipeline_is_deterministic_on_everything_it_locks(tmp_path: Path) -> None:
    """Anti-vacuous, and the premise the exclusion rests on.

    If a second field turned out to be non-deterministic, the lock above would be
    flaky rather than wrong -- and a flaky lock gets deleted. Two fresh runs must
    agree on every compared field, so the exclusion is justified by measurement
    rather than by assumption.
    """
    first = _run_matrix_into(tmp_path / "a")
    second = _run_matrix_into(tmp_path / "b")
    for strategy in STRATEGY_ORDER:
        assert _comparable(first[strategy]) == _comparable(second[strategy]), (
            f"{strategy!r} is not reproducible across two runs of the same code; "
            f"_HOST_DEPENDENT_FIELDS is understated"
        )


def test_the_regen_hint_is_a_command_run_matrix_accepts() -> None:
    """The hint's command must parse under `run_matrix`'s own parser (#190).

    The hint named `python scripts/run_matrix.py` for as long as
    `--canonical-out` has existed, so it printed a command that wrote
    `results/<timestamp>__summary.md` and left `results/summary.md` untouched.
    Asserting the text is not enough -- that just pins a string. This feeds the
    flags to the real parser, so the hint goes red if the flag is renamed or
    removed.
    """
    import shlex

    from scripts import run_matrix

    assert REGEN_COMMAND in REGEN_HINT, "the hint must actually print the command"
    tokens = shlex.split(REGEN_COMMAND)
    assert tokens[:3] == ["python", "scripts/run_matrix.py"][:2] + [tokens[2]], tokens
    flags = tokens[2:]
    assert flags, "the hint must pass the flag that writes the committed fixtures"
    parser = run_matrix._build_parser() if hasattr(run_matrix, "_build_parser") else None
    if parser is None:
        # `run_matrix` builds its parser inside `main`; exercise the real entry
        # point instead, which is the stronger check anyway.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            rc = run_matrix.main([*flags, "--results-dir", d])
        assert rc == 0, f"the regen command the hint prints exited {rc}"
        return
    parser.parse_args(flags)  # pragma: no cover - only if a _build_parser appears


def test_running_the_documented_regen_command_leaves_the_suite_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that started #190.

    The old hint's command dropped five gitignored timestamped JSONs into
    `results/`, and `_committed_run_jsons`'s `*.json` glob then reported them as
    committed, turning five tests in this module red. Simulated here by writing
    scratch files with the exact names the timestamped path produces and asserting
    the committed-fixture helpers are unaffected.
    """
    scratch = [RESULTS_DIR / f"20260911T005939__{name}.json" for name in STRATEGY_ORDER] + [
        RESULTS_DIR / "20260911T005939__summary.md"
    ]
    created: list[Path] = []
    try:
        for path in scratch:
            if path.exists():  # pragma: no cover - a real regen left one behind
                continue
            path.write_text("{}", encoding="utf-8")
            created.append(path)
        found = {p.name for p in _committed_run_jsons()}
        assert found == {f"canonical__{name}.json" for name in STRATEGY_ORDER}, (
            f"scratch files leaked into the committed set: {sorted(found)}"
        )
        # And the assertions that went red must still pass with scratch present.
        runs = _runs_in_strategy_order()
        assert len(runs) == len(STRATEGY_ORDER)
    finally:
        for path in created:
            path.unlink(missing_ok=True)


def test_a_duplicate_strategy_json_is_an_error_not_a_last_wins_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tested directly, because narrowing the glob made it unreachable.

    With `_CANONICAL_GLOB` there can only be one `canonical__<strategy>.json` per
    strategy, so the duplicate branch is defence in depth rather than a live path
    — and a guard reachable only through another guard's failure is one a later
    edit can orphan. Injecting the duplicate list exercises the contract itself.

    What it replaces: a dict comprehension over a sorted list, which resolved a
    duplicate by whichever filename sorted later, silently. In #190 that was the
    difference between `canonical__semantic.json` (`c`, 0x63) and
    `20260911T005939__semantic.json` (`2`, 0x32) — the lock was correct by byte
    ordering, and an inversion would have compared `summary.md` against scratch.
    """
    duplicated = [
        RESULTS_DIR / "canonical__semantic.json",
        RESULTS_DIR / "20260911T005939__semantic.json",
    ]
    monkeypatch.setattr(
        sys.modules[__name__], "_committed_run_jsons", lambda: duplicated, raising=True
    )
    with pytest.raises(AssertionError, match="more than one committed JSON per strategy"):
        _runs_in_strategy_order()
