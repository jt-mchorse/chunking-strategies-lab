"""An unmeasured metric cell is never published as a number (#160).

`_render_summary` derived its column set from ONE of the two metric maps —
`sorted(runs[0].recall_at_k)` — and rendered BOTH with `.get(k, 0)`. Two
operands, one derivation.

`#76` fixed the hardcoded-1/3/5 version of exactly this. Its comment, still in
`run_matrix.py`, says the renderer "showed 0.000 for cells whose JSONs held
real values". That sentence stayed true afterwards, with `runs[0].recall_at_k`
standing in for the hardcoded list.

Two distinct failures, so two fixes, and these tests keep them apart:

- A SINGLE run whose two maps disagree is incoherent — one strategy, one query
  set, one `ks` — and `from_json` now rejects it at the seam.
- SEPARATE runs at different `k` are legitimate; each is internally coherent.
  The renderer has to represent them honestly, which means widening `ks` to the
  union AND rendering a genuinely absent cell as a non-number.

Assertions name the measured pre-fix output, because the pre-fix code raised
nothing — it published a confident wrong table.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from chunking_lab.metrics import RetrievalRun

_ROOT = Path(__file__).resolve().parents[1]


def _load_run_matrix():
    spec = importlib.util.spec_from_file_location(
        "run_matrix_for_tests", _ROOT / "scripts" / "run_matrix.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_matrix = _load_run_matrix()


def _payload(name: str, recall: dict[str, float], snippet: dict[str, float]) -> dict:
    return {
        "strategy_name": name,
        "embedder_model": "hash",
        "dataset_version": "v0",
        "n_queries": 10,
        "n_chunks_total": 100,
        "recall_at_k": recall,
        "snippet_hit_at_k": snippet,
        "per_query": [],
        "wall_clock_ms": 12.0,
        "notes": [],
    }


def _data_rows(rendered: str) -> list[str]:
    """The table's strategy rows — not the header, separator, or prose."""
    return [
        line
        for line in rendered.splitlines()
        if line.startswith("| ") and not line.startswith("| strategy") and "---" not in line
    ]


class TestFromJsonRejectsAnIncoherentRun:
    def test_divergent_key_sets_raise(self) -> None:
        with pytest.raises(ValueError, match="must cover the same k values") as exc:
            RetrievalRun.from_json(_payload("fixed", {"1": 0.5, "3": 0.6}, {"5": 0.7}))
        message = str(exc.value)
        assert "recall_at_k=[1, 3]" in message, "the message must name both key sets"
        assert "snippet_hit_at_k=[5]" in message

    def test_a_subset_is_also_rejected(self) -> None:
        # Not just disjoint sets — a missing k on one side is the same defect.
        with pytest.raises(ValueError, match="must cover the same k values"):
            RetrievalRun.from_json(_payload("fixed", {"1": 0.5, "3": 0.6}, {"1": 0.4}))

    def test_matching_key_sets_still_load(self) -> None:
        run = RetrievalRun.from_json(_payload("fixed", {"1": 0.5, "3": 0.6}, {"1": 0.4, "3": 0.45}))
        assert sorted(run.recall_at_k) == [1, 3]
        assert sorted(run.snippet_hit_at_k) == [1, 3]

    def test_the_writers_own_output_round_trips(self) -> None:
        """The guard must not reject anything `to_json` can produce."""
        run = RetrievalRun.from_json(_payload("fixed", {"1": 0.5, "3": 0.6}, {"1": 0.4, "3": 0.45}))
        assert RetrievalRun.from_json(run.to_json()) == run


class TestHeterogeneousRunsRenderHonestly:
    """Separate runs at different k — legitimate input, must not be faked."""

    @staticmethod
    def _runs() -> list[RetrievalRun]:
        return [
            RetrievalRun.from_json(_payload("fixed", {"1": 0.5, "3": 0.6}, {"1": 0.4, "3": 0.45})),
            RetrievalRun.from_json(_payload("recursive", {"5": 0.90}, {"5": 0.88})),
        ]

    def test_the_second_runs_real_numbers_appear(self) -> None:
        rendered = run_matrix._render_summary(self._runs(), "hash")
        recursive_row = next(r for r in _data_rows(rendered) if "recursive" in r)
        assert "0.900" in recursive_row, (
            "recursive measured recall@5 = 0.90. Pre-fix its entire row read "
            "0.000 | 0.000 | 0.000 | 0.000 and a reader would conclude the "
            "strategy failed."
        )
        assert "0.880" in recursive_row, "recursive measured snippet-hit@5 = 0.88"

    def test_the_union_of_k_values_becomes_the_columns(self) -> None:
        rendered = run_matrix._render_summary(self._runs(), "hash")
        header = next(line for line in rendered.splitlines() if line.startswith("| strategy"))
        for k in (1, 3, 5):
            assert f"recall@{k}" in header
            assert f"snippet-hit@{k}" in header

    def test_an_unmeasured_cell_is_not_a_number(self) -> None:
        rendered = run_matrix._render_summary(self._runs(), "hash")
        recursive_row = next(r for r in _data_rows(rendered) if "recursive" in r)
        assert "0.000" not in recursive_row, (
            "an unmeasured cell must never publish a number — 0.000 reads as a "
            "measured floor, and handoff section 10 forbids invented benchmarks"
        )
        assert recursive_row.count(run_matrix._ABSENT_CELL) == 4

    def test_a_genuine_zero_is_still_rendered_as_zero(self) -> None:
        """The distinction the placeholder exists to make.

        A strategy that really scored 0.0 must still show 0.000, otherwise the
        fix has traded a false number for a false absence.
        """
        runs = [RetrievalRun.from_json(_payload("fixed", {"1": 0.0}, {"1": 0.0}))]
        row = _data_rows(run_matrix._render_summary(runs, "hash"))[0]
        assert "0.000" in row
        assert run_matrix._ABSENT_CELL not in row


class TestTheCanonicalPathIsUnchanged:
    def test_uniform_ks_render_with_no_placeholder(self) -> None:
        runs = [
            RetrievalRun.from_json(
                _payload(
                    "fixed",
                    {"1": 0.1, "3": 0.2, "5": 0.3},
                    {"1": 0.4, "3": 0.5, "5": 0.6},
                )
            ),
            RetrievalRun.from_json(
                _payload(
                    "recursive",
                    {"1": 0.7, "3": 0.8, "5": 0.9},
                    {"1": 0.11, "3": 0.22, "5": 0.33},
                )
            ),
        ]
        rows = _data_rows(run_matrix._render_summary(runs, "hash"))
        assert len(rows) == 2
        for row in rows:
            assert run_matrix._ABSENT_CELL not in row, (
                "the canonical --ks path must render byte-identically; the "
                "committed summary snapshot depends on it"
            )
        assert "0.100" in rows[0]
        assert "0.900" in rows[1]

    def test_empty_run_list_keeps_the_default_columns(self) -> None:
        header = next(
            line
            for line in run_matrix._render_summary([], "hash").splitlines()
            if line.startswith("| strategy")
        )
        for k in (1, 3, 5):
            assert f"recall@{k}" in header
