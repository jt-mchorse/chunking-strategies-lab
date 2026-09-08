"""`__post_init__` must apply the metric-map rules `from_json` applies (#182).

#180's defect statement was *a writer that emits what its own reader rejects*.
#181 closed it for the three **scalar** numeric fields — `n_queries`,
`n_chunks_total`, `wall_clock_ms` — and its own comment named the hazard:

    Guarding two of the three numeric fields would be the half-fix this issue
    is about.

It guarded three of three *scalar* numeric fields and left the two *map*
fields, which carry more read-side validation than any scalar in the class:
key type/sign (`_coerce_metric_keys` -> `validate_ks`), value
type/finiteness/range (`_validate_metric_map`), and cross-map key-set parity
(#160). The enumeration was "numeric fields"; the population it walked was
"scalar numeric fields".

The harm does not need a round trip. `scripts/run_matrix.py` renders
`results/summary.md` from the in-memory runs, so a `nan` recall was published
as a literal `nan` cell in the results table.

The tests below are written as a **round-trip property** rather than as a list
of expected messages: for every shape, if `from_json` refuses it then
construction must refuse it too, with the same message. That is the invariant;
the twelve rows are the corpus it is checked over.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from chunking_lab import metrics as metrics_module
from chunking_lab.metrics import RetrievalRun

# --- fixtures -------------------------------------------------------------

_VALID: dict[str, Any] = {
    "strategy_name": "fixed",
    "embedder_model": "hash-64",
    "dataset_version": "v1",
    "n_queries": 2,
    "n_chunks_total": 4,
    "recall_at_k": {1: 0.5, 3: 0.75},
    "snippet_hit_at_k": {1: 0.5, 3: 0.75},
    "per_query": (),
    "wall_clock_ms": 1.5,
}


def _run(**overrides: Any) -> RetrievalRun:
    return RetrievalRun(**{**_VALID, **overrides})


#: (case id, constructor overrides). Every row is a shape that constructed,
#: serialised, and was then refused by this class's own `from_json`.
_ASYMMETRIC: list[tuple[str, dict[str, Any], str]] = [
    ("recall-nan", {"recall_at_k": {1: math.nan, 3: 0.5}}, r"must be finite"),
    ("recall-inf", {"recall_at_k": {1: math.inf, 3: 0.5}}, r"must be finite"),
    ("recall-neg-inf", {"recall_at_k": {1: -math.inf, 3: 0.5}}, r"must be finite"),
    ("recall-above-one", {"recall_at_k": {1: 2.5, 3: 0.5}}, r"must be in \[0, 1\]"),
    ("recall-below-zero", {"recall_at_k": {1: -0.5, 3: 0.5}}, r"must be in \[0, 1\]"),
    ("recall-bool-value", {"recall_at_k": {1: True, 3: 0.5}}, r"must be a number"),
    ("recall-str-value", {"recall_at_k": {1: "abc", 3: 0.5}}, r"must be a number"),
    ("recall-none-value", {"recall_at_k": {1: None, 3: 0.5}}, r"must be a number"),
    ("snippet-nan", {"snippet_hit_at_k": {1: math.nan, 3: 0.5}}, r"must be finite"),
    ("snippet-above-one", {"snippet_hit_at_k": {1: 1.5, 3: 0.5}}, r"must be in \[0, 1\]"),
    (
        "disjoint-key-sets",
        {"recall_at_k": {1: 0.5}, "snippet_hit_at_k": {5: 0.5}},
        r"must cover the same k values",
    ),
    (
        "recall-has-extra-k",
        {"recall_at_k": {1: 0.5, 3: 0.5}, "snippet_hit_at_k": {1: 0.5}},
        r"must cover the same k values",
    ),
    (
        "snippet-has-extra-k",
        {"recall_at_k": {1: 0.5}, "snippet_hit_at_k": {1: 0.5, 3: 0.5}},
        r"must cover the same k values",
    ),
    ("k-zero", {"recall_at_k": {0: 0.5}, "snippet_hit_at_k": {0: 0.5}}, r"must be positive"),
    ("k-negative", {"recall_at_k": {-3: 0.5}, "snippet_hit_at_k": {-3: 0.5}}, r"must be positive"),
    (
        "k-bool",
        {"recall_at_k": {True: 0.5}, "snippet_hit_at_k": {True: 0.5}},
        r"must be ints, not bools",
    ),
]


# --- the property ---------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "overrides", "phrase"), _ASYMMETRIC, ids=[c for c, _, _ in _ASYMMETRIC]
)
def test_construction_rejects_every_shape_the_reader_rejects(
    case: str, overrides: dict[str, Any], phrase: str
) -> None:
    with pytest.raises(ValueError, match=phrase):
        _run(**overrides)


def test_the_corpus_is_not_empty_and_covers_all_three_axes() -> None:
    """Anti-vacuous, and it counts the *axes*, not the rows.

    Sixteen rows all exercising the value axis would look like thorough
    coverage and leave the key axis and the parity axis exactly as unguarded
    as they were. That mistake — an enumeration whose population is narrower
    than its name — is what this issue is.
    """
    assert len(_ASYMMETRIC) >= 12
    ids = {case for case, _, _ in _ASYMMETRIC}
    assert len(ids) == len(_ASYMMETRIC)
    assert any(case.startswith("k-") for case in ids), "no key-axis row"
    assert any("key-sets" in case or "extra-k" in case for case in ids), "no parity-axis row"
    assert any("nan" in case or "above-one" in case for case in ids), "no value-axis row"


def test_a_valid_run_still_constructs_and_round_trips() -> None:
    """Over-rejection guard. A guard that refuses everything passes the table."""
    run = _run()
    restored = RetrievalRun.from_json(json.loads(json.dumps(run.to_json())))
    assert restored.recall_at_k == run.recall_at_k
    assert restored.snippet_hit_at_k == run.snippet_hit_at_k
    assert restored.n_queries == run.n_queries
    assert restored.wall_clock_ms == run.wall_clock_ms


@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("empty-maps", {"recall_at_k": {}, "snippet_hit_at_k": {}}),
        ("int-valued-proportions", {"recall_at_k": {1: 1, 3: 0}, "snippet_hit_at_k": {1: 0, 3: 1}}),
        ("boundary-zero-and-one", {"recall_at_k": {1: 0.0}, "snippet_hit_at_k": {1: 1.0}}),
        ("large-k", {"recall_at_k": {1000: 0.5}, "snippet_hit_at_k": {1000: 0.5}}),
    ],
    ids=["empty-maps", "int-valued-proportions", "boundary-zero-and-one", "large-k"],
)
def test_legitimate_shapes_are_not_caught_by_the_new_guard(
    case: str, overrides: dict[str, Any]
) -> None:
    """The rows an over-eager guard would break.

    `recall_at_k={}` is what `evaluate_strategy` produces for an empty `ks`
    and what the read path accepts; `1`/`0` are legitimate JSON numbers for a
    proportion, and `0.0`/`1.0` are the inclusive ends of the documented range.
    """
    run = _run(**overrides)
    RetrievalRun.from_json(json.loads(json.dumps(run.to_json())))


# --- the invariant itself, stated as a relation between the two paths -----


#: The one row where the two paths *cannot* agree on wording, and the reason
#: is structural rather than drift: by the time the reader sees a bool key,
#: `to_json` has already written it as the string `"True"`, so
#: `_coerce_metric_keys` reports an unparseable *string* key while the write
#: side still has an actual `bool` in hand. Exempting it is a claim in itself,
#: so `test_the_bool_key_row_is_exempt_for_a_real_reason` checks that claim
#: rather than taking it on trust.
_DIFFERENT_MESSAGE_BY_CONSTRUCTION = {"k-bool"}


@pytest.mark.parametrize(
    ("case", "overrides", "phrase"),
    [row for row in _ASYMMETRIC if row[0] not in _DIFFERENT_MESSAGE_BY_CONSTRUCTION],
    ids=[c for c, _, _ in _ASYMMETRIC if c not in _DIFFERENT_MESSAGE_BY_CONSTRUCTION],
)
def test_both_paths_give_the_same_message_for_the_same_shape(
    case: str, overrides: dict[str, Any], phrase: str
) -> None:
    """The two paths must not merely both reject — they must agree on why.

    `from_json`'s messages are the operator-facing contract and are pinned by
    existing tests. If the write side grew its own wording, an operator would
    get two different explanations for one defect depending on which door they
    came through — which is the drift this issue exists to end.

    The payload is built directly in `to_json` shape rather than via a real
    `RetrievalRun`, because the whole point of the fix is that such an object
    can no longer be constructed — so the reader has to be handed the shape
    without going through the constructor at all.
    """
    with pytest.raises(ValueError, match=phrase) as construct_error:
        _run(**overrides)

    payload = _payload_bypassing_construction(overrides)
    with pytest.raises(ValueError, match=phrase) as read_error:
        RetrievalRun.from_json(payload)

    assert str(construct_error.value) == str(read_error.value), case


def test_the_exemption_covers_exactly_one_row() -> None:
    """An exemption set is a place defects hide; pin its size and membership.

    If a later change made two more rows disagree, the quiet fix would be to
    add them here — and the drift this issue is about would be back with a
    test blessing it.
    """
    assert {"k-bool"} == _DIFFERENT_MESSAGE_BY_CONSTRUCTION
    assert {case for case, _, _ in _ASYMMETRIC} >= _DIFFERENT_MESSAGE_BY_CONSTRUCTION


def test_the_bool_key_row_is_exempt_for_a_real_reason() -> None:
    """The exemption's stated reason, run rather than believed.

    Both paths must still reject it, both must name the field, and both must
    name the offending key — the exemption is only about *wording*, and it is
    only legitimate because `to_json` genuinely changes the key's type before
    the reader can see it. That last part is the claim, so it is asserted:
    `str(True)` is `'True'`, not `'1'`.
    """
    overrides = dict(next(o for c, o, _ in _ASYMMETRIC if c == "k-bool"))

    with pytest.raises(ValueError, match="must be ints, not bools") as construct_error:
        _run(**overrides)
    with pytest.raises(ValueError, match="is not an integer") as read_error:
        RetrievalRun.from_json(_payload_bypassing_construction(overrides))

    for message in (str(construct_error.value), str(read_error.value)):
        assert "recall_at_k" in message
        assert "True" in message

    # The reason the two differ at all: serialisation has already changed the
    # key's type. If this ever stopped being true the exemption would be drift.
    assert str(True) == "True"
    assert _payload_bypassing_construction(overrides)["recall_at_k"] == {"True": 0.5}


def _payload_bypassing_construction(overrides: dict[str, Any]) -> dict[str, Any]:
    """A `to_json`-shaped payload for a run the constructor now refuses."""
    fields = {**_VALID, **overrides}
    return {
        "strategy_name": fields["strategy_name"],
        "embedder_model": fields["embedder_model"],
        "dataset_version": fields["dataset_version"],
        "n_queries": fields["n_queries"],
        "n_chunks_total": fields["n_chunks_total"],
        "wall_clock_ms": fields["wall_clock_ms"],
        "recall_at_k": {str(k): v for k, v in fields["recall_at_k"].items()},
        "snippet_hit_at_k": {str(k): v for k, v in fields["snippet_hit_at_k"].items()},
        "per_query": [],
        "notes": [],
    }


def test_both_paths_reach_the_same_function_object() -> None:
    """One definition, not two copies — the load-bearing structural claim.

    Copying the checks into `__post_init__` passes every assertion above and
    re-creates the exact drift #180, #181 and #182 are three instances of: two
    paths describing each other instead of sharing a rule. This asserts the
    source of both call sites names `_validate_metric_maps`, so a later edit
    that inlines either one fails here rather than one issue later.
    """
    import ast
    import inspect

    source = inspect.getsource(metrics_module)
    tree = ast.parse(source)

    callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_validate_metric_maps"
            ):
                callers.add(node.name)

    assert callers == {"__post_init__", "from_json"}, callers

    # Anti-vacuous: the walk must be able to find calls at all, or an empty
    # set would satisfy nothing and a typo in the name above would pass.
    assert "_validate_metric_maps" in source
    assert source.count("def _validate_metric_maps") == 1


def test_the_shared_validator_is_the_only_place_the_parity_rule_lives() -> None:
    """The parity message must not have been left behind in `from_json`.

    A half-migration — delegate the value checks, keep a copy of the parity
    check — is the most likely wrong edit here, and it looks identical from
    the outside until the two copies disagree.
    """
    import inspect

    source = inspect.getsource(metrics_module)
    marker = "must cover the same k values"
    assert source.count(marker) == 1, (
        f"the parity message appears {source.count(marker)} times; it must live "
        "only in `_validate_metric_maps`"
    )
