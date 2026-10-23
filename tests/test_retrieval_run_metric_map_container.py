"""`RetrievalRun`'s two metric maps are container-guarded on the write path (#306).

#186 listed four containers `from_json` guards -- the top-level payload, **both
metric maps**, `per_query` and `notes` -- and its constructor half (PR #187)
guarded only the last two. `_validate_metric_maps` calls itself "the whole
metric-map contract, in one place, for both paths", and the dict check still
lived only in `from_json`.

Measured on origin/main 7fc1ad9::

    recall_at_k = None                 RAW TypeError  'NoneType' object is not iterable
    recall_at_k = []                   RAW AttributeError  'list' ... no attribute 'items'
    recall_at_k = {1: 0.0, "1": 0.0}   RAW TypeError  '<' not supported (str, int)
    recall_at_k = {None: 0.0, 1: 0.0}  RAW TypeError  '<' not supported (int, NoneType)

The last two are the key axis: `_validate_metric_map_keys` handed
`sorted(mapping)` to `validate_ks`, so the sort compared the keys before the
type rule that names them could run. A single bad key (`{"1": 0.0}`) already
raised the field-named `ValueError`; it takes a second key of another type for
the sort to fail first.
"""

from __future__ import annotations

from typing import Any

import pytest

from chunking_lab.metrics import QueryResult, RetrievalRun


def _query_result() -> QueryResult:
    return QueryResult(
        query_id="q1",
        expected_doc="d1",
        expected_snippet="snippet",
        retrieved_doc_ids_in_rank_order=("d1", "d2"),
        snippet_hits_in_rank_order=(True, False),
    )


def _run(**overrides: Any) -> RetrievalRun:
    base: dict[str, Any] = {
        "strategy_name": "fixed",
        "embedder_model": "hash-64",
        "dataset_version": "v1",
        "n_queries": 0,
        "n_chunks_total": 0,
        "recall_at_k": {},
        "snippet_hit_at_k": {},
        "per_query": (),
    }
    base.update(overrides)
    return RetrievalRun(**base)


_METRICS = ("recall_at_k", "snippet_hit_at_k")


# Red on origin/main: each row raised TypeError/AttributeError, not ValueError.
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize(
    ("label", "bad", "needle"),
    [
        ("none", None, "must be a dict"),
        ("list", [], "must be a dict"),
        ("list-of-floats", [0.5, 0.7], "must be a dict"),
        ("mixed-int-str-keys", {1: 0.0, "1": 0.0}, "must be an int"),
        ("mixed-int-none-keys", {None: 0.0, 1: 0.0}, "must be an int"),
    ],
)
def test_constructor_refuses_bad_metric_map_with_a_field_named_value_error(
    metric: str, label: str, bad: object, needle: str
) -> None:
    # Only one map is bad; the other is a valid empty map, so the refusal has
    # to come from the bad one and has to name it -- not the sibling, and not
    # the cross-map key-set check, which would also mention both.
    with pytest.raises(ValueError, match=rf"^{metric}\b.*{needle}"):
        _run(**{metric: bad})


# Controls -- green before and after.


def test_control_single_str_key_was_already_a_field_named_value_error() -> None:
    # The sort only fails with two incomparable keys; one bad key reached
    # `validate_ks`'s type rule on main too. Pins that the #306 change kept it.
    with pytest.raises(ValueError, match=r"^recall_at_k keys: every k in ks must be an int"):
        _run(recall_at_k={"1": 0.0}, snippet_hit_at_k={"1": 0.0})


def test_control_valid_maps_still_construct() -> None:
    run = _run(
        n_queries=1,
        n_chunks_total=2,
        recall_at_k={1: 1.0, 3: 1.0},
        snippet_hit_at_k={1: 1.0, 3: 1.0},
        per_query=(_query_result(),),
    )
    assert run.recall_at_k == {1: 1.0, 3: 1.0}
    assert RetrievalRun.from_json(run.to_json()) == run


def test_control_non_positive_key_message_unchanged() -> None:
    # Swapping `sorted` for `list` must not reorder the sign-rule message:
    # `validate_ks` sorts its own offenders.
    with pytest.raises(
        ValueError, match=r"^recall_at_k keys: every k in ks must be positive; got \[-2, 0\]"
    ):
        _run(recall_at_k={0: 0.0, -2: 0.0}, snippet_hit_at_k={0: 0.0, -2: 0.0})


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bad", [None, [], [0.5], 0.5, "x"])
def test_control_from_json_keeps_its_own_json_object_message(metric: str, bad: object) -> None:
    # `from_json` has to refuse before `_coerce_metric_keys` iterates the map,
    # so its guard -- and its wording -- stays.
    payload = _run().to_json()
    payload[metric] = bad
    with pytest.raises(ValueError, match=rf"^{metric} must be a JSON object, got "):
        RetrievalRun.from_json(payload)
