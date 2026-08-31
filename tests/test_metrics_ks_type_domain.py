"""`validate_ks` checked sign and not type (#158).

`chunking_lab.metrics` enforces integer-ness on the **read** path for the very
fields these `k` become — `_validate_count` uses
`isinstance(value, bool) or not isinstance(value, int)` and explains why: "bool
is excluded explicitly because it subclasses int and would otherwise pass the
type check." `validate_ks`, the **write** path that generates
`recall_at_k`/`snippet_hit_at_k`'s keys, checked `if not ks` and `if k <= 0`
and nothing more. The int-ness was asserted by the `Sequence[int]` annotation
and enforced nowhere.

Measured on `main` @ 2deb413:

    validate_ks([True])   PASSES
    validate_ks([2.5])    PASSES
    validate_ks([3.0])    PASSES
    validate_ks([nan])    PASSES
    validate_ks([inf])    PASSES
    validate_ks(['3'])    TypeError  (from the `<=`, not a ValueError)
    validate_ks([None])   TypeError

    evaluate_strategy(..., ks=(True,))  -> completed, json={"True": 0.0},
                                           from_json ValueError:
                                           invalid literal for int() with base 10: 'True'
    evaluate_strategy(..., ks=(1,True)) -> completed, json={"1": 0.0}
                                           (the True entry silently vanished)
    evaluate_strategy(..., ks=(3.0,))   -> TypeError: slice indices must be integers
                                           (raised from `scored[:max_k]`)

The `ks=(True,)` case is the one that matters most and the reason the tests
below assert the **round-trip** rather than only the guard: a guard-only test
would not have shown that the writer produced a file its own reader rejects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from chunking_lab.corpus import Document
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import RetrievalRun, evaluate_strategy, validate_ks
from chunking_lab.queries import Query
from chunking_lab.strategies.fixed import FixedSizeStrategy

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

NAN = float("nan")
INF = float("inf")


def _corpus() -> list[Document]:
    return [
        Document(filename="d1.md", text="alpha beta gamma delta epsilon " * 6),
        Document(filename="d2.md", text="one two three four five six " * 6),
    ]


def _queries() -> list[Query]:
    return [
        Query(id="q1", question="alpha beta", expected_doc="d1.md", expected_snippet="alpha"),
    ]


def _run(ks: Any) -> RetrievalRun:
    return evaluate_strategy(FixedSizeStrategy(), _corpus(), _queries(), HashEmbedder(), ks=ks)


# ----------------------------------------------------------------------
# The writer must not be able to produce a file its own reader rejects
# ----------------------------------------------------------------------


def test_ks_true_no_longer_produces_a_payload_from_json_rejects() -> None:
    # Pre-fix this run *succeeded*. `scored[:True]` takes one element, so it was
    # a mislabelled recall@1, and `to_json` emitted the key "True" — which
    # `from_json` then refused with `invalid literal for int() with base 10`.
    # `_render_summary` derives its headers from `sorted(runs[0].recall_at_k)`,
    # so the published table would have grown a `recall@True` column.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _run((True,))


def test_the_default_ks_still_round_trips_which_is_the_property_being_protected() -> None:
    # The contract the guard exists to keep: whatever `evaluate_strategy`
    # writes, `from_json` reads. Asserted positively so the test file states the
    # invariant rather than only enumerating violations of it.
    run = _run((1, 3, 5))
    payload = run.to_json()
    # Assert the KEYS (and that they survive a real JSON encode/decode), not the
    # recall values — those depend on HashEmbedder's scores, which is not what
    # this test is about, and pinning them here would make it fail for an
    # unrelated reason if the corpus fixture changed.
    assert sorted(json.loads(json.dumps(payload["recall_at_k"]))) == ["1", "3", "5"]
    restored = RetrievalRun.from_json(json.loads(json.dumps(payload)))
    assert restored.recall_at_k == run.recall_at_k
    assert restored.snippet_hit_at_k == run.snippet_hit_at_k


def test_ks_one_and_true_is_rejected_rather_than_silently_deduped() -> None:
    # `ks = tuple(dict.fromkeys(ks))` collapses True into 1 because
    # `hash(True) == hash(1)`, so pre-fix this reported `{1: 0.0}` and the
    # second element simply disappeared. The dedup's own comment claims "no
    # effect for already-unique ks" — these two are distinct elements of the
    # input sequence.
    with pytest.raises(ValueError, match=r"got \[True\]"):
        validate_ks([1, True])


# ----------------------------------------------------------------------
# The types that reached the slice
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [2.5, 3.0, NAN, INF, -INF])
def test_non_int_numeric_k_is_rejected_at_the_guard_not_at_the_slice(bad: float) -> None:
    # Pre-fix all five passed `validate_ks` and raised
    # `TypeError: slice indices must be integers` from `scored[:max_k]` — three
    # frames below the guard whose entire job is to reject an unusable k, from
    # an expression that names neither `ks` nor the value.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        # `validate_ks` declares `Sequence[int]`; passing a float is the point.
        # The guard exists for callers that are *not* type-checked — a `--ks`
        # value off the command line, a k read out of a JSON file — so the test
        # has to reach it with input the annotation forbids (#174). Narrow
        # suppression, same idiom as `test_corpus.py`'s `arg-type`.
        validate_ks([bad])  # type: ignore[list-item]
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _run((bad,))


def test_3_point_0_is_rejected_because_it_is_the_json_shape_of_an_integer() -> None:
    # The reachable case. `json.loads("3.0")` is `3.0`, so a `ks` list read from
    # a config file or a notebook cell carries floats without anyone typing a
    # decimal point. Rejected rather than coerced so this write-path guard
    # agrees with `_validate_count`, which rejects `3.0` for `n_queries` on the
    # read path; the message names the coercion.
    with pytest.raises(ValueError, match=r"coerce with int\(k\)"):
        validate_ks(json.loads("[3.0]"))


@pytest.mark.parametrize("bad", ["3", None, [], {}])
def test_non_numeric_k_raises_ValueError_not_TypeError(bad: object) -> None:
    # Pre-fix these escaped from the `k <= 0` comparison itself as
    # `TypeError: '<=' not supported between instances of 'str' and 'int'` —
    # a different exception class from every other rejection this function
    # makes, and one that says nothing about `ks`.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        validate_ks([bad])  # type: ignore[list-item]


def test_every_offender_is_reported_in_one_pass() -> None:
    # Matching the existing non-positive behaviour, whose docstring says "every
    # offender is surfaced in one pass so operators don't chase them
    # one-at-a-time". A mixed-garbage list should name all three bad elements.
    with pytest.raises(ValueError, match=r"must be an int") as excinfo:
        validate_ks([1, "x", None, 2.5])  # type: ignore[list-item]
    msg = str(excinfo.value)
    assert "'x'" in msg
    assert "None" in msg
    assert "2.5" in msg


def test_the_type_check_runs_before_the_sign_check() -> None:
    # Ordering is not cosmetic: `k <= 0` raises TypeError on a str/None element,
    # so the sign check cannot run at all until the non-numeric ones are gone.
    # A list with both a bad type and a bad sign must report the type.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        validate_ks([0, None])  # type: ignore[list-item]


# ----------------------------------------------------------------------
# What must not change
# ----------------------------------------------------------------------


def test_the_existing_sign_and_empty_guards_are_untouched() -> None:
    with pytest.raises(ValueError, match="ks must be non-empty"):
        validate_ks([])
    with pytest.raises(ValueError, match=r"every k in ks must be positive; got \[0\]"):
        validate_ks([0])
    with pytest.raises(ValueError, match=r"every k in ks must be positive; got \[-3, -1\]"):
        validate_ks([-1, -3])


def test_ordinary_int_ks_still_pass() -> None:
    for ks in ([1], [1, 3, 5], (1, 3, 5), [10, 2, 7], range(1, 4)):
        validate_ks(list(ks))
    # And a duplicate is still deduped rather than rejected — that behaviour
    # predates this change (#84) and is deliberate.
    run = _run((3, 3, 1))
    assert sorted(run.recall_at_k) == [1, 3]


def test_run_matrix_still_preflights_through_this_same_function() -> None:
    # The CLI is unaffected by this change — it parses `--ks` with `int(k)`, so
    # `3.0`/`true`/`nan` never reach `validate_ks` from that path. Pinned so a
    # future author doesn't "fix" the CLI by loosening the library guard.
    import scripts.run_matrix as rm

    assert rm.validate_ks is validate_ks
