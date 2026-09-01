"""`int(k)` is not the inverse of `str(k)`, and nothing checked the key (#169).

`RetrievalRun.to_json` writes metric-map keys as `str(k)` for `k: int`.
`from_json` read them back with a bare `{int(k): v for k, v in ...}`, and
`_validate_metric_map` walks `mapping.items()` inspecting only `v` -- `k`
appears solely inside the error message. So the key axis was `int()` and
nothing else, while three docstrings in the module said otherwise:

    from_json:            "the failure mode is loud, not silent, on both the
                           key and the value axis"
    _validate_metric_map: "matching the loud-key contract of `from_json`"
    validate_ks (#158):   "the *read* path in this same module enforces it
                           fully for the very fields these `k` become"

Measured on `main`, over an otherwise-canonical payload::

    key spelling                   from_json     loaded recall_at_k
    CONTROL {"1":.., "5":..}       loaded        {1: 0.5, 5: 0.9}
    {"5": 0.9, "05": 0.1}          loaded        {5: 0.1}     <- 0.9 GONE
    {"5": 0.9, " 5": 0.1}          loaded        {5: 0.1}     <- 0.9 GONE
    {"5": 0.9, "+5": 0.1}          loaded        {5: 0.1}     <- 0.9 GONE
    {"5": 0.9, <Arabic-Indic 5>}   loaded        {5: 0.1}     <- 0.9 GONE
    {"5_0": 0.9}                   loaded        {50: 0.9}    <- a typo becomes another k
    {"-3": 0.9}                    loaded        {-3: 0.9}
    {"0": 0.9}                     loaded        {0: 0.9}

Two harms. The collisions are **silent** -- a real measurement is overwritten
and nothing says so. And they defeat `#160`'s cross-map check, which compares
key sets *after* the coercion, so the coercion hides exactly the mismatch that
guard exists to find.

`k <= 0` is the sibling half: `evaluate_strategy` calls `validate_ks(ks)` before
doing anything, and the comment there names the harm ("`k=0` silently produces
`recall@0=0.0` always; `k<0` silently miscounts"). The reader accepted both from
a file, and `_render_summary` derives its columns from the loaded keys, so a
`recall@-3` column could reach the tracked `results/summary.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chunking_lab.metrics import RetrievalRun

# Built from a codepoint. U+0665 ARABIC-INDIC DIGIT FIVE: `int()` accepts it as
# 5, and `str(5)` can never produce it.
ARABIC_FIVE = chr(0x0665)

_BASE: dict[str, Any] = {
    "strategy_name": "fixed-size",
    "embedder_model": "hash",
    "dataset_version": "v1",
    "n_queries": 2,
    "n_chunks_total": 10,
    "wall_clock_ms": 1.0,
    "per_query": [],
    "notes": [],
}


def _payload(recall: dict[str, Any], snippet: dict[str, Any] | None = None) -> dict[str, Any]:
    p = dict(_BASE)
    p["recall_at_k"] = recall
    p["snippet_hit_at_k"] = recall if snippet is None else snippet
    # Round-trip through JSON so the test exercises the shape a real file has.
    return dict(json.loads(json.dumps(p)))


def _load(recall: dict[str, Any], snippet: dict[str, Any] | None = None) -> RetrievalRun:
    return RetrievalRun.from_json(_payload(recall, snippet))


# ----------------------------------------------------------------------
# Non-canonical spellings: rejected, loudly, by field name
# ----------------------------------------------------------------------

# (label, key) -- every one of these `int()`s cleanly and none is producible by
# `str(k)`.
NON_CANONICAL = [
    ("leading zero", "05"),
    ("many leading zeros", "0005"),
    ("leading space", " 5"),
    ("trailing space", "5 "),
    ("surrounding whitespace", "\t5\n"),
    ("leading plus", "+5"),
    ("underscore digit separator", "5_0"),
    ("Arabic-Indic decimal digit", ARABIC_FIVE),
    ("negative zero", "-0"),
]


@pytest.mark.parametrize(("label", "key"), NON_CANONICAL, ids=[r[0] for r in NON_CANONICAL])
def test_a_non_canonical_key_is_rejected_by_field_name(label: str, key: str) -> None:
    assert int(key) == int(key)  # precondition: `int()` accepts it
    with pytest.raises(ValueError, match=r"recall_at_k key .* is not the canonical spelling"):
        _load({key: 0.9})


@pytest.mark.parametrize(("label", "key"), NON_CANONICAL, ids=[r[0] for r in NON_CANONICAL])
def test_the_rule_applies_to_the_snippet_map_too(label: str, key: str) -> None:
    """Both operands. The two maps are read by the same helper, so a rule that
    reached only `recall_at_k` would be the same class of half-guard."""
    with pytest.raises(ValueError, match=r"snippet_hit_at_k key .* canonical"):
        _load({"5": 0.9}, {key: 0.9})


@pytest.mark.parametrize(
    ("label", "key"),
    [("non-numeric", "best"), ("float-looking", "5.0"), ("empty string", ""), ("hex", "0x5")],
)
def test_a_non_integer_key_names_the_field_rather_than_int_builtin(label: str, key: str) -> None:
    """These already raised `ValueError`, but the message was
    `invalid literal for int() with base 10: 'best'` -- it named neither the
    field nor what the writer actually produces."""
    with pytest.raises(ValueError, match=r"recall_at_k key .* is not an integer") as exc:
        _load({key: 0.9})
    assert "to_json" in str(exc.value)


# ----------------------------------------------------------------------
# The collisions, and what they hid
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "colliding_key"),
    [
        ("leading zero", "05"),
        ("leading space", " 5"),
        ("leading plus", "+5"),
        ("Arabic-Indic five", ARABIC_FIVE),
    ],
)
def test_a_measurement_can_no_longer_be_silently_overwritten(
    label: str, colliding_key: str
) -> None:
    """Before #169 this loaded and `recall_at_k` was `{5: 0.1}` -- the 0.9 was
    gone with no diagnostic anywhere."""
    assert int(colliding_key) == 5
    with pytest.raises(ValueError, match="canonical"):
        _load({"5": 0.9, colliding_key: 0.1})


def test_the_collision_no_longer_hides_160s_cross_map_mismatch() -> None:
    """`#160` compares `recall.keys() != snippet.keys()` *after* the coercion.
    `{"5":.., "05":..}` collapsed to `{5}` and matched a snippet map of `{"5"}`,
    so the guard written to catch exactly this mismatch passed and the run
    loaded clean with one measurement dropped."""
    with pytest.raises(ValueError, match="canonical"):
        _load({"5": 0.9, "05": 0.1}, {"5": 0.9})


def test_160s_own_check_still_fires_on_a_genuinely_mismatched_pair() -> None:
    """Control: the guard this one was hiding must still work."""
    with pytest.raises(ValueError, match="same k"):
        _load({"1": 0.5, "5": 0.9}, {"1": 0.5})


# ----------------------------------------------------------------------
# The range half: one rule, shared with the write path
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("label", "key"), [("zero", "0"), ("negative", "-3"), ("negative", "-1")])
def test_a_k_the_write_path_refuses_to_produce_cannot_be_read_back(label: str, key: str) -> None:
    """`evaluate_strategy` calls `validate_ks(ks)` first, and the comment there
    names the harm. The reader accepted both from a file."""
    with pytest.raises(ValueError, match=r"recall_at_k keys: .*positive"):
        _load({key: 0.9})


def test_the_range_rule_is_shared_not_restated() -> None:
    """The read path must call `validate_ks`, the same function
    `evaluate_strategy` and the `--ks` CLI pre-flight call (#149, #158) -- not a
    second copy of the rule. Asserted by behaviour: patching `validate_ks` to
    accept everything must let a bad k through the reader too."""
    import chunking_lab.metrics as metrics

    original = metrics.validate_ks
    try:
        # No `# type: ignore[assignment]` on either line: mypy does not treat a
        # module attribute as a declared-type target here, so the suppressions
        # that used to sit on them were dead the moment `tests/` entered the
        # gate under `warn_unused_ignores` (#174).
        metrics.validate_ks = lambda ks: None
        run = _load({"0": 0.9})
    finally:
        metrics.validate_ks = original
    assert dict(run.recall_at_k) == {0: 0.9}, (
        "the reader did not go through validate_ks; the rule is restated somewhere"
    )
    # And with the real rule back in place it is refused again.
    with pytest.raises(ValueError, match="positive"):
        _load({"0": 0.9})


# ----------------------------------------------------------------------
# Controls -- the guard is a rejection, not a transformation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "recall"),
    [
        ("the default ks", {"1": 0.5, "3": 0.7, "5": 0.9}),
        ("a single k", {"1": 1.0}),
        ("a large k", {"1000": 0.0}),
        ("unordered keys", {"5": 0.9, "1": 0.5}),
        ("boundary values", {"1": 0.0, "3": 1.0}),
    ],
)
def test_canonical_payloads_load_unchanged(label: str, recall: dict[str, Any]) -> None:
    run = _load(recall)
    assert dict(run.recall_at_k) == {int(k): v for k, v in recall.items()}
    assert dict(run.snippet_hit_at_k) == {int(k): v for k, v in recall.items()}


def test_the_round_trip_is_still_exact() -> None:
    """to_json -> from_json -> to_json, byte-identical."""
    run = _load({"1": 0.5, "3": 0.7, "5": 0.9})
    once = json.dumps(run.to_json(), sort_keys=True)
    twice = json.dumps(RetrievalRun.from_json(json.loads(once)).to_json(), sort_keys=True)
    assert once == twice


def test_every_committed_result_file_still_loads() -> None:
    """The guard must not reject the artifacts this repo actually ships."""
    files = sorted(Path("results").glob("*.json"))
    assert len(files) >= 5, files
    for path in files:
        run = RetrievalRun.from_json(json.loads(path.read_text(encoding="utf-8")))
        assert sorted(run.recall_at_k) == sorted(run.snippet_hit_at_k)
        assert all(k >= 1 for k in run.recall_at_k), path


def test_the_table_is_not_vacuous() -> None:
    """Every non-canonical row must be a key `int()` genuinely accepts --
    otherwise the parametrization would be testing the pre-existing
    `invalid literal` path and proving nothing new."""
    assert len(NON_CANONICAL) >= 8
    for label, key in NON_CANONICAL:
        int(key)  # raises if the row drifted into the already-rejected class
        assert str(int(key)) != key, f"{label}: {key!r} IS canonical; the row proves nothing"
