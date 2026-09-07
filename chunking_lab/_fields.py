"""Shared field validators for the package's dataclass construction boundaries.

One definition, here rather than in any of the four modules that need it, for
the reason #180 exists: the rule below was already written down twice — as
`isinstance(x, int) and not isinstance(x, bool)` on nine strategy-constructor
fields (#29/#31), and as `metrics._validate_count` on `RetrievalRun.from_json`'s
read path — and the boundaries that had *neither* were the ones nobody
enumerated.

Imports nothing from `chunking_lab`, so every module can use it without a
cycle. Stdlib-only, per D-002.

Why `bool` is excluded explicitly, every time: it subclasses `int`, so a plain
`isinstance(v, int)` accepts `True` — and `True` is the value that fails
*silently* rather than raising downstream. `Chunk(text="Hello",
start_offset=True, end_offset=5)` recorded a span that slices `'ello'`, and
`RetrievalRun(n_queries=True, ...)` wrote `"n_queries": true` to JSON that its
own `from_json` then refused to read back.
"""

from __future__ import annotations

import math
from typing import Any


def require_non_negative_int(name: str, value: Any) -> None:
    """Raise `ValueError` unless `value` is a non-bool `int` that is `>= 0`.

    `ValueError`, not `TypeError`, to match `check_chunk_input` and
    `queries._require_str` — a caller wrapping construction in `except
    ValueError` should catch every field-contract failure by one road.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value!r}")


def require_str(name: str, value: Any) -> None:
    """Raise `ValueError` unless `value` is a `str`.

    Emptiness is deliberately *not* checked here. `queries._require_str` rejects
    a blank string because a blank query id is meaningless, but an empty
    `Chunk.text` is legitimate (a zero-width span), so the two contracts differ
    on content and agree only on type.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str; got {type(value).__name__} ({value!r})")


def require_non_negative_finite_number(name: str, value: Any) -> None:
    """Raise `ValueError` unless `value` is a non-bool, finite number `>= 0`.

    The float sibling of :func:`require_non_negative_int`, and a deliberate
    widening of #180's population. That issue's discovery walked fields
    annotated `int`, which is why it found seven fields and missed
    `RetrievalRun.wall_clock_ms` — a `float` with the *same* read/write
    asymmetry, and the one whose bool case `test_corrupt_wall_clock_would_
    render_into_the_summary` had already measured: `True` renders as a
    fabricated `1` ms measurement in `results/summary.md`.

    An `int` is accepted: a whole-millisecond latency is a legitimate JSON
    number, and `from_json` accepts one on the read side.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number; got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value!r}")
