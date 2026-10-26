"""The one exact round-trip check for a result file `run_matrix.py` wrote (#311).

Four tests lock `RetrievalRun.from_json(payload).to_json()` against the payload
read from `results/canonical__*.json`. Each of them used to compare the two
directly. Since #303, `run_matrix.py` writes `run_at` next to the `to_json()`
fields, and `from_json` ignores it, so a freshly regenerated fixture never
equalled its own round trip. The README's fixture-refresh command
(`run_matrix.py --canonical-out`) turned all four red. They passed only because
the committed fixtures had not been regenerated since #303.

The keys to drop are listed by name. Dropping "whatever `to_json()` does not
emit" would be the plausible shortcut, and it would also hide a field that
`from_json` silently lost, which is the thing these locks exist to catch.
"""

from __future__ import annotations

import json
from typing import Any

from chunking_lab.metrics import RetrievalRun

#: Keys `run_matrix.py` writes into a result file beside `RetrievalRun.to_json()`.
#: They are file provenance, not part of the run, and `from_json` ignores them.
#: `run_at` is the run's UTC start time, which the notebook loader orders by
#: (#303). A new key `run_matrix.py` adds belongs here, and adding it should be
#: a deliberate change.
FILE_PROVENANCE_KEYS = frozenset({"run_at"})


def assert_round_trips(payload: dict[str, Any], name: str) -> RetrievalRun:
    """Load `payload` and assert `to_json()` reproduces it exactly, provenance aside.

    Compared as `sort_keys` JSON text, the form three of the four locks already
    used. Returns the loaded run for the caller's own extra checks.
    """
    run = RetrievalRun.from_json(payload)
    expected = {k: v for k, v in payload.items() if k not in FILE_PROVENANCE_KEYS}
    assert json.dumps(run.to_json(), sort_keys=True) == json.dumps(expected, sort_keys=True), (
        f"{name}: RetrievalRun.from_json(...).to_json() does not reproduce the file"
    )
    return run
