#!/usr/bin/env bash
# Deterministic driver for the 60-second README demo (issue #17).
#
# Runs the two demo surfaces in sequence on a fresh clone with no API
# key:
#
#   1. matrix run    — `python scripts/run_matrix.py --results-dir <tmp>`
#                      walks all five strategies (fixed-size, recursive,
#                      semantic, late-chunking, structure-aware) against
#                      the pinned corpus + queries with the dep-free
#                      HashEmbedder and prints one line per strategy
#                      (recall@5 / snippet-hit@5 / wall_clock) live.
#
#   2. summary cat   — the same run wrote a sibling
#                      `<ts>__summary.md`; we cat it so the viewer sees
#                      the same five-row markdown table that
#                      `results/summary.md` ships (locked by
#                      `tests/test_summary_snapshot.py`).
#
# The output is the recording — when JT records the GIF/video, this
# script's stdout is what gets captured. Hermetic: no API key, no
# network, no committed artifacts touched (everything writes under a
# per-run tempdir).
#
# Why HashEmbedder (not MiniLM): the headline cross-strategy claims live
# in the canonical `results/summary.md` and the notebook's takeaways
# section, both grounded in operator-run MiniLM numbers. The capture's
# job is to show the *runner* — that the five strategies all wire to
# the same Protocol and produce the same output shape — not to
# re-publish quality claims. The banner says so explicitly so a viewer
# doesn't read HashEmbedder recall as a strategy comparison.
#
# Variables:
#   CAPTURE_PACE_SECONDS  pause between sections (default 2 for
#                         recording; tests/test_capture_demo_smoke.py
#                         sets this to 0).
#
# Exit: 0 on full success; non-zero on any sub-step failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACE="${CAPTURE_PACE_SECONDS:-2}"

banner() {
  printf '\n'
  printf '═══ %s\n' "$1"
  printf '\n'
}

pace() {
  if [ "$PACE" != "0" ]; then
    sleep "$PACE"
  fi
}

cd "$REPO_ROOT"

# Per-run scratch so concurrent recordings (and the smoke test) don't
# collide. Cleaned up on exit including error paths.
TMPDIR_DEMO="$(mktemp -d -t chunking-lab-capture-XXXXXX)"
cleanup() {
  rm -rf "$TMPDIR_DEMO"
}
trap cleanup EXIT INT TERM

# Resolve the Python interpreter from the active venv if one is present.
# Falls back to plain `python` (or python3) so the script works under
# whatever interpreter the operator's PATH provides.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

banner "chunking-strategies-lab · 60-second demo"
printf 'two surfaces · HashEmbedder (dep-free) · pinned corpus + queries\n'
printf 'cross-strategy quality claims live in canonical results/summary.md (MiniLM).\n'
pace

banner "1/2 · matrix run · five strategies, same Protocol, same corpus + queries"
printf 'python scripts/run_matrix.py --results-dir <tmp>\n'
printf '  five strategies: fixed-size · recursive · semantic · late-chunking · structure-aware\n'
printf '  one line per strategy: recall@5 · snippet-hit@5 (D-008) · wall_clock ms\n\n'
"$PYTHON_BIN" scripts/run_matrix.py --results-dir "$TMPDIR_DEMO"
pace

banner "2/2 · rendered summary · same five-row table format that results/summary.md ships"
SUMMARY_PATH="$(ls -1 "$TMPDIR_DEMO"/*__summary.md | tail -n1)"
printf 'cat %s\n' "$(basename "$SUMMARY_PATH")"
printf '  (test_summary_snapshot.py locks this exact format against the canonical fixtures)\n\n'
printf '─── rendered summary ────────────────────────────────────────────────\n\n'
cat "$SUMMARY_PATH"
pace

banner "done · five strategies, one Protocol, one corpus, one summary contract"
printf 'next stop for real-quality numbers:\n'
printf '  pip install -e .[sbert]\n'
printf '  python scripts/run_matrix.py --embedder minilm --canonical-out   # refreshes results/\n'
printf '  jupyter nbconvert --to notebook --inplace --execute notebooks/comparison.ipynb\n'
