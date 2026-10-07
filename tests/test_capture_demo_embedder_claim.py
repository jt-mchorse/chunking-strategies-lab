"""The demo names the embedder the committed summary was made with (#228).

The capture banner said "cross-strategy quality claims live in canonical
results/summary.md (MiniLM)", its header said that file was "grounded in
operator-run MiniLM numbers", and the README said it was "produced by the
operator with --embedder minilm". The committed summary's first line is
``_embedder_: `HashEmbedder` ``, followed by a note that a HashEmbedder run is
not a quality comparison; no MiniLM numbers are committed anywhere. The banner
now reads the embedder from that line, and these arms derive it too, so a
MiniLM regeneration keeps them green without an edit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "capture_demo.sh"


def _summary_embedder() -> str:
    text = (ROOT / "results" / "summary.md").read_text(encoding="utf-8")
    m = re.search(r"^_embedder_: `([^`]*)`", text, re.MULTILINE)
    assert m is not None, "results/summary.md has no _embedder_ header line"
    return m.group(1)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_the_banner_names_the_committed_summarys_embedder() -> None:
    r = subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={**os.environ, "CAPTURE_PACE_SECONDS": "0"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-800:]
    head = r.stdout.split("1/2", 1)[0]
    embedder = _summary_embedder()
    assert f"committed results/summary.md: {embedder}" in head
    if embedder == "HashEmbedder":
        assert "MiniLM)" not in head
        assert "pending" in head


def test_no_prose_attributes_minilm_numbers_to_a_summary_that_is_not_minilm() -> None:
    if _summary_embedder() != "HashEmbedder":
        pytest.skip("the committed summary is a real-embedder run; the attribution can be true")
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    header = " ".join(
        line.lstrip("# ")
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )
    for text, where in ((readme, "README.md"), (header, "scripts/capture_demo.sh header")):
        assert "produced by the operator with `--embedder minilm`" not in text, where
        assert "grounded in operator-run MiniLM numbers" not in text, where
