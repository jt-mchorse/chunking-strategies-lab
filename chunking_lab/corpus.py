"""Corpus loader for the pinned substrate.

The corpus lives at ``data/corpus/*.md``, one document per file. Each
document is multi-paragraph technical prose authored for this repo
(MIT-licensed, no external license to attribute). Strategies are
compared on this fixed substrate per **D-002**.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "corpus"


@dataclass(frozen=True)
class Document:
    """One source document. ``filename`` is the chunk-strategy-agnostic id.

    ``filename`` is the basename (not the full path) so it round-trips
    cleanly through the queries file's ``expected_doc`` field.
    """

    filename: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def load_corpus(corpus_dir: PathLike[str] | str | None = None) -> list[Document]:
    """Load every ``*.md`` document from the corpus directory.

    Results are sorted by filename so iteration is deterministic. Files
    are read as UTF-8, tolerating a leading BOM (see below).
    """
    base = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    if not base.exists():
        raise FileNotFoundError(f"corpus directory not found: {base}")
    docs: list[Document] = []
    for path in sorted(base.glob("*.md")):
        # Files only. `glob("*.md")` also matches a *directory* whose name ends
        # in `.md` (e.g. an exported `bundle.md/` folder); without this guard the
        # `read_text()` below crashes with IsADirectoryError. validate.py's corpus
        # enumeration already filters `if p.is_file()`, and its #98 comment asserts
        # it "Mirror[s] load_corpus's enumeration EXACTLY … files only" — but the
        # loader was never brought into that parity, so a corpus with a stray
        # `*.md` directory passed validation clean yet crashed on load (#108). A
        # non-file `*.md` entry is never a loadable Document, so skip it.
        if not path.is_file():
            continue
        # utf-8-sig transparently strips a leading BOM (EF BB BF — the default
        # for Windows Notepad and some doc exporters) and is a no-op for
        # BOM-less UTF-8. Without it the U+FEFF survives into the document text
        # as an invisible leading char, shifting first-chunk offsets by one and
        # leaking into snippet-match comparisons (#95). Parity with the queries
        # loader / validator fixed in #93.
        docs.append(Document(filename=path.name, text=path.read_text(encoding="utf-8-sig")))
    if not docs:
        raise FileNotFoundError(f"no markdown documents in: {base}")
    return docs
