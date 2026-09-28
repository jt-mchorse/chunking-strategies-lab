"""Atomic on-disk write helper, and the copy a frozen record needs for a free-form field.

`scripts/run_matrix.py` writes per-strategy `RetrievalRun` JSON
(canonical fixtures committed to `results/`) plus a markdown summary
that `tests/test_summary_snapshot.py` locks. `Path.write_text` is not
atomic: SIGINT/SIGTERM/disk-full/OOM between the implicit
`open(..., "w")` truncate and `close()` flush leaves the destination
zero-length or partial. The downside on the canonical-fixture write
path is particularly nasty: a half-written `canonical__*.json` either
fails the snapshot test loudly (good — but obscures the underlying
crash) or, worse, gets committed and silently changes the published
numbers.

Pattern mirrors the portfolio siblings:
- `rag_kit/io_utils.atomic_write_text` (rag-production-kit#44/#45)
- `eval_harness/io_utils.atomic_write_text` (llm-eval-harness#51, D-015)
- `emb_shootout/io_utils.atomic_write_text` (embedding-model-shootout#37, D-009)
- `prompt_regression/io.atomic_write_text` (prompt-regression-suite#40)
- `async_pipelines/io_utils.atomic_write_text` (python-async-llm-pipelines#36, D-011)
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

# Cap the target basename's contribution to the temp filename. The temp name is
# `.<base>.<random>.tmp`; the affixes add ~13-20 bytes, so prepending a full
# basename that is itself near NAME_MAX (255 on ext4/APFS) overflows the limit
# and the write fails with `OSError: [Errno 63] File name too long` — even though
# a plain `Path.write_text` of that same target succeeds (sibling of
# rag-production-kit#128 and mcp-server-cookbook#96). The base in the temp name
# is cosmetic (`ls`-ability); uniqueness comes from `NamedTemporaryFile`'s random
# component, so truncating it is safe. Budget is in BYTES (NAME_MAX is a byte
# limit) and we trim on a char boundary so multibyte names are never split
# mid-codepoint.
_MAX_TEMP_BASE_BYTES = 200


def _name_bytes(base: str) -> int:
    """Length of *base* in the bytes the filesystem actually sees.

    `os.fsencode`, not `base.encode("utf-8")` (#178). Both halves of the
    comment above are true and the old implementation still counted the wrong
    bytes: NAME_MAX limits the bytes handed to the kernel, which is
    `os.fsencode` — `sys.getfilesystemencoding()` together with
    `sys.getfilesystemencodeerrors()`, i.e. `surrogateescape` on POSIX.

    That handler is why the distinction bites rather than being pedantry. A
    path byte that is not valid UTF-8 arrives in Python as a lone surrogate in
    `U+DC80..U+DCFF`, and strict `str.encode("utf-8")` refuses to encode it —
    so `_cap_base_for_temp` used to raise `UnicodeEncodeError` on a destination
    the OS can name, *before* reaching the length question. `sys.argv` decodes
    with the same handler, so a shell `--out $'report\\xff.txt'` is enough.

    `UnicodeEncodeError` is a `ValueError`, so `validate`'s `except OSError`
    write guard missed it — and the cost of that miss is higher here than in
    the sibling repos. An uncaught exception exits the interpreter with code 1,
    and 1 is not "an error" in this CLI: it is *the corpus has findings*. The
    guard's own comment says it exists to stop a write failure "colliding with
    the 'findings' code and breaking the documented '0 clean / 1 findings /
    2 I/O error' contract" — which is precisely what a byte in the `--out`
    name did, handing a gating CI job a wrong content answer rather than a
    visible I/O failure.

    `os.fsencode` never raises: `surrogateescape` on POSIX, `surrogatepass` on
    Windows, so every `str` a `Path` can hold round-trips. For a name that is
    valid UTF-8 it returns exactly the old number, so the budget is unchanged
    for every name that worked before.
    """
    return len(os.fsencode(base))


def _cap_base_for_temp(base: str) -> str:
    if _name_bytes(base) <= _MAX_TEMP_BASE_BYTES:
        return base
    out = base
    while out and _name_bytes(out) > _MAX_TEMP_BASE_BYTES:
        out = out[:-1]
    return out


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically.

    On success the destination contains exactly *text*. On any failure
    path (signal, disk-full, OOM during flush), the destination is
    either unchanged (overwrite case) or absent (new-file case) —
    never partial. Parent directories are created with `mkdir(parents=True,
    exist_ok=True)`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=target.parent,
            prefix=f".{_cap_base_for_temp(target.name)}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()


def copy_json_value(value: Any) -> Any:
    """Copy a JSON value deeply over *containers*, leaving everything else alone.

    The copy a frozen record needs for a field whose **element type is not
    proved immutable** (#200).

    ``frozen=True`` prevents *rebinding* an attribute and says nothing about the
    object the attribute points at, so a ``dict`` field on a frozen record stays
    editable in place through any reference the caller still holds -- and
    nothing raises, because nothing is ever rebound.

    **Only `Chunk.metadata` needs this one**, and the reason the other three
    container fields on frozen records in this package do not is the whole
    decision (D-018). A shallow copy is complete exactly when the element type
    is proved immutable, and ``RetrievalRun`` *proves* it:
    ``_validate_metric_maps`` enforces non-bool finite numbers for
    ``recall_at_k`` / ``snippet_hit_at_k``, and ``_validate_notes`` enforces
    ``str`` for ``notes``. ``Chunk.metadata`` is ``dict[str, Any]`` with no
    validator at all, so ``Any`` proves nothing and ``dict(...)`` would leave
    every nested container the caller's.

    **Iterative, with an ``id()``-keyed memo, and that is not a style choice.**
    A recursive copy never terminates on a cyclic ``metadata`` and exhausts the
    stack on a deeply nested one -- and ``RecursionError`` is not a
    ``ValueError``. ``Chunk.__post_init__``'s own comment names that contract:
    the boundary raises "a field-named ``ValueError``" precisely so "a caller's
    ``except ValueError`` (the class ``check_chunk_input`` raises)" catches it.
    A recursive copy in the same method would hand that caller an exception
    class the comment argues against. ``rag-production-kit#227`` measured the
    same port going eight red against that repo's totality suite.

    A cycle is **preserved** rather than replaced with a marker: this is a
    copier, not a sanitizer, so the result is isomorphic to the input and any
    downstream serializer still sees exactly what it would have seen. The memo
    also preserves the input's *sharing* structure -- two keys pointing at one
    dict still point at one dict afterwards, a fresh one -- which keeps a
    DAG-shaped value linear rather than exponential.

    ``dict`` and ``list`` only. A mutable container nested inside a ``tuple``
    stays shared; rebuilding the tuple would lose a ``namedtuple``'s class,
    which is the measured reason ``llm-eval-harness``' D-027 drew the same line.
    Pinned by name in ``tests/test_frozen_record_container_aliasing.py`` so it
    reads as a decision rather than an oversight.
    """
    if not isinstance(value, (dict, list)):
        return value
    root: Any = {} if isinstance(value, dict) else []
    memo: dict[int, Any] = {id(value): root}
    # Every source container stays referenced while the walk runs, so CPython
    # cannot recycle an `id` out from under `memo`.
    keep: list[Any] = [value]
    stack: list[tuple[Any, Any]] = [(value, root)]
    while stack:
        src, dst = stack.pop()
        items = src.items() if isinstance(src, dict) else enumerate(src)
        for key, child in items:
            if isinstance(child, (dict, list)):
                copied = memo.get(id(child))
                if copied is None:
                    copied = {} if isinstance(child, dict) else []
                    memo[id(child)] = copied
                    keep.append(child)
                    stack.append((child, copied))
            else:
                copied = child
            if isinstance(dst, dict):
                dst[key] = copied
            else:
                dst.append(copied)
    return root
