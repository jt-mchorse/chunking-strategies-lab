"""Embedder Protocol + dep-free reference + sentence-transformers adapter.

The canonical embedding model for this lab's measurements is
``sentence-transformers/all-MiniLM-L6-v2`` (384-d, Apache 2.0,
CPU-friendly). The class lives behind an opt-in import (D-003) — the
package itself takes no runtime deps, and the optional ``sbert`` extra
brings in ``sentence-transformers`` + ``numpy`` only when needed.

``HashEmbedder`` is the dependency-free reference; tests use it and CI
runs against it. Real numbers in issue #3's metrics matrix use
``MiniLMEmbedder``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol, runtime_checkable

CANONICAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""The model id pinned by D-002 for all strategy comparisons."""

DEFAULT_DIM = 384
"""Dimensionality of the canonical model. HashEmbedder defaults to this too."""


@runtime_checkable
class Embedder(Protocol):
    """Anything with ``.embed(text) -> list[float]`` is an embedder."""

    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


class HashEmbedder:
    """Deterministic SHA-256-derived embedding. Not for production use.

    Properties retained for retrieval semantics: same text → same vector,
    unit-length, distinct texts → distinct vectors. CI tests use this
    so they're hermetic; real numbers come from MiniLMEmbedder.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 8 != 0:
            raise ValueError(f"dim must be a multiple of 8, got {dim}")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        needed_bytes = self.dim * 4
        buf = bytearray()
        counter = 0
        seed = text.encode("utf-8")
        while len(buf) < needed_bytes:
            h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            buf.extend(h)
            counter += 1
        floats: list[float] = []
        for i in range(self.dim):
            (u32,) = struct.unpack(">I", bytes(buf[i * 4 : i * 4 + 4]))
            floats.append((u32 / 0xFFFFFFFF) * 2.0 - 1.0)
        norm = math.sqrt(sum(x * x for x in floats))
        if norm == 0:
            return floats
        return [x / norm for x in floats]


class MiniLMEmbedder:
    """sentence-transformers adapter for the canonical embedding model.

    Requires the optional ``sbert`` extra: ``pip install '.[sbert]'``.
    Lazy-imports so the package still imports cleanly without the extra.
    """

    def __init__(self, model_name: str = CANONICAL_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - exercised by users without the extra
            raise ImportError(
                "MiniLMEmbedder requires the 'sbert' extra. "
                "Install with: pip install 'chunking-strategies-lab[sbert]'"
            ) from e
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:  # pragma: no cover - integration
        vec = self._model.encode(text, normalize_embeddings=True)
        return list(map(float, vec))
