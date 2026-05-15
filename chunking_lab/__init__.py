"""chunking-strategies-lab: pinned substrate for chunking-strategy comparisons.

Issue #1 surface:

    from chunking_lab import (
        CANONICAL_EMBEDDING_MODEL,
        Document,
        Query,
        Embedder,
        HashEmbedder,
        load_corpus,
        load_queries,
    )

The strategy implementations themselves (fixed-size, recursive, semantic,
late-chunking, document-structure-aware) ship in issue #2; the retrieval
metrics matrix ships in #3. This module locks the substrate they share.
"""

from .corpus import Document, load_corpus
from .embedder import CANONICAL_EMBEDDING_MODEL, Embedder, HashEmbedder
from .queries import Query, load_queries

__all__ = [
    "CANONICAL_EMBEDDING_MODEL",
    "Document",
    "Embedder",
    "HashEmbedder",
    "Query",
    "load_corpus",
    "load_queries",
]
