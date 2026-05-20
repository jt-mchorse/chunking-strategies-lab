"""chunking-strategies-lab: pinned substrate + 5 chunking strategies.

Issue #1 substrate:

    from chunking_lab import (
        CANONICAL_EMBEDDING_MODEL,
        Document, Query,
        Embedder, HashEmbedder,
        load_corpus, load_queries,
    )

Issue #2 strategies (this PR):

    from chunking_lab import (
        Chunk, Strategy, LateChunk,
        FixedSizeStrategy, RecursiveStrategy, SemanticBoundaryStrategy,
        LateChunkingStrategy, StructureAwareStrategy,
    )

The retrieval-metrics matrix ships in #3. This module locks the
substrate + the strategy contract they share.
"""

__version__ = "0.0.1"  # mirror of pyproject.toml [project] version

from .corpus import Document, load_corpus
from .embedder import CANONICAL_EMBEDDING_MODEL, Embedder, HashEmbedder
from .queries import Query, load_queries
from .strategies import (
    Chunk,
    FixedSizeStrategy,
    LateChunk,
    LateChunkingStrategy,
    RecursiveStrategy,
    SemanticBoundaryStrategy,
    Strategy,
    StructureAwareStrategy,
)

__all__ = [
    # Substrate (#1)
    "CANONICAL_EMBEDDING_MODEL",
    "Document",
    "Embedder",
    "HashEmbedder",
    "Query",
    "load_corpus",
    "load_queries",
    # Strategies (#2)
    "Chunk",
    "FixedSizeStrategy",
    "LateChunk",
    "LateChunkingStrategy",
    "RecursiveStrategy",
    "SemanticBoundaryStrategy",
    "Strategy",
    "StructureAwareStrategy",
]
