"""KnowWhere — Subconscious Memory for AI Agents.

Automatic cross-session context injection for LLM agents.
5-process pipeline: Ingest → Summarize → Embed → Inject → Dream.

Quickstart:
    pip install knowwhere
    knowwhere init
"""

__version__ = "0.7.1"

from knowwhere.db import KnowWhereDB
from knowwhere.injection import (
    build_search_query,
    filter_guardrails,
    format_injection,
    format_injection_block,
    merge_relevant_and_debuts,
)
from knowwhere.pipeline import embed_text, make_instant_summary
from knowwhere.providers import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
)

__all__ = [
    "KnowWhereDB",
    "build_search_query",
    "filter_guardrails",
    "format_injection",
    "format_injection_block",
    "merge_relevant_and_debuts",
    "embed_text",
    "make_instant_summary",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "LocalEmbeddingProvider",
    "get_embedding_provider",
]
