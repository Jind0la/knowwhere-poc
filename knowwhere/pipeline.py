"""knowwhere.pipeline — Summarization and embedding pipeline.

Instant (rule-based) and full (LLM async) summarization, plus
embeddings via Ollama, OpenAI, or local sentence-transformers.
"""

# Try the in-package copy first (for pip installs),
# fall back to repo-root summary_pipeline.py (for dev mode)
try:
    from knowwhere._summary_pipeline import (  # noqa: E402
        embed_text,
        make_instant_summary,
        call_deepseek_full_summary,
        detect_project,
        format_turns_for_summary,
        OLLAMA_MODEL,
        OLLAMA_URL,
        TRUNC_DIM,
    )
except ImportError:
    import sys
    from pathlib import Path
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from summary_pipeline import (  # noqa: E402
        embed_text,
        make_instant_summary,
        call_deepseek_full_summary,
        detect_project,
        format_turns_for_summary,
        OLLAMA_MODEL,
        OLLAMA_URL,
        TRUNC_DIM,
    )

__all__ = [
    "embed_text",
    "make_instant_summary",
    "call_deepseek_full_summary",
    "detect_project",
    "format_turns_for_summary",
    "OLLAMA_MODEL",
    "OLLAMA_URL",
    "TRUNC_DIM",
]
