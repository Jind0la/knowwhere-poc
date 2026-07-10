"""knowwhere.injection — Pure helpers for subconscious injection.

No DB or network I/O. Used by the Hermes plugin and unit tests.
"""

# Try the in-package copy first (for pip installs),
# fall back to repo-root kw_injection.py (for dev mode)
try:
    from knowwhere._kw_injection import (  # noqa: E402
        HEADER,
        FOOTER,
        MAX_INJECTION_CHARS,
        MAX_SUMMARY_BODY,
        MIN_QUERY_CHARS,
        build_search_query,
        contains_distractor,
        extract_outcome_facts,
        filter_guardrails,
        format_injection,
        format_injection_block,
        merge_relevant_and_debuts,
    )
except ImportError:
    import sys
    from pathlib import Path
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from kw_injection import (  # noqa: E402
        HEADER,
        FOOTER,
        MAX_INJECTION_CHARS,
        MAX_SUMMARY_BODY,
        MIN_QUERY_CHARS,
        build_search_query,
        contains_distractor,
        extract_outcome_facts,
        filter_guardrails,
        format_injection,
        format_injection_block,
        merge_relevant_and_debuts,
    )

__all__ = [
    "build_search_query",
    "contains_distractor",
    "extract_outcome_facts",
    "filter_guardrails",
    "format_injection",
    "format_injection_block",
    "merge_relevant_and_debuts",
    "HEADER",
    "FOOTER",
    "MAX_INJECTION_CHARS",
    "MAX_SUMMARY_BODY",
    "MIN_QUERY_CHARS",
]
