"""knowwhere.db — PostgreSQL/pgvector storage backend.

Implements the StorageBackend protocol: CRUD for summaries/sources,
UCB-weighted similarity search via pgvector HNSW index, deep recall,
and health checks.

Usage:
    from knowwhere import KnowWhereDB
    db = KnowWhereDB("postgresql://...")
    results = db.search_relevant(embedding, top_k=3)
"""

import sys
from pathlib import Path

# Try the in-package module first (for pip installs),
# fall back to repo-root knowwhere_db.py (for dev mode)
try:
    from knowwhere._knowwhere_db import (  # noqa: E402
        DEFAULT_DB_URL,
        DEFAULT_DIM,
        KnowWhereDB,
        MIN_SCORE,
    )
except ImportError:
    # Dev mode: repo root on path
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from knowwhere_db import (  # noqa: E402
        DEFAULT_DB_URL,
        DEFAULT_DIM,
        KnowWhereDB,
        MIN_SCORE,
    )

from knowwhere._knowwhere_db import KnowWhereDB as DB  # alias (always from in-package copy)

__all__ = [
    "KnowWhereDB",
    "DB",
    "DEFAULT_DB_URL",
    "DEFAULT_DIM",
    "MIN_SCORE",
]
