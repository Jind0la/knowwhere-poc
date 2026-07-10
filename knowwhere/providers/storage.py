"""knowwhere.providers.storage — Pluggable storage backends.

Protocol + two implementations:
    PostgresStorageBackend  — PostgreSQL/pgvector (current, battle-tested)
    SqliteStorageBackend    — SQLite/sqlite-vec (v0.8, zero-dependency)

Usage:
    from knowwhere.providers.storage import get_storage_backend
    backend = get_storage_backend({"storage": {"url": "postgresql://..."}})
    results = backend.search_similar(embedding, top_k=3)
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    """Protocol for storage backends (PostgreSQL, SQLite, etc.)."""

    def insert_source(
        self,
        session_id: str,
        content: str,
        metadata: dict | None = None,
        user_id: str = "default",
    ) -> str:
        """Store raw source text. Returns anchor ID."""
        ...

    def upsert_summary(
        self,
        session_id: str,
        project: str,
        summary_text: str,
        embedding: np.ndarray | None = None,
        tier: str = "warm",
        anchor_id: str | None = None,
        user_id: str = "default",
    ) -> str:
        """Insert or update a summary. Returns summary UUID."""
        ...

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.30,
        project: str | None = None,
        user_id: str = "default",
        ucb_weight: float = 0.5,
        record_access: bool = True,
    ) -> list[dict]:
        """UCB-weighted similarity search. Returns list of summary dicts."""
        ...

    def get_debuts(
        self,
        limit: int = 5,
        user_id: str = "default",
    ) -> list[dict]:
        """Get unseen (debut) summaries for forced injection."""
        ...

    def recall_deep(
        self,
        session_id: str | None = None,
        anchor_id: str | None = None,
        user_id: str = "default",
    ) -> dict:
        """Deep recall: fetch original source text ± context window. Returns {found: bool, ...}."""
        ...

    def health_check(self) -> dict:
        """Return {summaries, sources, debuts_pending, embeddings_present}."""
        ...

    def close(self) -> None:
        """Close connection."""
        ...


# ═══════════════════════════════════════════════════════════════════
# PostgreSQL / pgvector (canonical backend)
# ═══════════════════════════════════════════════════════════════════

class PostgresStorageBackend:
    """PostgreSQL + pgvector backend. Wraps KnowWhereDB."""

    def __init__(self, db_url: str, user_id: str = "default"):
        # Defer import to avoid numpy ImportError at module level
        import sys
        from pathlib import Path
        _repo = Path(__file__).resolve().parent.parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from knowwhere_db import KnowWhereDB
        self._db = KnowWhereDB(db_url)
        self.user_id = user_id

    @property
    def dimension(self) -> int:
        return 256

    def _with_user(self, user_id: str | None = None) -> str:
        return user_id or self.user_id

    def insert_source(
        self,
        session_id: str,
        content: str,
        metadata: dict | None = None,
        user_id: str = "default",
    ) -> str:
        uid = self._with_user(user_id)
        # user_id is stored in metadata until schema supports it natively (v0.8)
        meta = (metadata or {}) | {"user_id": uid}
        return self._db.insert_source(session_id, content, meta)

    def upsert_summary(
        self,
        session_id: str,
        project: str,
        summary_text: str,
        embedding: np.ndarray | None = None,
        tier: str = "warm",
        anchor_id: str | None = None,
        user_id: str = "default",
    ) -> str:
        uid = self._with_user(user_id)
        return self._db.upsert_summary(
            session_id, project, summary_text, embedding, tier, anchor_id
        )

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.30,
        project: str | None = None,
        user_id: str = "default",
        ucb_weight: float = 0.5,
        record_access: bool = True,
    ) -> list[dict]:
        return self._db.search_relevant(
            query_embedding,
            project=project,
            top_k=top_k,
            min_score=min_score,
            ucb_weight=ucb_weight,
            record_access=record_access,
        )

    def get_debuts(
        self,
        limit: int = 5,
        user_id: str = "default",
        project: str | None = None,
    ) -> list[dict]:
        return self._db.get_debuts(project=project, limit=limit)

    def recall_deep(
        self,
        session_id: str | None = None,
        anchor_id: str | None = None,
        user_id: str = "default",
    ) -> dict:
        return self._db.recall_deep(
            session_id=session_id, anchor_id=anchor_id
        )

    def health_check(self) -> dict:
        return self._db.health_check()

    def close(self) -> None:
        self._db.close()


# ═══════════════════════════════════════════════════════════════════
# SQLite / sqlite-vec (zero-dependency mode, v0.8)
# ═══════════════════════════════════════════════════════════════════

class SqliteStorageBackend:
    """SQLite + sqlite-vec backend for zero-dependency deployments.

    Status: STUB for v0.8. Use PostgresStorageBackend in production.
    """

    def __init__(self, db_path: str = "~/.knowwhere/knowwhere.db", user_id: str = "default"):
        self.db_path = str(db_path)
        self.user_id = user_id
        self._conn = None
        raise NotImplementedError(
            "SQLite backend coming in v0.8. "
            "Use PostgresStorageBackend or install PostgreSQL + pgvector."
        )


# ═══════════════════════════════════════════════════════════════════
# Auto-detection
# ═══════════════════════════════════════════════════════════════════

def get_storage_backend(
    config: dict | None = None,
    *,
    db_url: str | None = None,
    user_id: str = "default",
) -> StorageBackend:
    """Auto-detect storage backend from config or environment.

    Priority: db_url arg > config['storage']['url'] > KNOWWHERE_DB_URL env var.

    Returns PostgresStorageBackend for postgres:// URLs.
    SQLite support in v0.8.
    """
    if not db_url and config:
        db_url = config.get("storage", {}).get("url", "")

    if not db_url:
        import os
        db_url = os.environ.get("KNOWWHERE_DB_URL", "")

    if not db_url:
        raise ValueError(
            "No database URL configured. "
            "Set KNOWWHERE_DB_URL or run: knowwhere init"
        )

    if db_url.startswith("postgres") or db_url.startswith("postgresql"):
        return PostgresStorageBackend(db_url, user_id=user_id)

    if db_url.startswith("sqlite"):
        return SqliteStorageBackend(db_url, user_id=user_id)

    # Default: assume PostgreSQL
    return PostgresStorageBackend(db_url, user_id=user_id)
