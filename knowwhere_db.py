#!/usr/bin/env -S $HOME/.hermes/hermes-agent/venv/bin/python3
"""knowwhere_db.py — PostgreSQL/pgvector client for KnowWhere.

Handles all database operations: connection pooling, CRUD for summaries/sources,
embedding-based similarity search via pgvector HNSW index, UCB score management.

Usage:
    from knowwhere_db import KnowWhereDB
    db = KnowWhereDB(os.environ["KNOWWHERE_DB_URL"])
    results = db.search_similar(embedding_vector, top_k=3)
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

# ---- Config ----
# .zshrc is the authoritative source — env vars can be stale from launchd/cron sessions.
# Always read .zshrc, prefer it over the env var when both exist.
_DB_URL = os.environ.get("KNOWWHERE_DB_URL", "")
_ZSH_URL = ""
try:
    import re as _re
    _zshrc = Path.home() / ".zshrc"
    if _zshrc.exists():
        _m = _re.search(r'export KNOWWHERE_DB_URL="([^"]+)"', _zshrc.read_text())
        if _m:
            _ZSH_URL = _m.group(1)
except Exception:
    pass
# .zshrc wins if it exists AND differs from env (env is likely stale from launchd/cron)
if _ZSH_URL and _ZSH_URL != _DB_URL:
    _DB_URL = _ZSH_URL
elif not _DB_URL and _ZSH_URL:
    _DB_URL = _ZSH_URL
DEFAULT_DB_URL = _DB_URL
DEFAULT_DIM = 256  # nomic-embed-text Matryoshka trunkation
MIN_SCORE = 0.30   # Minimum cosine similarity for search results


class KnowWhereDB:
    """KnowWhere database client wrapping PostgreSQL + pgvector."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or DEFAULT_DB_URL
        self._conn: Optional[psycopg2.extensions.connection] = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url, keepalives_idle=60)
            register_vector(self._conn)
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # ---- Summary Store ----

    def upsert_summary(
        self,
        session_id: str,
        project: str,
        summary_text: str,
        embedding: np.ndarray | None = None,
        tier: str = "warm",
    ) -> str:
        """Insert or update a summary. Returns the summary UUID."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO summaries (session_id, project, summary_text, embedding, tier)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET
                       project = EXCLUDED.project,
                       summary_text = EXCLUDED.summary_text,
                       embedding = COALESCE(EXCLUDED.embedding, summaries.embedding),
                       tier = EXCLUDED.tier,
                       updated_at = NOW()
                   RETURNING id""",
                (session_id, project, summary_text, embedding, tier),
            )
            result = str(cur.fetchone()[0])
            self.conn.commit()
            return result

    def search_ucb(
        self,
        query_embedding: np.ndarray,
        project: str | None = None,
        top_k: int = 5,
        min_score: float = MIN_SCORE,
        ucb_weight: float = 0.5,
    ) -> list[dict]:
        """Find top-K summaries using UCB-weighted cosine similarity.
        
        Score = similarity * (1.0 + ucb_weight * (ucb_score - 1.0))
        This gives higher scores to summaries with high UCB (less viewed).
        Debut summaries (debut_seen=FALSE) get priority boost.
        """
        if project:
            query = """
                SELECT id, session_id, project, summary_text, anchor_id,
                       ucb_score, debut_seen, tier, view_count,
                       (1 - (embedding <=> %s::vector)) * 
                       (1.0 + %s * (ucb_score - 1.0)) AS weighted_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM summaries
                WHERE project = %s
                  AND (1 - (embedding <=> %s::vector) >= %s OR debut_seen = FALSE)
                ORDER BY debut_seen ASC, weighted_score DESC
                LIMIT %s
            """
            params = [query_embedding, ucb_weight, query_embedding,
                      project, query_embedding, min_score, top_k]
        else:
            query = """
                SELECT id, session_id, project, summary_text, anchor_id,
                       ucb_score, debut_seen, tier, view_count,
                       (1 - (embedding <=> %s::vector)) *
                       (1.0 + %s * (ucb_score - 1.0)) AS weighted_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM summaries
                WHERE (1 - (embedding <=> %s::vector) >= %s OR debut_seen = FALSE)
                ORDER BY debut_seen ASC, weighted_score DESC
                LIMIT %s
            """
            params = [query_embedding, ucb_weight, query_embedding,
                      query_embedding, min_score, top_k]

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, params)
            results = [dict(r) for r in cur.fetchall()]
            
            # Mark debuts as seen
            debut_ids = [r["id"] for r in results if r["debut_seen"] is False]
            if debut_ids:
                self.mark_seen(debut_ids)
            
            # Record access for UCB update
            all_ids = [r["id"] for r in results]
            if all_ids:
                self.record_access(all_ids)
            
            return results

    def get_debuts(self, project: str | None = None, limit: int = 5) -> list[dict]:
        """Get summaries that have never been injected (debut_seen=FALSE)."""
        if project:
            query = "WHERE debut_seen = FALSE AND project = %s ORDER BY created_at DESC LIMIT %s"
            params = [project, limit]
        else:
            query = "WHERE debut_seen = FALSE ORDER BY created_at DESC LIMIT %s"
            params = [limit]

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"SELECT * FROM summaries {query}", params)
            return [dict(r) for r in cur.fetchall()]

    def mark_seen(self, summary_ids: list[str]) -> int:
        """Mark summaries as seen (debut_seen=TRUE). Does NOT increment view_count."""
        if not summary_ids:
            return 0
        with self.conn.cursor() as cur:
            cur.execute(
                """UPDATE summaries
                   SET debut_seen = TRUE,
                       last_injected = NOW(),
                       updated_at = NOW()
                   WHERE id::text = ANY(%s)""",
                (summary_ids,),
            )
            self.conn.commit()
            return cur.rowcount

    def record_access(self, summary_ids: list[str]) -> int:
        """Increment view_count for accessed summaries (triggers UCB update)."""
        if not summary_ids:
            return 0
        with self.conn.cursor() as cur:
            cur.execute(
                """UPDATE summaries
                   SET view_count = view_count + 1,
                       last_injected = NOW(),
                       updated_at = NOW()
                   WHERE id::text = ANY(%s)""",
                (summary_ids,),
            )
            self.conn.commit()
            return cur.rowcount

    def list_by_project(self, project: str, limit: int = 50) -> list[dict]:
        """List all summaries for a project, newest first."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT id, session_id, summary_text, ucb_score, tier,
                          debut_seen, view_count, last_injected, created_at
                   FROM summaries
                   WHERE project = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (project, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_counts(self) -> dict:
        """Get summary counts by project."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT project, COUNT(*) FROM summaries GROUP BY project ORDER BY COUNT(*) DESC"
            )
            return {r[0]: r[1] for r in cur.fetchall()}

    # ---- Source Store ----

    def insert_source(
        self, session_id: str, full_text: str, metadata: dict | None = None
    ) -> str:
        """Insert a source chunk. Deduplicates by content_hash (SHA-256)."""
        content_hash = hashlib.sha256(full_text.encode()).hexdigest()

        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO sources (session_id, full_text, content_hash, char_count, metadata)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (content_hash) DO UPDATE
                       SET metadata = sources.metadata || EXCLUDED.metadata
                   RETURNING id""",
                (session_id, full_text, content_hash, len(full_text),
                 psycopg2.extras.Json(metadata or {})),
            )
            result = str(cur.fetchone()[0])
            self.conn.commit()
            return result

    def get_source(self, source_id: str) -> dict | None:
        """Get a source by UUID."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM sources WHERE id = %s", (source_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_sources_by_session(
        self, session_id: str, limit: int = 5
    ) -> list[dict]:
        """Get recent sources for a session, newest first.
        Used by kw_recall for Deep Recall by session context.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT id, session_id, full_text, char_count, created_at, metadata
                   FROM sources
                   WHERE session_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (session_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    # ---- Vector Index ----

    def index_vector(
        self, source_id: str, embedding: np.ndarray, preview: str, project: str | None = None
    ) -> str:
        """Store an embedding in the vector index."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO vector_index (source_id, embedding, preview, project)
                   VALUES (%s, %s::vector, %s, %s)
                   RETURNING id""",
                (source_id, embedding, preview[:200], project),
            )
            result = str(cur.fetchone()[0])
            self.conn.commit()
            return result

    # ---- Stats ----

    def health_check(self) -> dict:
        """Quick database health check."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM summaries")
            summary_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM sources")
            source_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM vector_index")
            index_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM summaries WHERE debut_seen = FALSE")
            debut_count = cur.fetchone()[0]

        return {
            "summaries": summary_count,
            "sources": source_count,
            "vector_index": index_count,
            "debuts_pending": debut_count,
        }


# ---- Singleton ----

_db_instance: Optional[KnowWhereDB] = None


def get_db() -> KnowWhereDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = KnowWhereDB()
    return _db_instance


# ---- CLI (for testing) ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--counts", action="store_true")
    args = parser.parse_args()

    db = KnowWhereDB()
    try:
        if args.health:
            health = db.health_check()
            print("✅ KnowWhere DB Health:")
            for k, v in health.items():
                print(f"   {k}: {v}")
        elif args.counts:
            counts = db.get_counts()
            print("📊 Summaries by project:")
            for proj, cnt in counts.items():
                print(f"   {proj}: {cnt}")
            if not counts:
                print("   (empty)")
        else:
            print("Usage: knowwhere_db.py --health | --counts")
    finally:
        db.close()
