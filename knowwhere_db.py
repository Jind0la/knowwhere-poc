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
        anchor_id: str | None = None,
    ) -> str:
        """Insert or update a summary. Returns the summary UUID."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO summaries (session_id, project, summary_text, embedding, tier, anchor_id)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET
                       project = EXCLUDED.project,
                       summary_text = EXCLUDED.summary_text,
                       embedding = COALESCE(EXCLUDED.embedding, summaries.embedding),
                       tier = EXCLUDED.tier,
                       anchor_id = COALESCE(EXCLUDED.anchor_id, summaries.anchor_id),
                       updated_at = NOW()
                   RETURNING id""",
                (session_id, project, summary_text, embedding, tier, anchor_id),
            )
            result = str(cur.fetchone()[0])
            self.conn.commit()
            return result

    def search_relevant(
        self,
        query_embedding: np.ndarray,
        project: str | None = None,
        top_k: int = 5,
        min_score: float = MIN_SCORE,
        ucb_weight: float = 0.5,
        *,
        record_access: bool = True,
        session_id_prefix: str | None = None,
    ) -> list[dict]:
        """UCB-weighted similarity search — relevance only, no debut bypass."""
        prefix_clause = ""
        prefix_param: list[Any] = []
        if session_id_prefix:
            prefix_clause = " AND session_id LIKE %s"
            prefix_param = [f"{session_id_prefix}%"]

        if project:
            query = f"""
                SELECT id, session_id, project, summary_text, anchor_id,
                       ucb_score, debut_seen, tier, view_count,
                       (1 - (embedding <=> %s::vector)) *
                       (1.0 + %s * (ucb_score - 1.0)) AS weighted_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM summaries
                WHERE project = %s
                  AND embedding IS NOT NULL
                  AND (1 - (embedding <=> %s::vector) >= %s){prefix_clause}
                ORDER BY weighted_score DESC
                LIMIT %s
            """
            params = [
                query_embedding,
                ucb_weight,
                query_embedding,
                project,
                query_embedding,
                min_score,
                *prefix_param,
                top_k,
            ]
        else:
            query = f"""
                SELECT id, session_id, project, summary_text, anchor_id,
                       ucb_score, debut_seen, tier, view_count,
                       (1 - (embedding <=> %s::vector)) *
                       (1.0 + %s * (ucb_score - 1.0)) AS weighted_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM summaries
                WHERE embedding IS NOT NULL
                  AND (1 - (embedding <=> %s::vector) >= %s){prefix_clause}
                ORDER BY weighted_score DESC
                LIMIT %s
            """
            params = [
                query_embedding,
                ucb_weight,
                query_embedding,
                query_embedding,
                min_score,
                *prefix_param,
                top_k,
            ]

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, params)
            results = [dict(r) for r in cur.fetchall()]

        if record_access and results:
            self.record_access([r["id"] for r in results])

        return results

    def search_ucb(
        self,
        query_embedding: np.ndarray,
        project: str | None = None,
        top_k: int = 5,
        min_score: float = MIN_SCORE,
        ucb_weight: float = 0.5,
    ) -> list[dict]:
        """Find top-K summaries using UCB-weighted cosine similarity.

        Backward-compatible alias for relevance search (no debut bypass).
        """
        results = self.search_relevant(
            query_embedding,
            project=project,
            top_k=top_k,
            min_score=min_score,
            ucb_weight=ucb_weight,
        )
        debut_ids = [r["id"] for r in results if r.get("debut_seen") is False]
        if debut_ids:
            self.mark_seen(debut_ids)
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

    def get_source_by_anchor(self, anchor_id: str) -> dict | None:
        """Deep recall: fetch source row by anchor UUID (sources.id)."""
        return self.get_source(anchor_id)

    def recall_deep(
        self,
        *,
        session_id: str | None = None,
        anchor_id: str | None = None,
        limit: int = 5,
    ) -> dict:
        """Deep recall by session_id or anchor_id; returns original full_text."""
        if anchor_id:
            source = self.get_source_by_anchor(anchor_id)
            if not source:
                return {"found": False, "anchor_id": anchor_id}
            return {
                "found": True,
                "anchor_id": str(source["id"]),
                "session_id": source.get("session_id"),
                "full_text": source["full_text"],
                "char_count": source.get("char_count"),
                "created_at": str(source.get("created_at", "")),
            }

        sid = (session_id or "").strip()
        if not sid:
            return {"found": False, "error": "session_id or anchor_id required"}

        sources = self.get_sources_by_session(sid, limit=limit)
        if not sources:
            return {"found": False, "session_id": sid}
        return {
            "found": True,
            "session_id": sid,
            "source_count": len(sources),
            "sources": [
                {
                    "id": str(s["id"]),
                    "full_text": s["full_text"],
                    "char_count": s["char_count"],
                    "created_at": str(s.get("created_at", "")),
                }
                for s in sources
            ],
        }

    def cleanup_fixture_prefix(self, prefix: str) -> dict:
        """Delete test fixtures by session_id prefix (summaries then sources)."""
        if not prefix or len(prefix) < 8:
            raise ValueError("fixture prefix must be at least 8 chars")
        pattern = prefix if prefix.endswith("%") else f"{prefix}%"
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM summaries WHERE session_id LIKE %s RETURNING id",
                (pattern,),
            )
            summary_ids = [str(r[0]) for r in cur.fetchall()]
            cur.execute(
                "DELETE FROM sources WHERE session_id LIKE %s RETURNING id",
                (pattern,),
            )
            source_ids = [str(r[0]) for r in cur.fetchall()]
            self.conn.commit()
        return {
            "summaries_deleted": len(summary_ids),
            "sources_deleted": len(source_ids),
        }

    def get_summary_by_session(self, session_id: str) -> dict | None:
        """Fetch summary row for a session."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT id, session_id, project, summary_text, anchor_id FROM summaries WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def link_summary_to_sources(self, session_id: str) -> dict:
        """Bridge the provenance gap: link summary → its most recent source.

        Sets summaries.anchor_id to the most recent sources.id for the
        same session_id. This enables Deep Recall (kw_recall) to retrieve
        full source text from injected summary blocks.

        Returns a dict with rows_updated count and the anchor_id that was set.
        """
        with self.conn.cursor() as cur:
            # Find the most recent source for this session
            cur.execute(
                """SELECT id FROM sources
                   WHERE session_id = %s
                   ORDER BY created_at DESC LIMIT 1""",
                (session_id,),
            )
            source_row = cur.fetchone()
            if not source_row:
                self.conn.commit()
                return {"rows_updated": 0, "error": "No source found for session"}

            anchor_id = str(source_row[0])

            # Link the summary to this source
            cur.execute(
                """UPDATE summaries
                   SET anchor_id = %s,
                       updated_at = NOW()
                   WHERE session_id = %s
                     AND anchor_id IS NULL""",
                (anchor_id, session_id),
            )
            rows = cur.rowcount
            self.conn.commit()
            return {"rows_updated": rows, "anchor_id": anchor_id}

    def get_provenance_chain(self, summary_id: str) -> dict | None:
        """Check the full provenance chain: summary → source.

        Returns a dict with the summary and its linked source text,
        or a 'gap' key if the chain is broken.
        """
        with self.conn.cursor(
            cursor_factory=psycopg2.extras.DictCursor
        ) as cur:
            cur.execute(
                """SELECT s.id AS summary_id, s.session_id, s.project,
                          s.summary_text, s.anchor_id,
                          src.full_text AS source_text, src.char_count,
                          src.created_at AS source_created
                   FROM summaries s
                   LEFT JOIN sources src ON s.anchor_id = src.id
                   WHERE s.id = %s""",
                (summary_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            result = dict(row)
            if result["anchor_id"] is None:
                result["gap"] = (
                    "No anchor — provenance chain broken. "
                    "Run link_summary_to_sources() to bridge."
                )
            elif result["source_text"] is None:
                result["gap"] = (
                    "Anchor exists but source deleted — "
                    "chain is severed."
                )
            return result

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
