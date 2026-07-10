"""
KnowWhere Hermes Plugin — subconscious cross-session outcome loop.

Hooks:
  pre_llm_call         — fresh relevance search + optional debut on first turn
  post_llm_call        — nonblocking source + instant summary upsert
  on_session_finalize  — async DeepSeek full summary (replaces instant)
  on_session_reset     — rotate session-bound caches

Tool:
  kw_recall            — deep recall by session_id or anchor_id
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Repo root (knowwhere-poc) for shared modules when installed via symlink.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kw_injection import (  # noqa: E402
    build_search_query,
    filter_guardrails,
    format_injection,
    merge_relevant_and_debuts,
)
from summary_pipeline import (  # noqa: E402
    call_deepseek_full_summary,
    detect_project,
    embed_text,
    format_turns_for_summary,
    make_instant_summary,
)

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("KNOWWHERE_OLLAMA_URL", "http://localhost:11434/api/embed")
OLLAMA_MODEL = "nomic-embed-text"
TRUNC_DIM = 256
SOURCE_CHAR_LIMIT = 4000
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.30
DEFAULT_UCB_WEIGHT = 0.5
DEFAULT_DEBUT_LIMIT = 2
MAX_RECENT_MSGS = 5


def _db_url() -> str:
    return os.environ.get("KNOWWHERE_DB_URL", "")


def _fresh_db():
    from knowwhere_db import KnowWhereDB

    url = _db_url()
    if not url:
        raise RuntimeError("KNOWWHERE_DB_URL not set")
    return KnowWhereDB(db_url=url)


def _embed_query(text: str) -> Optional[np.ndarray]:
    payload = json.dumps({"model": OLLAMA_MODEL, "input": [text[:2000]]}).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        emb = np.array(data["embeddings"][0], dtype=np.float32)[:TRUNC_DIM]
        norm = float(np.linalg.norm(emb)) or 1.0
        return (emb / norm).astype(np.float32)
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("KnowWhere query embed failed: %s", exc)
        return None


class KnowWhereProvider:
    """Hook-driven KnowWhere engine — no MemoryProvider slot."""

    def __init__(self) -> None:
        self._initialized = False
        self._enabled = False
        self._session_id = ""
        self._pre_llm_called_this_turn = False
        self._hook_lock = threading.Lock()
        self._msg_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._recent_user_msgs: List[str] = []
        self._pending_turns: List[dict] = []
        self._pending_lock = threading.Lock()

        self.top_k = int(os.environ.get("KNOWWHERE_TOP_K", DEFAULT_TOP_K))
        self.min_score = float(os.environ.get("KNOWWHERE_MIN_SCORE", DEFAULT_MIN_SCORE))
        self.ucb_weight = float(os.environ.get("KNOWWHERE_UCB", DEFAULT_UCB_WEIGHT))
        self.debut_limit = int(os.environ.get("KNOWWHERE_DEBUT_LIMIT", DEFAULT_DEBUT_LIMIT))

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._enabled = bool(_db_url())
        if self._enabled:
            try:
                db = _fresh_db()
                health = db.health_check()
                db.close()
                self._enabled = health.get("summaries", -1) >= 0
            except Exception as exc:
                logger.warning("KnowWhere init health check failed: %s", exc)
                self._enabled = False
        self._initialized = True

    def _reset_session_state(self, session_id: str) -> None:
        self._session_id = session_id
        self._pre_llm_called_this_turn = False
        with self._msg_lock:
            self._recent_user_msgs.clear()
        with self._pending_lock:
            self._pending_turns.clear()

    def _hook_pre_llm_call(
        self,
        session_id: str = "",
        user_message: str = "",
        is_first_turn: bool = False,
        **kwargs: Any,
    ):
        self._ensure_initialized()
        if not self._enabled:
            return None

        if session_id and session_id != self._session_id:
            self._reset_session_state(session_id)

        with self._hook_lock:
            if self._pre_llm_called_this_turn:
                return None
            self._pre_llm_called_this_turn = True

        query = (user_message or "").strip()
        if query and len(query) >= 3:
            with self._msg_lock:
                self._recent_user_msgs.append(query)
                if len(self._recent_user_msgs) > MAX_RECENT_MSGS:
                    self._recent_user_msgs.pop(0)

        try:
            injection = self._build_injection(query, is_first_turn=is_first_turn)
            if injection:
                return {"context": injection}
        except Exception as exc:
            logger.warning("KnowWhere pre_llm_call failed: %s", exc)
        return None

    def _build_injection(self, user_message: str, *, is_first_turn: bool) -> str:
        search_query = build_search_query(user_message, self._recent_user_msgs)
        emb = _embed_query(search_query)
        if emb is None:
            return ""

        db = _fresh_db()
        try:
            relevant = db.search_hybrid(
                emb,
                query_text=user_message,
                top_k=self.top_k,
                min_score=self.min_score,
                ucb_weight=self.ucb_weight,
            )
            debuts: List[dict] = []
            if is_first_turn:
                debuts = db.get_debuts(limit=self.debut_limit)
                if debuts:
                    db.mark_seen([d["id"] for d in debuts])
            merged = merge_relevant_and_debuts(
                relevant, debuts, debut_limit=self.debut_limit
            )
        finally:
            db.close()

        clean = filter_guardrails(merged)
        return format_injection(clean)

    def _hook_post_llm_call(
        self,
        session_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        **kwargs: Any,
    ) -> None:
        self._ensure_initialized()
        if not self._enabled:
            return

        sid = session_id or self._session_id
        if session_id and session_id != self._session_id:
            self._session_id = session_id

        with self._hook_lock:
            self._pre_llm_called_this_turn = False

        user = (user_message or "")[:SOURCE_CHAR_LIMIT]
        assistant = (assistant_response or "")[:SOURCE_CHAR_LIMIT]
        if not sid or not (user or assistant):
            return

        with self._pending_lock:
            self._pending_turns.append({"user": user, "assistant": assistant})

        content = f"[user] {user}\n[assistant] {assistant}"
        threading.Thread(
            target=self._persist_turn,
            args=(sid, user, assistant, content),
            daemon=True,
        ).start()

    def _persist_turn(
        self, session_id: str, user: str, assistant: str, content: str
    ) -> None:
        try:
            db = _fresh_db()
            try:
                source_id = db.insert_source(
                    session_id,
                    content,
                    metadata={"source": "hermes_hook", "type": "turn"},
                )
                project = detect_project(user, assistant)
                summary_text = make_instant_summary(
                    user,
                    assistant,
                    session_id=session_id,
                    project=project,
                    anchor_id=source_id,
                )
                embedding = embed_text(summary_text)
                db.upsert_summary(
                    session_id=session_id,
                    project=project,
                    summary_text=summary_text,
                    embedding=embedding,
                    tier="warm",
                    anchor_id=source_id,
                )
            finally:
                db.close()
        except Exception as exc:
            logger.debug("KnowWhere turn persist failed: %s", exc)

    def _hook_on_session_finalize(self, session_id: str = "", **kwargs: Any) -> None:
        self._ensure_initialized()
        sid = session_id or self._session_id
        with self._pending_lock:
            pending_count = len(self._pending_turns)
        logger.warning(
            "KnowWhere on_session_finalize: sid=%s enabled=%s pending_turns=%d",
            sid, self._enabled, pending_count,
        )
        if not sid or not self._enabled:
            return

        with self._pending_lock:
            turns = list(self._pending_turns)

        if not turns:
            return

        threading.Thread(
            target=self._finalize_summary,
            args=(sid, turns),
            daemon=True,
        ).start()

    def _finalize_summary(self, session_id: str, turns: List[dict]) -> None:
        try:
            logger.warning(
                "KnowWhere _finalize_summary: starting for sid=%s with %d turns",
                session_id, len(turns),
            )
            text = format_turns_for_summary(turns)
            full = call_deepseek_full_summary(text)
            if not full:
                logger.warning(
                    "KnowWhere _finalize_summary: DeepSeek returned None for sid=%s",
                    session_id,
                )
                return

            logger.warning(
                "KnowWhere _finalize_summary: DeepSeek OK for sid=%s (%d chars)",
                session_id, len(full),
            )

            db = _fresh_db()
            try:
                existing = db.get_summary_by_session(session_id) or {}
                anchor_id = existing.get("anchor_id")
                project = existing.get("project") or detect_project("", text)
                if anchor_id:
                    full = (
                        f"[KnowWhere|sid={session_id}|aid={anchor_id}|project={project}] "
                        f"{full}"
                    )
                embedding = embed_text(full)
                db.upsert_summary(
                    session_id=session_id,
                    project=project,
                    summary_text=full,
                    embedding=embedding,
                    tier="warm",
                    anchor_id=str(anchor_id) if anchor_id else None,
                )
            finally:
                db.close()
        except Exception as exc:
            logger.debug("KnowWhere finalize summary failed: %s", exc)

    def _hook_on_session_reset(self, session_id: str = "", **kwargs: Any) -> None:
        self._ensure_initialized()
        # Finalize the OLD session before resetting state.
        # on_session_reset fires at /new or session rotation — the right
        # boundary for async LLM summarization (not every turn).
        old_sid = self._session_id
        if old_sid and self._enabled:
            with self._pending_lock:
                turns = list(self._pending_turns)
            if turns:
                logger.warning(
                    "KnowWhere session reset: finalizing old sid=%s with %d turns",
                    old_sid, len(turns),
                )
                threading.Thread(
                    target=self._finalize_summary,
                    args=(old_sid, turns),
                    daemon=True,
                ).start()

        if session_id:
            self._reset_session_state(session_id)
        else:
            self._reset_session_state("")

    def _handle_kw_recall(
        self,
        session_id: str = "",
        anchor_id: str = "",
        **kwargs: Any,
    ) -> str:
        self._ensure_initialized()
        if not self._enabled:
            return json.dumps({"error": "KnowWhere not connected"})

        sid = (session_id or "").strip()
        aid = (anchor_id or "").strip()
        try:
            db = _fresh_db()
            try:
                result = db.recall_deep(session_id=sid or None, anchor_id=aid or None)
            finally:
                db.close()
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as exc:
            logger.warning("kw_recall failed: %s", exc)
            return json.dumps({"error": str(exc)})


def register(ctx) -> None:
    """Register hooks and kw_recall tool."""
    provider = KnowWhereProvider()

    ctx.register_hook("pre_llm_call", provider._hook_pre_llm_call)
    ctx.register_hook("post_llm_call", provider._hook_post_llm_call)
    ctx.register_hook("on_session_finalize", provider._hook_on_session_finalize)
    ctx.register_hook("on_session_reset", provider._hook_on_session_reset)

    ctx.register_tool(
        name="kw_recall",
        toolset="knowwhere",
        schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Hermes session id from [KnowWhere|...|sid=...] injection block.",
                },
                "anchor_id": {
                    "type": "string",
                    "description": "Source UUID from [KnowWhere|...|aid=...] for verbatim recall.",
                },
            },
        },
        handler=provider._handle_kw_recall,
        description=(
            "Deep Recall: retrieve original stored text by session_id or anchor_id. "
            "Use when injected summary lacks detail."
        ),
    )

    logger.info("KnowWhere plugin registered (standalone hooks + kw_recall)")
