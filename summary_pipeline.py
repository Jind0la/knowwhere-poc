"""summary_pipeline.py — Instant + full summaries and Ollama embeddings."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("KNOWWHERE_OLLAMA_URL", "http://localhost:11434/api/embed")
OLLAMA_MODEL = "nomic-embed-text"
TRUNC_DIM = 256
EMBED_TIMEOUT = 30
DEEPSEEK_TIMEOUT = 45

INSTANT_MAX_CHARS = 500
FULL_MAX_CHARS = 500


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).astype(np.float32)


def embed_text(text: str) -> Optional[np.ndarray]:
    """Embed text via Ollama; return L2-normalized 256d vector or None."""
    if not (text or "").strip():
        return None

    payload = json.dumps({"model": OLLAMA_MODEL, "input": [text[:2000]]}).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read())
        emb = np.array(data["embeddings"][0], dtype=np.float32)[:TRUNC_DIM]
        return _normalize(emb)
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Ollama embed failed: %s", exc)
        return None


def _clip_max(text: str, hi: int) -> str:
    """Trim whitespace and enforce upper bound only — never pad or invent content."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= hi:
        return text
    cut = text[: hi - 3].rsplit(" ", 1)[0]
    return cut + "..."


def make_instant_summary(
    user_message: str,
    assistant_message: str,
    *,
    session_id: str,
    project: str,
    anchor_id: str | None = None,
) -> str:
    """Build a self-contained instant summary (≤500 chars, no filler)."""
    user = (user_message or "").strip()[:800]
    assistant = (assistant_message or "").strip()[:1200]
    aid = anchor_id or "pending"
    head = f"[KnowWhere|sid={session_id}|aid={aid}|project={project}]"
    body = f"User: {user} Outcome: {assistant}"
    summary = f"{head} {body}"
    return _clip_max(summary, INSTANT_MAX_CHARS)


def call_deepseek_full_summary(turns_text: str) -> Optional[str]:
    """Generate a richer session summary via DeepSeek; None on failure."""
    from hermes_env import get_secret

    api_key = get_secret("DEEPSEEK_API_KEY", "")
    if not api_key or not (turns_text or "").strip():
        return None

    system = (
        "Summarize this Hermes session for cross-session agent memory. "
        "Include root cause, decision, and exact fix if any. "
        f"Max {FULL_MAX_CHARS} characters. German if the session is German. "
        "Self-contained; no meta commentary."
    )
    payload = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": turns_text[:8000]},
            ],
            "max_tokens": 220,
            "temperature": 0.2,
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=DEEPSEEK_TIMEOUT) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        return _clip_max(text, FULL_MAX_CHARS)
    except Exception as exc:
        logger.warning("DeepSeek full summary failed: %s", exc)
        return None


def detect_project(title: str, content: str) -> str:
    """Reuse nightly pipeline project detection."""
    try:
        from summarize_today import detect_project as _detect

        return _detect(title or "", content or "")
    except Exception:
        return "General"


def format_turns_for_summary(turns: list[dict]) -> str:
    """Join accumulated turns for full summarization."""
    parts = []
    for t in turns:
        u = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if u:
            parts.append(f"User: {u}")
        if a:
            parts.append(f"Assistant: {a}")
    return "\n".join(parts)
