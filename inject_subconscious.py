#!/usr/bin/env -S $HOME/.hermes/hermes-agent/venv/bin/python3
"""inject_subconscious.py v3 — Queries pgvector for KnowWhere Subconscious Injection.

Embeds the current project context via Ollama, retrieves top-N similar
summaries from PostgreSQL/pgvector via HNSW cosine distance, formats as
context blocks for agent injection.

Usage:
    python3 inject_subconscious.py [--top 3] [--json] [--project KnowWhere]

Environment:
    KNOWWHERE_QUERY: Optional query string override
    KNOWWHERE_DB_URL: PostgreSQL connection (falls back to NPZ if unset)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

HOME = Path.home()
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
TRUNC_DIM = 256
TOP_K = 3
MIN_SCORE = 0.30


def build_query() -> str:
    """Build a context-aware query from today's daily log keywords + env override."""
    env_query = os.environ.get("KNOWWHERE_QUERY", "")
    if env_query:
        return env_query
    
    from datetime import date
    log_path = HOME / ".hermes" / "logs" / f"{date.today().isoformat()}.md"
    if not log_path.exists():
        return "KnowWhere Subconscious Memory Agent"
    
    text = log_path.read_text()[:5000]
    projects = set(re.findall(
        r'\b(KnowWhere|Moradbakhti|Krankenfahrt|Cafe\s*Agnes|Leaf\s*Go|Railway|Ollama|DeepSeek|pgvector)\b',
        text
    ))
    
    if projects:
        return " ".join(sorted(projects))
    return "KnowWhere Subconscious Memory Agent"


def embed_query(query: str) -> np.ndarray:
    """Embed a query via Ollama, truncate to TRUNC_DIM, normalize."""
    payload = json.dumps({"model": MODEL, "input": [query]}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        emb = np.array(data["embeddings"][0], dtype=np.float32)
        # Truncate to match pgvector dimension
        emb = emb[:TRUNC_DIM]
        return emb / (np.linalg.norm(emb) or 1.0)
    except Exception as e:
        print(f"⚠️ Ollama nicht erreichbar: {e}", file=sys.stderr)
        print("[KnowWhere: Ollama down — Subconscious-Injektion uebersprungen]")
        sys.exit(0)


# ---- pgvector retrieval ----

def retrieve_from_pgvector(query_emb: np.ndarray, top_k: int = TOP_K,
                           project: str | None = None) -> list[dict]:
    """Retrieve top-K summaries from pgvector via UCB-weighted HNSW search.
    Automatically marks debuts as seen and records access for UCB updates."""
    from knowwhere_db import get_db
    
    db = get_db()
    try:
        results = db.search_ucb(query_emb, project=project, top_k=top_k)
        return results
    finally:
        db.close()


# ---- NPZ fallback retrieval ----

def retrieve_from_npz(query_emb: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-K summaries from embeddings.npz (fallback)."""
    emb_path = HOME / ".hermes" / "knowwhere" / "embeddings.npz"
    if not emb_path.exists():
        return []
    
    data = np.load(emb_path, allow_pickle=True)
    stored = data["embeddings"]
    texts = data["texts"]
    short_ids = data["short_ids"]
    titles = data["titles"]
    
    # Truncate stored embeddings to match query dim
    stored_trunc = stored[:, :TRUNC_DIM]
    stored_norm = stored_trunc / (np.linalg.norm(stored_trunc, axis=1, keepdims=True) + 1e-8)
    
    scores = np.dot(stored_norm, query_emb[:TRUNC_DIM])
    
    results = []
    for idx in np.argsort(scores)[::-1][:top_k]:
        score = float(scores[idx])
        if score < MIN_SCORE:
            continue
        results.append({
            "short_id": str(short_ids[idx]),
            "title": str(titles[idx]),
            "text": str(texts[idx]),
            "score": round(score, 4),
        })
    
    return results


# ---- Formatting ----

def format_for_context(results: list[dict]) -> str:
    """Format retrieval results as KnowWhere context blocks for agent injection."""
    if not results:
        return ""
    
    blocks = ["[KnowWhere Subconscious — Top Session Summaries]"]
    for i, r in enumerate(results):
        sid = r.get("short_id") or r.get("session_id", "")[-6:] or "-"
        title = r.get("title") or r.get("project", "")
        score = r.get("score") or r.get("similarity", 0)
        text = r.get("summary_text") or r.get("text", "")
        
        blocks.append(f"[{i+1}] [{sid}] {title} (score: {score:.4f})")
        if len(text) > 300:
            text = text[:297] + "..."
        blocks.append(f"    {text}")
    
    blocks.append("[End KnowWhere]")
    return "\n".join(blocks)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=TOP_K)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--project", default=None,
                        help="Filter results to specific project")
    args = parser.parse_args()
    
    query = args.query or build_query()
    query_emb = embed_query(query)
    has_db = bool(os.environ.get("KNOWWHERE_DB_URL"))
    
    # Try pgvector first, fall back to NPZ
    if has_db:
        results = retrieve_from_pgvector(query_emb, top_k=args.top,
                                         project=args.project)
        if not results:
            # Fallback to NPZ
            results = retrieve_from_npz(query_emb, top_k=args.top)
    else:
        results = retrieve_from_npz(query_emb, top_k=args.top)
    
    if args.json:
        print(json.dumps({
            "query": query,
            "source": "pgvector" if has_db else "npz",
            "results": [{
                "short_id": r.get("short_id", r.get("session_id", ""))[:8],
                "title": r.get("title", r.get("project", "")),
                "score": r.get("score", r.get("similarity", 0)),
                "text": r.get("summary_text", r.get("text", "")),
            } for r in results]
        }, indent=2, ensure_ascii=False, default=str))
    else:
        output = format_for_context(results)
        if output:
            print(output)
        else:
            print("[KnowWhere: Keine relevanten Summaries gefunden]")


if __name__ == "__main__":
    main()
