#!/usr/bin/env -S $HOME/.hermes/hermes-agent/venv/bin/python3
"""retrieve_summaries.py — Query-Embedding → CosSim → Top-N Summaries.

Nimmt einen Query-String, embedded ihn via Ollama nomic-embed-text,
findet die ähnlichsten Summaries per Cosine Similarity aus embeddings.npz.

Usage:
    python3 retrieve_summaries.py "KnowWhere Railway Deployment" --top 3 [--json]
    
Output (--json):
    [{rank, score, type, date, short_id, title, text}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HOME = Path.home()
DEFAULT_EMBEDDINGS = HOME / ".hermes" / "knowwhere" / "embeddings.npz"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def embed_query(query: str) -> np.ndarray:
    """Embed a single query via Ollama."""
    payload = json.dumps({
        "model": MODEL,
        "input": [query]
    }).encode()
    
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        embedding = np.array(data["embeddings"][0], dtype=np.float32)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) or 1.0)
        return embedding
    except Exception as e:
        print(f"❌ Ollama embed error: {e}", file=sys.stderr)
        raise


def retrieve(query: str, top_k: int = 3, embeddings_path: str = None) -> list[dict]:
    """Retrieve top-K summaries for a query. Returns ranked list."""
    path = Path(embeddings_path or DEFAULT_EMBEDDINGS)
    if not path.exists():
        print(f"❌ Embeddings nicht gefunden: {path}", file=sys.stderr)
        print("   Zuerst embed_summaries.py ausführen.", file=sys.stderr)
        return []
    
    # Load embeddings
    data = np.load(path, allow_pickle=True)
    stored_embeddings = data["embeddings"]  # (N, 768)
    texts = data["texts"]
    short_ids = data["short_ids"]
    titles = data["titles"]
    
    # Get query embedding
    query_emb = embed_query(query)
    
    # Cosine similarity (embeddings are already normalized)
    scores = np.dot(stored_embeddings, query_emb)
    
    # Top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < 0.3:  # Minimum relevance threshold
            continue
        results.append({
            "rank": len(results) + 1,
            "score": round(score, 4),
            "type": str(data["type"][idx]) if "type" in data else "unknown",
            "date": str(data["date"][idx]) if "date" in data else "",
            "short_id": str(short_ids[idx]),
            "title": str(titles[idx]),
            "text": str(texts[idx])
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    args = parser.parse_args()
    
    query = " ".join(args.query)
    results = retrieve(query, top_k=args.top, embeddings_path=args.embeddings)
    
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        if not results:
            print("Keine relevanten Summaries gefunden.")
            return
        
        for r in results:
            sid = r["short_id"] or "-"
            print(f"#{r['rank']} [{sid}] {r['title']} (score: {r['score']})")
            print(f"   {r['text'][:120]}...")
            print()


if __name__ == "__main__":
    main()
