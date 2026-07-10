#!/usr/bin/env -S $HOME/.hermes/hermes-agent/venv/bin/python3
"""embed_summaries.py v3 — Embeds summaries from pgvector via Ollama, writes back.

Reads un-embedded summaries from PostgreSQL/pgvector, embeds them via
Ollama nomic-embed-text (768d, then truncates to 256d via Matryoshka),
updates the rows with embeddings.

Also reads from summaries.json as fallback if KNOWWHERE_DB_URL is not set.

Usage:
    python3 embed_summaries.py [--json-fallback]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HOME = Path.home()
DEFAULT_JSON_INPUT = HOME / ".hermes" / "knowwhere" / "summaries.json"
DEFAULT_NPZ_OUTPUT = HOME / ".hermes" / "knowwhere" / "embeddings.npz"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
FULL_DIM = 768
TRUNC_DIM = 256  # Matryoshka trunkation for pgvector
BATCH_SIZE = 25   # Ollama batch limit


def check_ollama() -> bool:
    """Verify Ollama is reachable."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        return True
    except Exception:
        return False


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts via Ollama batch API. Returns (N, 768) array."""
    payload = json.dumps({
        "model": MODEL,
        "input": texts
    }).encode()
    
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return embeddings
    except Exception as e:
        print(f"❌ Ollama embed error: {e}", file=sys.stderr)
        raise


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def truncate_dim(embeddings: np.ndarray, dim: int = TRUNC_DIM) -> np.ndarray:
    """Truncate to lower dimension via Matryoshka, then re-normalize."""
    return normalize(embeddings[:, :dim])


def embed_from_pgvector() -> int:
    """Read un-embedded summaries from pgvector, embed them, update rows."""
    from knowwhere_db import get_db
    
    db = get_db()
    try:
        # Find summaries without embeddings
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, summary_text FROM summaries
                WHERE embedding IS NULL
                ORDER BY created_at ASC
            """)
            rows = cur.fetchall()
        
        if not rows:
            print("✅ Alle Summaries bereits embedded (pgvector)")
            return 0
        
        print(f"📊 {len(rows)} un-embedded Summaries in pgvector")
        
        # Process in batches
        total_updated = 0
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]
            
            print(f"   Batch {i // BATCH_SIZE + 1}: Embedding {len(texts)} texts...")
            embeddings = embed_texts(texts)
            trunc = truncate_dim(embeddings)
            
            # Update each row
            for j, row_id in enumerate(ids):
                with db.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE summaries SET embedding = %s::vector, updated_at = NOW() WHERE id = %s",
                        (trunc[j], row_id),
                    )
                total_updated += 1
            
            db.conn.commit()
        
        print(f"💾 pgvector: {total_updated} embeddings updated")
        return total_updated
    finally:
        db.close()


def embed_from_json(output_path: str | None = None) -> int:
    """Read summaries.json, embed, write embeddings.npz (JSON fallback)."""
    input_path = DEFAULT_JSON_INPUT
    if not input_path.exists():
        print(f"❌ Input nicht gefunden: {input_path}")
        return 0
    
    with open(input_path) as f:
        data = json.load(f)
    
    items = []
    for date_key, day_data in data.items():
        # Combined summary
        if "summary" in day_data and day_data["summary"]:
            items.append({
                "type": "daily",
                "date": date_key,
                "text": day_data["summary"],
                "session_count": day_data.get("session_count", 0)
            })
        
        # Per-session summaries
        for short_id, sess in day_data.get("sessions", {}).items():
            if sess.get("summary"):
                items.append({
                    "type": "session",
                    "date": date_key,
                    "short_id": short_id,
                    "title": sess.get("title", ""),
                    "text": sess["summary"],
                    "msgs": sess.get("msgs", 0)
                })
    
    if not items:
        print("❌ Keine Summaries zum Embedden gefunden")
        return 0
    
    print(f"📊 {len(items)} Texte zum Embedden")
    
    texts = [item["text"] for item in items]
    embeddings = embed_texts(texts)
    embeddings = normalize(embeddings)
    
    # Build metadata
    metadata = {
        "type": np.array([i["type"] for i in items]),
        "date": np.array([i.get("date", "") for i in items]),
    }
    short_ids = [i.get("short_id", "") for i in items]
    titles = [i.get("title", "") for i in items]
    texts_arr = np.array(texts)
    
    out_path = Path(output_path) if output_path else DEFAULT_NPZ_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        texts=texts_arr,
        short_ids=np.array(short_ids),
        titles=np.array(titles),
        **metadata
    )
    
    print(f"📦 NPZ gespeichert: {out_path}")
    print(f"   ({len(items)} Eintraege, {embeddings.shape[1]}d Vektoren)")
    return len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-fallback", action="store_true",
                        help="Read from summaries.json and write NPZ (skip pgvector)")
    parser.add_argument("--output", default=None, help="NPZ output path (JSON fallback only)")
    args = parser.parse_args()
    
    if not check_ollama():
        print("❌ Ollama nicht erreichbar. Laeuft der Server?")
        sys.exit(1)
    
    print(f"🔌 Ollama OK ({MODEL})\n")
    
    has_db = bool(os.environ.get("KNOWWHERE_DB_URL"))
    
    if not args.json_fallback and has_db:
        # Primary path: pgvector
        count = embed_from_pgvector()
        # Also update NPZ for backward compatibility
        if count > 0:
            print("\n📦 Updating JSON/NPZ fallback...")
            embed_from_json()
    else:
        # Fallback: JSON → NPZ
        embed_from_json(output_path=args.output)


if __name__ == "__main__":
    main()
