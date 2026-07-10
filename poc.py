#!/usr/bin/env python3.11
"""KnowWhere v0 PoC — Embedding-Similarity-Suche in Nimars echten Session-Daten.

Kern-Hypothese: Findet Embedding-Ähnlichkeit überhaupt relevante Memories?
Datenquelle: Daily Logs (~/.hermes/logs/*.md), ca. 55 Tage, ~400 Chunks.
"""

import argparse, json, os, re, sys, time, urllib.request, urllib.error
from pathlib import Path

import numpy as np

OLLAMA_EMBED = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
LOGS_DIR = Path.home() / ".hermes" / "logs"
BATCH_SIZE = 25
MAX_CHARS = 1500  # Truncate chunks to ~nomic-embed-text 2048 token limit


def embed_batch(texts: list[str], dims: int = 768) -> np.ndarray:
    """Embed all texts using /api/embed batch endpoint."""
    all_vecs = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        done = min(i + BATCH_SIZE, len(texts))
        payload = json.dumps({"model": MODEL, "input": batch}).encode()
        req = urllib.request.Request(OLLAMA_EMBED, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        vecs = np.array(data["embeddings"], dtype=np.float32)
        if dims < 768:
            vecs = vecs[:, :dims]
        all_vecs.append(vecs)
        rate = done / (time.time() - t0)
        eta = (len(texts) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(texts)}] {rate:.0f}/s, ETA {eta:.0f}s", file=sys.stderr)
    return np.vstack(all_vecs)


def chunk_daily_logs(logs_dir: Path) -> list[dict]:
    chunks = []
    entry_start = re.compile(r"^- \[")
    for logfile in sorted(logs_dir.glob("*.md")):
        date = logfile.stem
        lines = logfile.read_text().split("\n")
        batch = []
        for line in lines:
            stripped = line.strip()
            if entry_start.match(stripped) or stripped.startswith("## ") or stripped.startswith("### "):
                if batch:
                    joined = " ".join(batch)
                    if len(joined) > 50:
                        chunks.append({"text": joined[:MAX_CHARS], "source": date, "date": date,
                                       "full_text": joined})
                    batch = []
            if stripped and not stripped.startswith("# "):
                batch.append(stripped)
        if batch:
            joined = " ".join(batch)
            if len(joined) > 50:
                chunks.append({"text": joined[:MAX_CHARS], "source": date, "date": date,
                               "full_text": joined})
    return chunks


def cosine_sim(query: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-8)
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return v @ q


def embed_one(text: str, dims: int) -> np.ndarray:
    payload = json.dumps({"model": MODEL, "input": [text]}).encode()
    req = urllib.request.Request(OLLAMA_EMBED, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    emb = np.array(data["embeddings"][0], dtype=np.float32)
    return emb[:dims] if dims < 768 else emb


def search(args, chunks, vecs):
    query_text = " ".join(args.query)
    print(f"\n🔎 Query: \"{query_text}\"", file=sys.stderr)
    qv = embed_one(query_text, dims=vecs.shape[1])
    sims = cosine_sim(qv, vecs)
    sims_clamped = np.clip(sims, -1, 1)
    top_idx = np.argsort(sims)[::-1][:args.top]

    for rank, idx in enumerate(top_idx):
        c = chunks[idx]
        sc = sims_clamped[idx]
        print(f"\n{'─'*60}")
        print(f"#{rank+1}  CosSim: {sc:.4f}  │  {c['source']}")
        print(f"    {c['text'][:300]}{'…' if len(c['text'])>300 else ''}")
    print(f"\n{'─'*60}")
    print(f"\n📊 range [{sims_clamped[top_idx[-1]]:.4f}, {sims_clamped[top_idx[0]]:.4f}], "
          f"mean top{args.top}: {sims_clamped[top_idx].mean():.4f}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="KnowWhere PoC")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--dims", type=int, default=768, choices=[64, 256, 768])
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    CACHE = Path("poc_cache.npz")

    if CACHE.exists() and not args.rebuild:
        print("📦 Cache geladen.", file=sys.stderr)
        data = np.load(CACHE, allow_pickle=True)
        vecs = data["vecs"]
        chunks = data["chunks"].tolist()
    else:
        chunks = chunk_daily_logs(LOGS_DIR)
        ndays = len(set(c["date"] for c in chunks))
        print(f"📄 {len(chunks)} Chunks aus {ndays} Tagen (max {MAX_CHARS} chars)", file=sys.stderr)
        texts = [c["text"] for c in chunks]
        print(f"🧮 Batch-Embedding {len(texts)} Chunks ({args.dims}d)...", file=sys.stderr)
        t0 = time.time()
        vecs = embed_batch(texts, dims=args.dims)
        print(f"   ✅ {time.time()-t0:.1f}s ({len(texts)/(time.time()-t0):.0f} chunks/s)", file=sys.stderr)
        np.savez(CACHE, vecs=vecs, chunks=np.array(chunks, dtype=object))

    if args.query:
        search(args, chunks, vecs)
    else:
        print(f"\n🔍 {len(chunks)} Chunks, {vecs.shape[1]}d. 'exit' zum Beenden.\n")
        while True:
            try:
                q = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            args.query = q.split()
            search(args, chunks, vecs)


if __name__ == "__main__":
    main()
