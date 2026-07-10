#!/usr/bin/env python3.11
"""dimtest_auto.py — Test: Automatic summarization + retrieval + dimtest A/B

Phase 1: Load LLM-generated summaries, embed them, run retrieval test
Phase 2: Print injection blocks for delegate_task A/B test
"""
import json, sys, time, urllib.request
import numpy as np
from pathlib import Path

OLLAMA_EMBED = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

SUMMARIES = [
    "[KnowWhere|aid=leafgo_001|2026-06-30] Leaf Go: CEO-Pitch + DSGVO-Erweiterung. Drei technische Fixes: (1) Video-Scrubbing defekt — libx264 ignorierte `-g 1` ohne `-keyint_min 1`. Fix: `-g 1 -keyint_min 1 -crf 22`. (2) Playwright-PDF zeigte Timestamp/\"about:blank\" — leere header/footer-template beseitigt Chromium-Default. (3) JS (Z.334) überschrieb CSS brightness(0.9) mit Inline-0.5 — Fix: `lerp(0.9, 0.45, dim)`. Cover-Screenshot auf typografisches Cover umgestellt.",
    "[KnowWhere|aid=moradbakhti_001|2026-06-22] Session 2026-06-22 zum Moradbakhti Portfolio-Redesign auf dem Branch design/geil. Wichtige Fortschritte: (1) 3-Strike Safety Valve entdeckt — ein Sicherheitsmechanismus für wiederholte Fehlschläge. (2) Terminal-Toolset-Architektur als Gewinn gefeiert — stabilisiert die Tool-Nutzung. (3) Zwei-Loop-Workflow fürs Portfolio-Design etabliert: kreativer Loop + technischer Loop. Zudem: Loop-Metriken umbenannt und Content-Degradation-Reinjektion gefixt. 368 Nachrichten, 180 Tools, ~65min Leerlauf.",
    "[KnowWhere|aid=era_podcast_001|2026-07-02] Podcast-Recherche zu JRE #2521 mit Perplexity-CEO Aravind Srinivas via Twitter, Reddit und jina.ai. Fünf Key Takeaways extrahiert, Pitch-Framing für Moradbakhti-KI abgeleitet: „Die knappe Ressource ist die Frage.\" — 100 Nachrichten, 43 Tools. Parallel dazu Era-Pet-Spritesheet optimiert: build_clean_v3.py entfernte transparente Pixel-Artefakte, Color-Key-Extraktion von rembg auf direkten Hintergrund-Color-Key (RGB ~34,32,43) umgestellt. Nimar bestand auf makelloser Transparenz — keine Löcher, Halos oder Artefakte.",
]

QUERIES = [
    "Was war das Problem mit dem Leafgo PDF und Video-Scrubbing?",
    "Was haben wir beim Moradbakhti Portfolio Redesign für Architekturentscheidungen getroffen?",
    "Wie haben wir das Era Spritesheet Pixel-Problem gelöst?",
]


def embed_batch(texts: list[str], dims: int = 256) -> np.ndarray:
    """Embed texts via Ollama batch API."""
    payload = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(OLLAMA_EMBED, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    vecs = np.array(data["embeddings"], dtype=np.float32)
    return vecs[:, :dims] if dims < 768 else vecs


def cosine_sim(query: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-8)
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return v @ q


def main():
    if len(SUMMARIES) < 3:
        print("❌ SUMMARIES list is empty. Fill it with subagent output first.")
        sys.exit(1)
    
    print(f"📊 {len(SUMMARIES)} Summaries, {len(QUERIES)} Queries")
    
    # Embed summaries (256d)
    print("🧮 Embedding summaries (256d)...")
    summary_vecs = embed_batch(SUMMARIES, dims=256)
    
    # Embed queries
    print("🧮 Embedding queries (256d)...")
    query_vecs = embed_batch(QUERIES, dims=256)
    
    # Retrieval test
    for i, (q, qv) in enumerate(zip(QUERIES, query_vecs)):
        sims = cosine_sim(qv, summary_vecs)
        top_idx = np.argsort(sims)[::-1]
        
        print(f"\n{'='*60}")
        print(f"🔎 Query {i+1}: \"{q}\"")
        for rank, idx in enumerate(top_idx):
            sc = float(np.clip(sims[idx], -1, 1))
            marker = " ✅" if rank == 0 else ""
            print(f"  #{rank+1} CosSim={sc:.4f} | {SUMMARIES[idx][:120]}...{marker}")
    
    # Print injection blocks for dimtest
    print(f"\n{'='*60}")
    print("📋 INJECTION BLOCKS (for delegate_task dimtest):")
    print()
    for i, s in enumerate(SUMMARIES):
        print(f"--- Summary {i+1} ---")
        print(s)
        print()


if __name__ == "__main__":
    main()
