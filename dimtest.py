#!/usr/bin/env python3.11
"""64d vs 768d Vergleichstest — Overlap, CosSim-Degradation, Top-N-Stabilität."""

import argparse, json, os, re, sys, time, urllib.request, urllib.error
from pathlib import Path

import numpy as np

OLLAMA_EMBED = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
LOGS_DIR = Path.home() / ".hermes" / "logs"
BATCH_SIZE = 25
MAX_CHARS = 1500


def embed_batch(texts: list[str]) -> np.ndarray:
    """Return full 768d embeddings for all texts."""
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


def embed_one(text: str) -> np.ndarray:
    """Return full 768d embedding."""
    payload = json.dumps({"model": MODEL, "input": [text]}).encode()
    req = urllib.request.Request(OLLAMA_EMBED, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return np.array(data["embeddings"][0], dtype=np.float32)


def cosine_sim(query: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-8)
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return v @ q


def top_results(vecs: np.ndarray, qv: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) for top-N results."""
    sims = cosine_sim(qv, vecs)
    top_idx = np.argsort(sims)[::-1][:top]
    return top_idx, np.clip(sims[top_idx], -1, 1)


# ── Test Queries ──────────────────────────────────────────────────────────
QUERIES = [
    # Spezifische Queries (sollten stabil sein)
    "Era pet spritesheet transparent",
    "KnowWhere Subconscious Layer Agent",
    "gbrain setup configuration",
    "Nimar HomePod Music",
    "Moradbakhti KI KMU Kaltakquise",
    # Generische Queries (Gravity-Well-Test)
    "Pitch Kunde",
    "Dashboard Mission Control",
    "Website SEO",
    # Random / Edge
    "Railway deployment",
    "Ollama embedding model",
    # Extrem generisch
    "wichtig",
    "Error fix",
]


def main():
    parser = argparse.ArgumentParser(description="N-d vs 768d Dim-Test")
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild 768d cache")
    parser.add_argument("--cache", type=str, default=None,
                        help="Use specific cache prefix (e.g. 'dimtest_cache_filtered' for filtered chunks)")
    parser.add_argument("--dims", type=int, default=64,
                        help="Comparison dimension to test against 768d (default: 64)")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    COMP_DIM = args.dims
    prefix = args.cache or "dimtest_cache"
    CACHE_768 = Path(f"{prefix}_768d.npz")
    CACHE_COMP = Path(f"{prefix}_{COMP_DIM}d.npz")

    # ── Load or build chunks ──────────────────────────────────────────
    if CACHE_768.exists() and not args.rebuild:
        print("📦 768d Cache geladen.", file=sys.stderr)
        data = np.load(CACHE_768, allow_pickle=True)
        vecs_768 = data["vecs"]
        chunks = data["chunks"].tolist()
    else:
        chunks = chunk_daily_logs(LOGS_DIR)
        texts = [c["text"] for c in chunks]
        ndays = len(set(c["date"] for c in chunks))
        print(f"📄 {len(chunks)} Chunks aus {ndays} Tagen", file=sys.stderr)
        print(f"🧮 Embedding {len(texts)} Chunks (768d)...", file=sys.stderr)
        t0 = time.time()
        vecs_768 = embed_batch(texts)
        print(f"   ✅ {time.time()-t0:.1f}s", file=sys.stderr)
        np.savez(CACHE_768, vecs=vecs_768, chunks=np.array(chunks, dtype=object))

    # ── Comparison dimension is truncation of 768d ─────────────────────
    vecs_comp = vecs_768[:, :COMP_DIM]
    # Also save the comparison cache if it doesn't exist
    if not CACHE_COMP.exists():
        np.savez(CACHE_COMP, vecs=vecs_comp, chunks=np.array(chunks, dtype=object))

    print(f"\n📊 {vecs_768.shape[1]}d vs {vecs_comp.shape[1]}d — "
          f"{len(chunks)} Chunks, Top-{args.top}\n", file=sys.stderr)

    # ── Run all queries ───────────────────────────────────────────────
    results = []
    for qi, qtext in enumerate(QUERIES):
        print(f"🔎 [{qi+1}/{len(QUERIES)}] \"{qtext}\"", file=sys.stderr)
        qv_full = embed_one(qtext)

        idx_768, sc_768 = top_results(vecs_768, qv_full, args.top)
        idx_comp, sc_comp = top_results(vecs_comp, qv_full[:COMP_DIM], args.top)

        # Overlap-Metriken
        overlap = len(set(idx_768) & set(idx_comp))
        # Jaccard
        jaccard = overlap / len(set(idx_768) | set(idx_comp))
        # Rank-Korrelation (Spearman auf gemeinsamen Indizes)
        common = list(set(idx_768) & set(idx_comp))
        if len(common) >= 2:
            r768 = {c: i for i, c in enumerate(idx_768) if c in common}
            r_comp = {c: i for i, c in enumerate(idx_comp) if c in common}
            d2 = sum((r768[c] - r_comp[c]) ** 2 for c in common)
            n = len(common)
            spearman = 1 - (6 * d2) / (n * (n**2 - 1)) if n > 1 else 1.0
        else:
            spearman = float("nan")

        # CosSim-Degradation
        # Map 768d scores to the comparison-dim indices
        sc_768_mapped = []
        for ic in idx_comp:
            pos = np.where(idx_768 == ic)[0]
            sc_768_mapped.append(sc_768[pos[0]] if len(pos) else float("nan"))
        sc_768_mapped = np.array(sc_768_mapped)
        sc_comp_arr = np.array(sc_comp)
        valid = ~np.isnan(sc_768_mapped)
        if valid.sum():
            cos_delta = (sc_768_mapped[valid] - sc_comp_arr[valid]).mean()
        else:
            cos_delta = float("nan")

        results.append({
            "query": qtext,
            "overlap": overlap,
            "overlap_pct": overlap / args.top * 100,
            "jaccard": jaccard,
            "spearman": spearman,
            "cos_delta": cos_delta,
            "top_768_texts": [chunks[i]["text"][:100] for i in idx_768],
            f"top_{COMP_DIM}d_texts": [chunks[i]["text"][:100] for i in idx_comp],
        })

        print(f"   overlap: {overlap}/{args.top} ({overlap/args.top*100:.0f}%), "
              f"Jaccard: {jaccard:.3f}, ρ: {spearman:.3f}, Δcos: {cos_delta:+.4f}",
              file=sys.stderr)

    # ── Summary ───────────────────────────────────────────────────────
    overlaps = [r["overlap"] for r in results]
    jaccards = [r["jaccard"] for r in results]
    spearmans = [r["spearman"] for r in results if not np.isnan(r["spearman"])]
    cos_deltas = [r["cos_delta"] for r in results if not np.isnan(r["cos_delta"])]

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY:  {vecs_768.shape[1]}d → {vecs_comp.shape[1]}d  (Matryoshka Truncation)")
    print(f"{'='*70}")
    print(f"  Avg Overlap:     {np.mean(overlaps):.1f}/{args.top}  ({np.mean(overlaps)/args.top*100:.1f}%)")
    print(f"  Avg Jaccard:     {np.mean(jaccards):.4f}")
    print(f"  Avg Spearman ρ:  {np.mean(spearmans):.4f}" if spearmans else "  Avg Spearman ρ:  N/A")
    print(f"  Avg ΔCosSim:     {np.mean(cos_deltas):+.4f}")
    print(f"  Size Reduction:  {(1 - COMP_DIM/768)*100:.0f}%  ({768} → {COMP_DIM})")
    print(f"{'='*70}")

    # Per-query detail
    print(f"\n{'Query':<45} {'Overlap':>7} {'Jaccard':>8} {'ρ':>7} {'ΔCos':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['query']:<45} {r['overlap']:>4}/{args.top} {r['overlap_pct']:>3.0f}%  "
              f"{r['jaccard']:>8.4f} {r['spearman']:>7.4f} {r['cos_delta']:>+7.4f}")

    # Show divergent queries
    bad = [r for r in results if r["overlap"] < args.top * 0.5]
    if bad:
        print(f"\n⚠️  Schwacher Overlap (<50%):")
        for r in bad:
            print(f"\n  {r['query']}  ({r['overlap']}/{args.top} overlap)")
            print(f"    768d: {' | '.join(r['top_768_texts'][:3])}")
            comp_key = f"top_{COMP_DIM}d_texts"
            print(f"    {COMP_DIM}d:  {' | '.join(r[comp_key][:3])}")

    # Save detailed results
    out = {
        "dims_768": 768,
        "n_chunks": len(chunks), "top_n": args.top,
        "summary": {
            "avg_overlap": float(np.mean(overlaps)),
            "avg_overlap_pct": float(np.mean(overlaps) / args.top * 100),
            "avg_jaccard": float(np.mean(jaccards)),
            "avg_spearman": float(np.mean(spearmans)) if spearmans else None,
            "avg_cos_delta": float(np.mean(cos_deltas)) if cos_deltas else None,
            "size_reduction_pct": (1 - COMP_DIM/768) * 100,
        },
        "queries": results,
    }
    out[f"dims_{COMP_DIM}"] = COMP_DIM
    outpath = Path(f"dimtest_{COMP_DIM}d_results.json")
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n📁 Detail-Ergebnisse: {outpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
