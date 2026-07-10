#!/usr/bin/env python3.11
"""Noise-Filter für Daily-Log-Chunks — entfernt Watchdog/Cron/SENSE/System-Health vor Embedding.

Usage:
    python3 filter_chunks.py              # Analyse: zeigt Statistiken, filtert nicht
    python3 filter_chunks.py --apply      # Erzeugt gefilterte NPZ-Caches für dimtest.py
"""

import argparse, re, sys
from pathlib import Path

import numpy as np

# Import the chunker from poc.py — same logic, consistent output
from poc import chunk_daily_logs, embed_batch, LOGS_DIR, MODEL, OLLAMA_EMBED

# ── Noise Detection ────────────────────────────────────────────────────────

def is_noise(chunk_text: str) -> tuple[bool, str]:
    """Returns (is_noise, reason)."""
    t = chunk_text.strip()

    # 1. Cron Activity — formulaic watchdog output, 97 Instanzen
    if re.match(r'^## Cron Activity\b', t):
        return True, "cron-activity"

    # 2. Durable Facts retained — reines Data-Retention-Logging, 5 Instanzen
    if re.match(r'^### Durable Facts retained to Hindsight\b', t):
        return True, "durable-facts"

    # 3. SENSE inventory — pure Zähl-Logging, 43 Instanzen
    if re.match(r'^SENSE inventory:', t):
        return True, "sense-inventory"

    # 4. System Health Reports — "All Green" Status-Blöcke
    if re.match(r'^## \d+\. System Health\b', t):
        return True, "system-health"

    # 5. Skills Touched — formelhafte Listen, filtern wenn >70% Skill-Referenzen
    if re.match(r'^## Skills Touched\b', t):
        # Entferne Header
        body = re.sub(r'^## Skills Touched\s*[-–—]\s*', '', t)
        # Pattern: `skill-name`: used/patched/created/installed (optionale Klammer + Kontext)
        skill_ref = r'`[^`]+`: (?:used|patched|created|installed|geladen|invoked|referenced|loaded)(?:\s*\([^)]*\))?(?:\s+(?:used|for|now|extensively|as)\s[^`\-–—]+)?'
        # Entferne alle Skill-Referenzen und zähle verbleibende signifikante Zeichen
        body_clean = re.sub(skill_ref, '', body)
        body_clean = re.sub(r'\s*[-–—]\s*', ' ', body_clean)
        body_clean = re.sub(r'\s+', ' ', body_clean).strip()
        # Wenn nach Entfernung aller Skill-Refs < 100 signifikante Zeichen übrig → Noise
        if len(body_clean) < 100:
            return True, "skills-touched-generic"

    # 6. Nächste Session (pure checklist) — hat keine beschreibende Substanz
    if re.match(r'^## Nächste Session\b', t):
        # Nächste Session enthält manchmal Kontext (Projekt-Name, Deadlines)
        # Nur filtern wenn SEHR kurz (< 120 chars) → reine Checklist
        if len(t) < 120:
            return True, "naechste-session-short"

    return False, ""


def filter_chunks(chunks: list[dict], verbose: bool = True) -> tuple[list[dict], dict]:
    """Filtert Chunks und gibt (gefilterte_liste, stats) zurück."""
    kept = []
    removed = []
    stats: dict[str, int] = {}

    for c in chunks:
        noise, reason = is_noise(c["text"])
        if noise:
            removed.append(c)
            stats[reason] = stats.get(reason, 0) + 1
        else:
            kept.append(c)

    if verbose:
        total = len(chunks)
        removed_total = len(removed)
        print(f"\n📊 Filter-Statistik: {removed_total}/{total} Chunks entfernt ({removed_total/total*100:.1f}%)")
        for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"   🗑️  {reason}: {count}")
        print(f"   ✅ behalten: {len(kept)}")

    return kept, stats


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Noise-Filter für KnowWhere Chunks")
    parser.add_argument("--apply", action="store_true",
                        help="Erzeuge gefilterte NPZ-Caches (dimtest_cache_filtered_*.npz)")
    parser.add_argument("--dims", type=int, default=768, choices=[64, 256, 768],
                        help="Embedding-Dimensionen für Cache-Erstellung")
    args = parser.parse_args()

    # ── Chunks laden ──────────────────────────────────────────────────
    chunks = chunk_daily_logs(LOGS_DIR)
    ndays = len(set(c["date"] for c in chunks))
    print(f"📄 {len(chunks)} Chunks aus {ndays} Tagen geladen.", file=sys.stderr)

    # ── Filtern ────────────────────────────────────────────────────────
    kept, stats = filter_chunks(chunks)

    if not args.apply:
        print("\n💡 Nur Analyse-Modus. Für Cache-Erstellung: --apply", file=sys.stderr)
        return

    # ── Gefilterte Chunks embedden & cachen ────────────────────────────
    texts = [c["text"] for c in kept]
    print(f"\n🧮 Embedding {len(texts)} gefilterte Chunks ({args.dims}d)...", file=sys.stderr)
    vecs = embed_batch(texts)

    cache_path = Path(f"dimtest_cache_filtered_{args.dims}d.npz")
    np.savez(cache_path, vecs=vecs[:, :args.dims] if args.dims < 768 else vecs,
             chunks=np.array(kept, dtype=object))
    print(f"📦 Cache: {cache_path} ({vecs.shape[0]} vectors, {vecs.shape[1]}d)", file=sys.stderr)

    # Auch 64d speichern wenn 768d erstellt wurde
    if args.dims == 768:
        vecs_64 = vecs[:, :64]
        cache_64 = Path("dimtest_cache_filtered_64d.npz")
        np.savez(cache_64, vecs=vecs_64, chunks=np.array(kept, dtype=object))
        print(f"📦 Cache: {cache_64} ({vecs_64.shape[0]} vectors, 64d, via Matryoshka-Trunkation)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
