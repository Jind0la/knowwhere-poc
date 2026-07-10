#!/usr/bin/env -S $HOME/.hermes/hermes-agent/venv/bin/python3
"""resummarize_nightlies.py — Re-summarize all KnowWhere summaries with improved prompt.

Reads each summary's original session content from state.db (or existing summary
for daily entries), calls DeepSeek with a decision/action-focused prompt, re-embeds
via Ollama, and updates pgvector.

Usage:
    python3 resummarize_nightlies.py [--dry-run] [--limit N] [--skip-daily]

Cost: ~$0.32 for 63 summaries (DeepSeek chat). Embeddings are local (Ollama).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HOME = Path.home()
STATE_DB = HOME / ".hermes" / "state.db"
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"
TRUNC_DIM = 256

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DELAY_S = 0.6  # rate limit buffer


# ── Improved prompts ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Du fasst Arbeitssessions in einen einzelnen, selbsttragenden Satz zusammen. "
    "Jeder Satz muss ohne Kontext verständlich sein. Kein 'Diese Session…', "
    "kein Daily-Digest-Stil. Deutsch. Max 250 Zeichen."
)

USER_PROMPT_TEMPLATE = """Fasse diese Session in EINEM Satz zusammen. Der Satz muss enthalten:
1. WAS wurde entschieden oder erkannt
2. WARUM (Grund / Motivation)
3. WELCHE konkrete Handlung folgte (oder: was ist der nächste Schritt)

Wichtig:
- Selbsttragend: der Satz muss ohne Kontext verständlich sein
- Projektnamen und Schlüsselbegriffe explizit nennen (z.B. "KnowWhere", "pgvector", "Railway", "Moradbakhti-KI")
- Kein "Diese Session…", kein "Heute wurde…"
- Max 250 Zeichen

SESSION: {title} ({msg_count} Nachrichten, Projekt: {project})

CONTENT:
{content}"""

DAILY_PROMPT_TEMPLATE = """Fasse die heutigen Sessions ({date}) in EINEM informationsdichten Satz zusammen.
Nenne: die 1-2 wichtigsten Entscheidungen des Tages, deren Grund, und die resultierende Aktion.

SESSION-SUMMARIES:
{summaries}

Regeln:
- Selbsttragend: der Satz muss ohne Kontext verständlich sein
- Max 300 Zeichen
- Deutsch"""


# ── Helpers ────────────────────────────────────────────────────────────────

def call_deepseek(system: str, user: str, max_tokens: int = 250) -> str:
    if not API_KEY:
        return "[NO API KEY]"
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        DEEPSEEK_URL, data=payload,
        headers={"Content-Type": "application/json",
                  "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def embed_text(text: str) -> np.ndarray:
    payload = json.dumps({"model": OLLAMA_MODEL, "input": [text[:2000]]}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    emb = np.array(data["embeddings"][0], dtype=np.float32)[:TRUNC_DIM]
    return emb / (np.linalg.norm(emb) or 1.0)


def get_session_content(session_id: str, max_chars: int = 2500) -> str:
    """Extract user+assistant messages from state.db."""
    conn = sqlite3.connect(str(STATE_DB))
    rows = conn.execute("""
        SELECT role, content FROM messages
        WHERE session_id = ? AND active = 1
          AND role IN ('user', 'assistant')
          AND (role = 'user' OR length(content) > 20)
        ORDER BY timestamp
    """, (session_id,)).fetchall()
    conn.close()

    parts = []
    total = 0
    for role, content in rows:
        if not content:
            continue
        prefix = "Nimar:" if role == "user" else "Era:"
        snippet = content[:400].replace("\n", " ")
        line = f"{prefix} {snippet}"
        if total + len(line) > max_chars:
            parts.append("...")
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def get_session_meta(session_id: str) -> dict:
    """Get title + msg count from state.db."""
    conn = sqlite3.connect(str(STATE_DB))
    row = conn.execute(
        "SELECT title, message_count FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    if row:
        return {"title": row[0] or "Unbenannt", "msg_count": row[1] or 0}
    return {"title": "Unbenannt", "msg_count": 0}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change, don't write to DB")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process max N summaries (0 = all)")
    parser.add_argument("--skip-daily", action="store_true",
                        help="Skip daily combined summaries")
    args = parser.parse_args()

    # Lazy import after path setup
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from knowwhere_db import KnowWhereDB

    db = KnowWhereDB()

    # Fetch all summaries
    with db.conn.cursor() as cur:
        cur.execute("""
            SELECT id, session_id, summary_text, project
            FROM summaries
            ORDER BY created_at
        """)
        rows = cur.fetchall()

    summaries = [{"id": r[0], "session_id": r[1], "old_text": r[2], "project": r[3]}
                 for r in rows]

    is_daily = lambda s: s["session_id"].startswith("daily-")
    targets = [s for s in summaries
               if not (args.skip_daily and is_daily(s))]

    if args.limit:
        targets = targets[:args.limit]

    print(f"📋 {len(targets)} summaries to re-process "
          f"({len([t for t in targets if is_daily(t)])} daily, "
          f"{len([t for t in targets if not is_daily(t)])} session)")
    print(f"   Dry-run: {args.dry_run}")
    print(f"   API key: {'✅' if API_KEY else '❌ MISSING'}")
    print()

    updated = 0
    skipped = 0
    errors = 0

    for i, s in enumerate(targets):
        sid = s["session_id"]
        is_d = is_daily(s)
        kind = "daily" if is_d else "session"
        date_str = sid.replace("daily-", "") if is_d else sid[:8]

        try:
            if is_d:
                # Daily: re-summarize from per-session summaries of that date
                with db.conn.cursor() as cur:
                    cur.execute("""
                        SELECT summary_text FROM summaries
                        WHERE session_id NOT LIKE 'daily-%%'
                          AND session_id LIKE %s
                        ORDER BY created_at
                    """, (f"{date_str}%",))
                    sub_summaries = [r[0] for r in cur.fetchall()]
                if not sub_summaries:
                    skipped += 1
                    continue
                prompt = DAILY_PROMPT_TEMPLATE.format(
                    date=date_str,
                    summaries="\n".join(f"- {t}" for t in sub_summaries[:20]),
                )
                new_text = call_deepseek(SYSTEM_PROMPT, prompt, max_tokens=300)
            else:
                # Session: re-summarize from state.db source content
                content = get_session_content(sid)
                if len(content) < 50:
                    print(f"  ⏭️  [{i+1}/{len(targets)}] {sid[:30]} — no source content")
                    skipped += 1
                    continue

                meta = get_session_meta(sid)
                prompt = USER_PROMPT_TEMPLATE.format(
                    title=meta["title"],
                    msg_count=meta["msg_count"],
                    project=s["project"],
                    content=content,
                )
                new_text = call_deepseek(SYSTEM_PROMPT, prompt, max_tokens=250)

            if not new_text or new_text.startswith("[") and "ERROR" in new_text:
                print(f"  ❌ [{i+1}/{len(targets)}] {sid[:30]} — API error: {new_text[:80]}")
                errors += 1
                continue

            # Compare old vs new
            old = s["old_text"]
            changed = (new_text != old)
            marker = "🔄" if changed else "✓"

            if changed:
                print(f"  {marker} [{i+1}/{len(targets)}] {sid[:30]:32s} [{kind}]")
                print(f"     OLD: {old[:120]}")
                print(f"     NEW: {new_text[:120]}")
            else:
                print(f"  {marker} [{i+1}/{len(targets)}] {sid[:30]:32s} [{kind}] unchanged")

            if changed and not args.dry_run:
                # Re-embed and update
                embedding = embed_text(new_text)
                db.upsert_summary(
                    session_id=sid,
                    project=s["project"],
                    summary_text=new_text,
                    embedding=embedding,
                    tier="warm",
                )
                updated += 1
            elif not changed:
                skipped += 1
            else:
                updated += 1  # dry-run counts as would-update

            time.sleep(DELAY_S)

        except Exception as e:
            print(f"  ❌ [{i+1}/{len(targets)}] {sid[:30]} — {e}")
            errors += 1

    db.close()

    print(f"\n📊 Done: {updated} updated, {skipped} skipped, {errors} errors")
    if args.dry_run:
        print("   (dry-run — no changes written)")


if __name__ == "__main__":
    main()
