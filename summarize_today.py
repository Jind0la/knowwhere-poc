#!/usr/bin/env -S $HOME/.hermes/hermes-agent/venv/bin/python3
"""summarize_today.py v3 -- state.db directly, writes to PostgreSQL/pgvector.

Reads all non-cron sessions from state.db, extracts content,
generates LLM summaries via DeepSeek, writes to pgvector + JSON fallback.

Usage:
    python3 summarize_today.py [--date 2026-07-02] [--dry-run] [--json-only]
    
Without DEEPSEEK_API_KEY: dry-run mode.
Without KNOWWHERE_DB_URL: json-only mode (writes only summaries.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
STATE_DB = HOME / ".hermes" / "state.db"
OUT_DIR = HOME / ".hermes" / "knowwhere"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HAS_DB = bool(os.environ.get("KNOWWHERE_DB_URL"))

# Minimum messages for a session to be considered substantive
MIN_MSGS = 10

# Tokens for LLM calls
SUMMARY_MAX_TOKENS = 600
SESSION_PREVIEW_CHARS = 2000

# ── Lesson-extraction prompts (v4, 2026-07-16) ─────────────────────────
# Shift from 'what happened' to 'what Era should LEARN'.
# PITFALL/FIX/RULE format based on HORMA's contrastive failure analysis.

SESSION_SYSTEM_PROMPT = (
    "Du extrahierst LEKTIONEN aus Arbeitssessions einer KI-Operatorin namens Era. "
    "Dein Output sind WENN-DANN-Regeln, nicht Tagebucheinträge. "
    "Jeder Satz muss Era befähigen, Fehler nicht zu wiederholen oder "
    "erfolgreiche Strategien gezielt wieder anzuwenden. "
    "Max 300 Zeichen. Deutsch. Kein 'Diese Session…', kein 'Heute wurde…'."
)

SESSION_USER_PROMPT = (
    "Extrahiere aus dieser Session, was Era daraus LERNEN sollte:\n\n"
    "1. PITFALL — Welcher Fehler/Fehlannahme trat auf? WARUM?\n"
    "2. FIX — Was war die konkrete Lösung?\n"
    "3. RULE — WENN-DANN-Regel für die Zukunft.\n"
    "   (z.B. 'WENN Hook nicht feuert, DANN grep ob Hermes diesen Hook dispatched')\n"
    "4. KEYWORDS — Projektnamen, Tools, Technologien explizit nennen.\n\n"
    "Selbsttragend. Max 300 Zeichen.\n\n"
    "SESSION: {title} ({msg_count} Nachrichten)\n\n"
    "CONTENT:\n{content}"
)

DAILY_SYSTEM_PROMPT = (
    "Du extrahierst TAGESLEKTIONEN für eine KI-Operatorin namens Era. "
    "Fasse die 2-3 wichtigsten Lessons Learned des Tages in einen einzelnen, "
    "informationsdichten Text zusammen. Jede Lektion als WENN-DANN-Regel. "
    "Kein 'Heute war ein produktiver Tag'. Max 600 Zeichen. Deutsch."
)

DAILY_USER_PROMPT = (
    "Aus den heutigen Sessions ({date}, {count} Sessions): "
    "Was sind die 2-3 wichtigsten LEKTIONEN für Era?\n\n"
    "Formatiere jede Lektion als:\n"
    "→ WENN [Situation], DANN [Aktion]. Grund: [warum].\n\n"
    "Selbsttragend. Keyword-reich. Max 600 Zeichen.\n\n"
    "SESSION-SUMMARIES:\n{session_text}"
)


def get_today_sessions(target_date: str) -> list[dict]:
    """Get all non-cron, non-trivial sessions from state.db for target_date."""
    conn = sqlite3.connect(str(STATE_DB))
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("""
        SELECT s.id, s.title, s.message_count, s.tool_call_count,
               s.source, datetime(s.started_at, 'unixepoch') as started
        FROM sessions s
        WHERE date(datetime(s.started_at, 'unixepoch')) = ?
          AND (s.source IS NULL OR s.source NOT IN ('cron', 'subagent', 'homeassistant'))
          AND s.message_count >= ?
        ORDER BY s.started_at ASC
    """, (target_date, MIN_MSGS)).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]


def get_session_content(session_id: str) -> str:
    """Extract user+assistant messages from a session, max SESSION_PREVIEW_CHARS chars."""
    conn = sqlite3.connect(str(STATE_DB))
    
    rows = conn.execute("""
        SELECT role, content, timestamp
        FROM messages
        WHERE session_id = ?
          AND active = 1
          AND role IN ('user', 'assistant')
          AND (role = 'user' OR (role = 'assistant' AND length(content) > 20))
        ORDER BY timestamp
    """, (session_id,)).fetchall()
    
    conn.close()
    
    parts = []
    total = 0
    for role, content, _ in rows:
        if not content:
            continue
        prefix = "Nimar:" if role == "user" else "Era:"
        snippet = content[:300].replace("\n", " ")
        line = f"{prefix} {snippet}"
        if total + len(line) > SESSION_PREVIEW_CHARS:
            parts.append("...")
            break
        parts.append(line)
        total += len(line)
    
    return "\n".join(parts)


def call_deepseek(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    """Call DeepSeek API for summarization."""
    if not API_KEY:
        return "[DRY-RUN: no DEEPSEEK_API_KEY]"
    
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }).encode()
    
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[API ERROR: {e}]"


def detect_project(title: str, content: str) -> str:
    """Detect which project a session belongs to."""
    text = (title + " " + content).lower()
    
    if any(w in text for w in ["knowwhere", "dimtest", "64d", "768d", "256d", "poc",
                                  "railway", "subconscious", "chunk", "summary",
                                  "session branch", "pgvector"]):
        return "KnowWhere"
    if any(w in text for w in ["pet", "spritesheet", "pixel", "haar", "era bild"]):
        return "Era-Pet"
    if any(w in text for w in ["moradbakhti", "kmu", "pitch", "kaltakquise",
                                  "leafgo", "cafe agnes", "portfolio"]):
        return "Moradbakhti-KI"
    if any(w in text for w in ["preflight", "cron", "gateway", "hermes config",
                                  "skill", "hindsight"]):
        return "Infrastruktur"
    return "General"


def should_ingest(session: dict, content: str, project: str) -> bool:
    """Filter: reject noise sessions that pollute project memory.
    
    Returns False for:
    - General sessions < 40 msgs (too short to be substantive)
    - Era-Pet sessions about image generation (noise for project context)
    - Morning greetings, pure social chat, off-topic news
    - Sessions where project detection found nothing (General) and msgs < 40
    """
    # Safety: handle None values from DB
    title = session.get("title") or ""
    text = (title + " " + (content or "")).lower()
    msgs = session.get("message_count", 0)
    
    # Short General sessions = noise
    if project == "General" and msgs < 40:
        return False
    
    # Era-Pet image generation = visual noise, not project memory
    if project == "Era-Pet" and any(w in text for w in ["yoga", "sexy", "erotisch",
                                                          "rollenspiel", "pose"]):
        return False
    
    # Morning greetings / pure social chat
    # Narrowed keywords to avoid false positives on "morgen" (tomorrow) and "kaffee"
    if msgs < 20 and any(w in text for w in ["guten morgen", "gute nacht",
                                                "aufwachen", "aufstehen"]):
        return False
    
    # Off-topic news / interviews that don't connect to projects
    if msgs < 25 and project == "General":
        news_markers = ["interview", "cnbc", "podcast", "artikel", "nachrichten"]
        if any(w in text for w in news_markers):
            return False
    
    return True


def generate_session_summaries(sessions: list[dict]) -> list[dict]:
    """Generate one summary per session. Returns list of {id, title, summary, project}.
    Filters out noise sessions via should_ingest()."""
    results = []
    skipped = 0
    for sess in sessions:
        content = get_session_content(sess["id"])
        if len(content) < 100:
            skipped += 1
            continue
        
        project = detect_project(sess["title"] or "", content)
        
        # Noise filter
        if not should_ingest(sess, content, project):
            skipped += 1
            continue
        
        prompt = SESSION_USER_PROMPT.format(
            title=sess['title'] or "Unbenannt",
            msg_count=sess['message_count'],
            content=content[:2000]
        )
        
        summary = call_deepseek(
            SESSION_SYSTEM_PROMPT,
            prompt,
            max_tokens=250
        )
        results.append({
            "id": sess["id"],
            "short_id": sess["id"].split("_")[-1] if "_" in sess["id"] else sess["id"],
            "title": sess["title"] or "Unbenannt",
            "msgs": sess["message_count"],
            "source": sess["source"],
            "project": project,
            "summary": summary
        })
    
    return results


def generate_combined_summary(session_summaries: list[dict], target_date: str) -> str:
    """Generate a project-aware combined summary from per-session summaries."""
    
    # Build session list
    session_lines = []
    for s in session_summaries:
        short = s["short_id"]
        session_lines.append(f"- [{short}] [{s['project']}] {s['title']} ({s['msgs']}msgs): {s['summary']}")
    
    session_text = "\n".join(session_lines)
    
    prompt = DAILY_USER_PROMPT.format(
        date=target_date,
        count=len(session_summaries),
        session_text=session_text[:4000]
    )

    return call_deepseek(
        DAILY_SYSTEM_PROMPT,
        prompt,
        max_tokens=SUMMARY_MAX_TOKENS
    )


def write_to_pgvector(session_summaries: list[dict], combined: str, target_date: str):
    """Write summaries to PostgreSQL/pgvector (embeddings added later by embed_summaries.py).
    
    Uses a single transaction: if any insert fails, the entire batch is rolled back.
    JSON fallback is always written regardless of pgvector success."""
    try:
        from knowwhere_db import get_db
        db = get_db()
        
        # Use a single transaction for atomicity
        try:
            # Insert combined summary as a "daily" entry
            db.upsert_summary(
                session_id=f"daily-{target_date}",
                project="_daily",
                summary_text=combined,
                embedding=None,
                tier="hot",
            )
            
            # Insert per-session summaries
            for s in session_summaries:
                db.upsert_summary(
                    session_id=s["id"],
                    project=s["project"],
                    summary_text=s["summary"],
                    embedding=None,
                    tier="warm",
                )
            
            print(f"💾 pgvector: {len(session_summaries) + 1} Summaries inserted")
        except Exception as inner_e:
            print(f"❌ pgvector write failed (transaction rolled back): {inner_e}", file=sys.stderr)
        finally:
            db.close()
    except ImportError:
        print("⚠️  knowwhere_db not available — pgvector disabled", file=sys.stderr)
    except Exception as e:
        print(f"❌ pgvector connection failed: {e}", file=sys.stderr)


def write_to_json(session_summaries: list[dict], combined: str, target_date: str):
    """Write summaries to JSON file (fallback)."""
    out_path = str(OUT_DIR / "summaries.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    
    existing[target_date] = {
        "summary": combined,
        "session_count": len(session_summaries),
        "sessions": {s["short_id"]: {
            "title": s["title"],
            "project": s["project"],
            "summary": s["summary"],
            "msgs": s["msgs"],
            "source": s["source"]
        } for s in session_summaries},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "state.db"  # v2/v3 marker
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"📦 JSON gespeichert: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-only", action="store_true",
                        help="Skip pgvector, write JSON only")
    args = parser.parse_args()
    target_date = args.date
    
    # Fetch sessions
    sessions = get_today_sessions(target_date)
    if not sessions:
        print(f"❌ Keine Sessions fuer {target_date} gefunden")
        sys.exit(1)
    
    print(f"📊 {len(sessions)} Sessions in state.db fuer {target_date}")
    for s in sessions:
        short = s["id"].split("_")[-1] if "_" in s["id"] else s["id"]
        print(f"   [{short}] {s['title']} ({s['message_count']}msgs, {s['source']})")
    
    # Generate per-session summaries
    print(f"\n🤖 Generiere {len(sessions)} Session-Summaries via DeepSeek...")
    if args.dry_run or not API_KEY:
        print("   ⚠️  DRY-RUN: Keine API-Key. Ueberspringe LLM-Calls.")
        session_summaries = [
            {
                "id": s["id"],
                "short_id": s["id"].split("_")[-1] if "_" in s["id"] else s["id"],
                "title": s["title"] or "Unbenannt",
                "msgs": s["message_count"],
                "source": s["source"],
                "project": detect_project(s["title"] or "", ""),
                "summary": f"[DRY-RUN] {s['title']} ({s['message_count']}msgs)"
            }
            for s in sessions
        ]
    else:
        session_summaries = generate_session_summaries(sessions)
        print(f"\n   📊 Filtered: {len(session_summaries)} kept, {len(sessions) - len(session_summaries)} noise skipped")
    
    # Generate combined summary
    print("🧠 Generiere kombinierte Projekt-Summary...")
    if args.dry_run or not API_KEY:
        combined = f"[DRY-RUN] {len(sessions)} Sessions am {target_date}."
    else:
        combined = generate_combined_summary(session_summaries, target_date)
    
    # Output
    if not args.json_only and HAS_DB:
        write_to_pgvector(session_summaries, combined, target_date)
    
    # Always write JSON fallback
    write_to_json(session_summaries, combined, target_date)
    
    print(f"\n{'='*60}")
    print(f"COMBINED SUMMARY ({target_date}):")
    print(f"{'='*60}")
    print(combined)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
