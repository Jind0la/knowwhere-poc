#!/usr/bin/env python3
"""
Phase 1 — Preflight Gap Detection
Cross-references state.db sessions against daily log entries.
Finds sessions that exist in the DB but have NO daily log entry.

Usage:
    python3 check_daily_log_gaps.py [--date YYYY-MM-DD] [--json]

Output:
    JSON with {missing: [{id, title, msgs, tools}], 
               documented: [{id, title, ...}],
               total_db: N, total_log: N}
"""

from __future__ import annotations

import sqlite3
import json
import sys
import re
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
STATE_DB = HOME / ".hermes" / "state.db"
LOGS_DIR = HOME / ".hermes" / "logs"


def get_today_sessions(target_date: str) -> list[dict]:
    """Get all non-cron sessions from state.db for target_date."""
    conn = sqlite3.connect(str(STATE_DB))
    conn.row_factory = sqlite3.Row
    
    # Get sessions that have messages on target_date
    rows = conn.execute("""
        SELECT DISTINCT s.id, s.title, s.message_count, s.tool_call_count,
               s.source, datetime(s.started_at, 'unixepoch') as started
        FROM sessions s
        JOIN messages m ON m.session_id = s.id
        WHERE date(datetime(s.started_at, 'unixepoch')) = ?
          AND (s.source IS NULL OR s.source NOT IN ('cron', 'subagent', 'homeassistant'))
          AND s.message_count > 5
        ORDER BY s.started_at ASC
    """, (target_date,)).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]


def extract_short_id(session_id: str) -> str | None:
    """Extract the short suffix from a full session ID.
    '20260702_183902_17d02e' -> '17d02e'
    '20260702_083347_cef49522' -> 'cef49522'
    """
    parts = session_id.split("_")
    if len(parts) >= 3:
        return parts[-1]
    return None


def read_daily_log(target_date: str) -> str:
    """Read the daily log file if it exists."""
    log_path = LOGS_DIR / f"{target_date}.md"
    if log_path.exists():
        return log_path.read_text()
    return ""


def strip_cron_activity(log_content: str) -> str:
    """Remove all '## Cron Activity' blocks from log content.
    These are formulaic, not proper session documentation."""
    lines = log_content.split("\n")
    result = []
    in_cron_block = False
    for line in lines:
        if line.startswith("## Cron Activity"):
            in_cron_block = True
            continue
        if in_cron_block:
            # Cron blocks end at next ## heading or empty line followed by non-Cron content
            if line.startswith("## ") and not line.startswith("## Cron Activity"):
                in_cron_block = False
                result.append(line)
            # Also end cron block on session entries or discoveries
            elif line.startswith("- **[") or line.startswith("- [") or line.startswith("## Discoveries") or line.startswith("## Skills") or line.startswith("## Nächste"):
                in_cron_block = False
                result.append(line)
            # Skip cron lines
            continue
        result.append(line)
    return "\n".join(result)


def find_missing_sessions(sessions: list[dict], log_content: str) -> tuple[list, list]:
    """Cross-reference sessions against daily log (excluding Cron Activity blocks).
    Returns (missing, documented)."""
    # Strip cron activity blocks — those are NOT proper documentation
    clean_log = strip_cron_activity(log_content)
    
    documented = []
    missing = []
    
    for sess in sessions:
        sid = sess["id"]
        short_id = extract_short_id(sid)
        
        # Only check: does the short ID appear OUTSIDE Cron Activity blocks?
        # Title matching is too fuzzy and creates false positives
        found = short_id and short_id in clean_log
        
        if found:
            documented.append(sess)
        else:
            missing.append(sess)
    
    return missing, documented


def main():
    target_date = date.today().isoformat()
    output_json = "--json" in sys.argv
    
    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            target_date = arg.split("=", 1)[1]
    
    sessions = get_today_sessions(target_date)
    log_content = read_daily_log(target_date)
    
    if not sessions:
        result = {"error": f"No sessions found for {target_date}", "total_db": 0}
        print(json.dumps(result, indent=2) if output_json else "Keine Sessions gefunden.")
        return
    
    if not log_content:
        result = {
            "error": f"No daily log for {target_date}",
            "total_db": len(sessions),
            "missing": [{"id": s["id"], "title": s.get("title", ""), 
                        "msgs": s["message_count"], "tools": s["tool_call_count"]} 
                       for s in sessions]
        }
        print(json.dumps(result, indent=2) if output_json else f"❌ Kein Daily Log. ALLE {len(sessions)} Sessions undokumentiert.")
        return
    
    missing, documented = find_missing_sessions(sessions, log_content)
    
    result = {
        "date": target_date,
        "total_db": len(sessions),
        "total_documented": len(documented),
        "total_missing": len(missing),
        "missing": [{"id": s["id"], "short_id": extract_short_id(s["id"]),
                     "title": s.get("title", ""), 
                     "msgs": s["message_count"], 
                     "tools": s["tool_call_count"]}
                    for s in missing],
        "documented": [{"id": s["id"], "short_id": extract_short_id(s["id"]),
                        "title": s.get("title", ""),
                        "msgs": s["message_count"]}
                       for s in documented]
    }
    
    if output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if missing:
            print(f"⚠️  {len(missing)}/{len(sessions)} Sessions FEHLEN im Daily Log:")
            for s in missing:
                print(f"   - {extract_short_id(s['id'])}: {s.get('title', 'N/A')} ({s['message_count']}msgs)")
        else:
            print(f"✅ Alle {len(sessions)} Sessions im Daily Log dokumentiert.")


if __name__ == "__main__":
    main()
