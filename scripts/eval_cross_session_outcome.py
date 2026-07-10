#!/usr/bin/env python3
"""Cross-session outcome eval — Session A fix must surface in Session B via injection.

Usage:
    python3 scripts/eval_cross_session_outcome.py [--live] [--dry-run]

Writes JSON report to scripts/outcome_eval_report.json
Exit 0 on pass, 1 on fail/blocked (with report).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kw_injection import contains_distractor, extract_outcome_facts, format_injection  # noqa: E402
from hermes_env import get_knowwhere_db_url, get_secret  # noqa: E402

FIXTURE_PREFIX = "kw_outcome_eval_"
REPORT_PATH = ROOT / "scripts" / "outcome_eval_report.json"

# Fabricated Session-A facts (never used in real sessions)
ROOT_CAUSE = "Flux-Kondensator Polaritäts-Inversion an Modul 7"
FIX_ACTION = "Jumper J14-J15 auf dem Zephyr Board Revision C2 tauschen"
MARKER = f"OUTCOME_{uuid.uuid4().hex[:8]}"

DISTRACTORS = [
    {
        "session_id": f"{FIXTURE_PREFIX}distractor_alpha_{MARKER}",
        "summary": (
            f"[KnowWhere|sid={FIXTURE_PREFIX}distractor_alpha|project=General] "
            "Alpha issue: Redis timeout on port 6379. Fix: increase maxmemory policy."
        ),
    },
    {
        "session_id": f"{FIXTURE_PREFIX}distractor_beta_{MARKER}",
        "summary": (
            f"[KnowWhere|sid={FIXTURE_PREFIX}distractor_beta|project=General] "
            "Beta issue: nginx 502 on upstream. Fix: reload systemd unit."
        ),
    },
    {
        "session_id": f"{FIXTURE_PREFIX}distractor_gamma_{MARKER}",
        "summary": (
            f"[KnowWhere|sid={FIXTURE_PREFIX}distractor_gamma|project=General] "
            "Gamma issue: webpack OOM. Fix: raise Node heap to 4096MB."
        ),
    },
]

SESSION_A = f"{FIXTURE_PREFIX}session_a_{MARKER}"
TARGET_SUMMARY = (
    f"[KnowWhere|sid={SESSION_A}|aid=PLACEHOLDER|project=KnowWhere] "
    f"Era Pet-Sync Zephyr Board ({MARKER}): Root Cause: {ROOT_CAUSE}. "
    f"Exakter Fix: {FIX_ACTION}."
)

SESSION_B_QUESTION = (
    "Era hängt beim Pet-Sync. Wir hatten das schon mal — "
    "was war die Root Cause und der exakte Fix am Zephyr Board?"
)


def _require_db() -> str:
    url = get_knowwhere_db_url()
    if not url:
        raise RuntimeError("KNOWWHERE_DB_URL not set")
    return url


def _embed_or_skip(db, text: str):
    from summary_pipeline import embed_text

    emb = embed_text(text)
    if emb is None:
        raise RuntimeError("Ollama embed unavailable — start Ollama or use --dry-run")
    return emb


def _insert_fixtures(db) -> dict:
    """Insert distractors first, then Session A target (newest debut wins)."""
    distractor_ids = []
    for d in DISTRACTORS:
        sid = d["session_id"]
        src = db.insert_source(sid, d["summary"], metadata={"eval": MARKER})
        e = _embed_or_skip(db, d["summary"])
        distractor_ids.append(
            db.upsert_summary(
                session_id=sid,
                project="General",
                summary_text=d["summary"],
                embedding=e,
                tier="warm",
                anchor_id=str(src),
            )
        )

    source_id = db.insert_source(
        SESSION_A,
        f"[user] Zephyr Pet-Sync hängt\n[assistant] {ROOT_CAUSE}. {FIX_ACTION}",
        metadata={"eval": MARKER, "type": "outcome_eval"},
    )
    summary_text = TARGET_SUMMARY.replace("PLACEHOLDER", str(source_id))
    emb = _embed_or_skip(db, summary_text)
    sum_id = db.upsert_summary(
        session_id=SESSION_A,
        project="KnowWhere",
        summary_text=summary_text,
        embedding=emb,
        tier="hot",
        anchor_id=str(source_id),
    )

    return {
        "marker": MARKER,
        "session_a": SESSION_A,
        "source_id": str(source_id),
        "summary_id": sum_id,
        "distractor_summary_ids": distractor_ids,
    }


def _build_injection_from_db(
    db,
    *,
    session_id_prefix: str | None = FIXTURE_PREFIX,
    include_debuts: bool = False,
) -> tuple[str, bool, list[str]]:
    from kw_injection import filter_guardrails, merge_relevant_and_debuts

    emb = _embed_or_skip(db, SESSION_B_QUESTION)
    production = session_id_prefix is None
    relevant = db.search_relevant(
        emb,
        top_k=5,
        min_score=0.30 if production else 0.15,
        ucb_weight=0.5,
        session_id_prefix=session_id_prefix,
        record_access=False,
    )
    debuts: list[dict] = []
    if include_debuts and production:
        debuts = [
            d for d in db.get_debuts(limit=8) if d.get("session_id") == SESSION_A
        ]
    merged = merge_relevant_and_debuts(relevant, debuts, debut_limit=2)
    clean = filter_guardrails(merged)
    injection = format_injection(clean)
    session_ids = [r.get("session_id", "") for r in clean]
    target_hit = any(r.get("session_id") == SESSION_A for r in clean)
    return injection, target_hit, session_ids


def _call_llm(system: str, user: str) -> str:
    api_key = get_secret("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    import urllib.request

    payload = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 400,
            "temperature": 0.1,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def _score_response(text: str, *, injected: bool) -> dict:
    if injected:
        required = ["J14-J15", "Modul 7", "Flux"]
        alt_ok = ["module 7", "flux-kondensator", "flux capacitor", "polarity", "polarität"]
        found = extract_outcome_facts(text, required)
        alt_found = extract_outcome_facts(text, alt_ok)
        has_distractor = contains_distractor(
            text, ["Redis timeout", "nginx 502", "webpack OOM", "6379"]
        )
        pass_ok = len(found) >= 2 and len(alt_found) >= 1 and not has_distractor
    else:
        found = extract_outcome_facts(text, [ROOT_CAUSE, FIX_ACTION, MARKER])
        has_distractor = contains_distractor(
            text, ["Redis timeout", "nginx 502", "webpack OOM"]
        )
        pass_ok = MARKER not in text and "J14-J15" not in text and len(found) == 0

    return {
        "pass": pass_ok,
        "facts_found": found,
        "has_distractor": has_distractor,
        "response_chars": len(text),
    }


def run_eval(*, live: bool) -> dict:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if live else "dry-run",
        "marker": MARKER,
        "status": "blocked",
        "baseline": {},
        "injected": {},
        "fixture_isolated": {},
        "global_corpus": {},
        "db_health_before": {},
        "db_health_after": {},
        "errors": [],
    }

    if not live:
        report["status"] = "dry-run"
        report["note"] = "Use --live with KNOWWHERE_DB_URL, Ollama, DEEPSEEK_API_KEY"
        return report

    from knowwhere_db import KnowWhereDB

    db = KnowWhereDB(db_url=_require_db())
    meta = {}
    try:
        report["db_health_before"] = db.health_check()
        meta = _insert_fixtures(db)
        report["fixtures"] = {k: v for k, v in meta.items() if k != "distractor_summary_ids"}

        inj_iso, target_iso, sids_iso = _build_injection_from_db(
            db, session_id_prefix=FIXTURE_PREFIX, include_debuts=False
        )
        inj_global, target_global, sids_global = _build_injection_from_db(
            db, session_id_prefix=None, include_debuts=True
        )
        report["fixture_isolated"] = {
            "target_in_injection": target_iso,
            "injection_chars": len(inj_iso),
            "session_ids": sids_iso,
        }
        report["global_corpus"] = {
            "target_in_injection": target_global,
            "injection_chars": len(inj_global),
            "session_ids": sids_global,
            "corpus_summaries": report["db_health_before"].get("summaries", 0),
        }

        system_base = (
            "Du bist Era, Ninars Assistent. Antworte präzise auf Deutsch. "
            "Der Kontextblock [KnowWhere ...] enthält verifizierte Session-Erinnerungen — "
            "nutze ihn wenn er zur Frage passt. Kein session_search, keine Datei-Tools."
        )
        baseline_user = SESSION_B_QUESTION
        injected_user = (
            f"{SESSION_B_QUESTION}\n\n---\n{inj_global}\n---"
            if inj_global
            else SESSION_B_QUESTION
        )

        try:
            baseline_text = _call_llm(system_base, baseline_user)
            report["baseline"] = _score_response(baseline_text, injected=False)
            report["baseline"]["sample"] = baseline_text[:280]
        except Exception as exc:
            report["errors"].append(f"baseline_llm: {exc}")

        try:
            injected_text = _call_llm(system_base, injected_user)
            report["injected"] = _score_response(injected_text, injected=True)
            report["injected"]["sample"] = injected_text[:280]
        except Exception as exc:
            report["errors"].append(f"injected_llm: {exc}")

        if report["errors"]:
            report["status"] = "blocked"
        elif (
            report["baseline"].get("pass")
            and report["injected"].get("pass")
            and report["global_corpus"].get("target_in_injection")
        ):
            report["status"] = "pass"
        else:
            report["status"] = "fail"

    finally:
        try:
            cleanup = db.cleanup_fixture_prefix(FIXTURE_PREFIX + "")
            report["cleanup"] = cleanup
            report["db_health_after"] = db.health_check()
            before = report.get("db_health_before", {})
            after = report.get("db_health_after", {})
            if before and after:
                report["db_restored"] = (
                    before.get("summaries") == after.get("summaries")
                    and before.get("sources") == after.get("sources")
                )
        except Exception as exc:
            report["errors"].append(f"cleanup: {exc}")
        db.close()

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run against live PG + LLM")
    parser.add_argument("--dry-run", action="store_true", help="Skip external services")
    args = parser.parse_args()

    live = args.live and not args.dry_run
    report = run_eval(live=live)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["status"] == "pass":
        return 0
    if report["status"] == "dry-run":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
