# KnowWhere — Test Report

**Date:** 2026-07-10
**Branch:** `feat/subconscious-outcome-loop`
**Runner:** `~/.hermes/hermes-agent/venv/bin/python3`

---

## 1. Regression — `test_pipeline.py`

**Result:** 109 ✅ / 109 ✅ (0 failures)

Changes vs. baseline:
- `search_ucb` / `search_relevant`: **no** `OR debut_seen = FALSE` bypass
- `ORDER BY weighted_score DESC` (relevance first)
- `upsert_summary` accepts optional `anchor_id`

---

## 2. Outcome Loop — `test_outcome_loop.py`

**Result:** 14 ✅ / 14 ✅ (0 failures)

| Class | # | Coverage |
|---|---|---|
| `TestInjectionHelpers` | 5 | query build, guardrails, debut merge, budget |
| `TestSummaryPipeline` | 1 | instant summary 300–500 chars |
| `TestKnowWhereDBOutcome` | 4 | anchor upsert, relevant search, recall, cleanup |
| `TestPluginHooks` | 3 | turn gate, post reset, session reset |
| `TestPluginRecallHandler` | 1 | kw_recall JSON |

---

## 3. Cross-Session Outcome — `scripts/eval_cross_session_outcome.py --live`

**Result:** `status: pass` (2026-07-10)

| Check | Baseline | Injected |
|---|---|---|
| Kennt Fabricated Fix | ❌ (pass) | ✅ J14-J15, Modul 7, Flux |
| Distraktoren | none | none |
| `target_in_injection` | n/a | true |
| Fixture cleanup | 4 summaries + 4 sources deleted | |

Report: `scripts/outcome_eval_report.json`

---

## 4. Plugin Loader Gate

- `hermes-plugin/knowwhere/plugin.yaml`: `kind: standalone` ✅
- Hooks declared: `pre_llm_call`, `post_llm_call`, `on_session_finalize`, `on_session_reset` ✅
- Tool: `kw_recall` ✅

---

## 5. Known Gaps

- Outcome eval isolates search to `kw_outcome_eval_*` fixtures; production competes with full summary corpus.
- Existing 61 nightly summaries lack `anchor_id`; provenance fills in for new hook-ingested turns only.
- Full eval requires live `KNOWWHERE_DB_URL`, Ollama, `DEEPSEEK_API_KEY`.

---

## 6. Prior Phase 3 Report (2026-07-02)

Original 109-test pipeline report remains valid; debut-bypass test semantics updated to match relevance-first search.
