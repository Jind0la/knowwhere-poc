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
| `TestSummaryPipeline` | 1 | instant summary ≤500 chars, no filler |
| `TestKnowWhereDBOutcome` | 4 | anchor upsert, relevant search, recall, cleanup |
| `TestPluginHooks` | 3 | turn gate, post reset, session reset |
| `TestPluginRecallHandler` | 1 | kw_recall JSON |

---

## 3. Cross-Session Perception — `test_cross_session.py`

**Result:** 17 ✅ / 17 ✅ (0 failures)

Provenance chain, perception gate, control group, hook simulation (mocked PG).

---

## 4. Cross-Session Outcome — `scripts/eval_cross_session_outcome.py --live`

**Result:** `status: pass` (2026-07-10, bare shell `env -u DEEPSEEK_API_KEY -u KNOWWHERE_DB_URL`)

| Check | Baseline | Injected |
|---|---|---|
| Kennt Fabricated Fix | ❌ (pass) | ✅ J14-J15, Modul 7, Flux |
| Distraktoren | none | none |
| `fixture_isolated.target_in_injection` | n/a | true |
| `global_corpus.target_in_injection` | n/a | true (61-summary corpus + debut merge) |
| `db_restored` | n/a | true (61/61 summaries, 1 source) |
| Fixture cleanup | 4 summaries + 4 sources deleted | |

Report: `scripts/outcome_eval_report.json`

---

## 5. Plugin Loader Gate

Verified via fresh `PluginManager` process (2026-07-10):

- `version=4.0.0` `kind=standalone` `tools=1` `hooks=4` ✅
- `has_hook`: pre_llm_call, post_llm_call, on_session_finalize, on_session_reset ✅
- No `KNOWWHERE-DEBUG` / `/tmp` markers ✅
- Backups outside discovery root: `~/.hermes/plugin-backups/knowwhere/` ✅

---

## 6. Known Gaps

- Existing 61 nightly summaries lack `anchor_id`; provenance fills in for new hook-ingested turns only.
- Global eval competes with full corpus; target surfaces via relevance + debut merge (production params).
- Live eval reads secrets from `~/.hermes/.env` when bare shell lacks exports.

---

## 7. Prior Phase 3 Report (2026-07-02)

Original 109-test pipeline report remains valid; debut-bypass test semantics updated to match relevance-first search.
