# KnowWhere Outcome Slice — Architecture (PoC)

**Stand:** 2026-07-10
**Branch:** `feat/subconscious-outcome-loop`
**Scope:** Hermes hook plugin + pgvector pipeline in this repo. Rust product = later, not here.

## Outcome (Session B ohne Suche)

Session A speichert eine markante Entscheidung/Fehlerlösung. Session B soll sie **ohne `session_search`, ohne File-Read und ohne expliziten Recall-Befehl** nutzen — durch subconscious Injection in `pre_llm_call`.

## PoC-now vs. Rust-product-later

| | PoC (dieses Repo) | Rust-Produkt ( später ) |
|---|---|---|
| Integration | Hermes hooks + symlinked plugin | Native agent runtime |
| Storage | PostgreSQL/pgvector (Railway) | TBD |
| Injection | Ephemeral `{ "context": "..." }` per user turn | Same semantics, hardened |
| Deep recall | `kw_recall` tool | Same API surface |

## Vertical Slice (implementiert)

```
pre_llm_call (every user turn, fresh query)
    → Ollama embed → search_relevant (NO debut bypass)
    → optional debuts (first turn only, limited)
    → guardrail filter → format ≤3000 chars → inject

post_llm_call (nonblocking thread, fresh DB conn)
    → insert_source → instant summary 300–500c → embed → upsert_summary(anchor_id)

on_session_finalize (async)
    → DeepSeek full summary replaces instant (fallback: instant stays)

on_session_reset
    → clear session caches + pending turns

kw_recall
    → recall_deep(session_id | anchor_id) → original full_text
```

## Repo-Layout

| Path | Rolle |
|---|---|
| `hermes-plugin/knowwhere/` | Versioniertes Hermes-Plugin (`kind: standalone`) |
| `knowwhere_db.py` | PG/pgvector CRUD, `search_relevant`, `recall_deep`, fixture cleanup |
| `summary_pipeline.py` | Instant/full summaries, Ollama embed |
| `kw_injection.py` | Pure injection formatting/filtering |
| `scripts/install_plugin.sh` | Symlink → `~/.hermes/plugins/knowwhere` |
| `scripts/eval_cross_session_outcome.py` | Live outcome harness + JSON report |

## Installation

```bash
cd ~/Dev/knowwhere-poc
./scripts/install_plugin.sh
hermes plugins enable knowwhere   # idempotent
# Gateway restart + neue Session nötig
# memory.provider bleibt leer — Hindsight koexistiert
```

## Bekannte Grenzen

- Live-PG: 61 Summaries, 61/61 Embeddings, 0 NULL (Stand Verifikation 2026-07-10).
- `anchor_id` auf bestehenden Nightly-Summaries meist NULL — Hook-Pipeline füllt das für neue Turns.
- Debut-Exploration nur am **ersten Turn**, getrennt von Relevanz-Suche.
- Outcome-Eval nutzt fixture-prefix-isolierte Suche; Produktion sucht global.
- DeepSeek/Ollama/DB müssen für `--live` Eval erreichbar sein.

## Tests (2026-07-10)

| Command | Result |
|---|---|
| `python3 test_pipeline.py` | 109/109 OK |
| `python3 test_outcome_loop.py` | 14/14 OK |
| `python3 scripts/eval_cross_session_outcome.py --live` | status: pass (baseline ohne Fix-Fakten; injected mit J14-J15/Modul 7/Flux) |

## dimtest-Erkenntnis (unverändert gültig)

80–500 Zeichen selbsttragende Summaries wirken als primäres Wissen. Deep Recall bleibt Fallback für Verbatim/Kontext.
