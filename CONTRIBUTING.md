# Contributing to KnowWhere

KnowWhere is a **five-process pipeline** for AI agent memory. This guide covers architecture, development setup, and how to add new providers.

---

## Architecture Overview

KnowWhere is NOT a monolith. Every layer has a defined interface. You can swap any component without touching the others.

```
┌─────────────────────────────────────────────────────────────────┐
│                        KNOWWHERE PIPELINE                        │
├───────────┬──────────┬──────────┬───────────┬───────────────────┤
│  INGEST   │SUMMARIZE │  EMBED   │  INJECT   │      DREAM        │
│           │          │          │           │    (v0.9+)         │
│ raw text  │ instant  │ vector   │ UCB       │ pattern sep.      │
│ → source  │ + async  │ → 256d   │ search    │ cluster update    │
│ store     │ LLM      │ pgvector │ → context │ decay             │
├───────────┴──────────┴──────────┴───────────┴───────────────────┤
│                    STORAGE BACKEND (pluggable)                   │
│         PostgreSQL/pgvector | SQLite/sqlite-vec (v0.8)          │
├─────────────────────────────────────────────────────────────────┤
│                  EMBEDDING PROVIDER (pluggable)                  │
│      Ollama | OpenAI | sentence-transformers (local)            │
├─────────────────────────────────────────────────────────────────┤
│                SUMMARIZATION PROVIDER (pluggable)                │
│          DeepSeek | OpenAI | Ollama | Rule-based                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (per session)

```
Session start:
  User: "Continue the dashboard work"
    → kw_injection.build_search_query()     # extract search terms
    → EmbeddingProvider.embed()             # vectorize query → 256d
    → StorageBackend.search_similar()       # UCB-weighted pgvector cosine search
    → kw_injection.filter_guardrails()      # remove system-prompt contamination
    → kw_injection.format_injection()       # format ≤3000 chars
    → injected into agent context           # agent now "remembers"

During session:
  Every turn:
    → post_llm_call hook fires
    → StorageBackend.insert_source()        # raw text → source store
    → SummaryPipeline.instant_summary()     # rule-based ≤500 chars
    → EmbeddingProvider.embed()             # vectorize summary
    → StorageBackend.upsert_summary()       # store with embedding

Session end:
    → SummaryPipeline.full_summary()        # async LLM summarization
    → replaces instant summary in DB
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **256d vectors** (not 768d) | Matryoshka truncation: 67% size reduction, 70.8% overlap with full 768d. Trading 3% precision for 3× speed. |
| **UCB, not Epsilon-Greedy** | Upper Confidence Bound auto-adapts: new memories forced-explored, proven ones exploited, stale ones decayed. No manual tuning. |
| **Self-contained summaries** | Agents treat injected text as primary knowledge, not search pointers. 300-500 chars is enough to act on without deep recall. |
| **Source/Summary separation** | Source store = immutable truth. Summary store = what gets injected. Index can be rebuilt from source without data loss. |
| **Plugin hooks, not MemoryProvider** | Hermes MemoryProvider slot is single-tenant. Plugin hooks (`pre_llm_call`/`post_llm_call`) coexist with Hindsight/Honcho. |

---

## Development Setup

```bash
# Clone
git clone https://github.com/Jind0la/knowwhere.git
cd knowwhere-poc  # the active Python pipeline

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,embeddings]"

# Database (choose one)
# Option A: Local PostgreSQL + pgvector
createdb knowwhere
psql knowwhere -c "CREATE EXTENSION vector;"
psql knowwhere -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# Option B: Use the schema file
psql $KNOWWHERE_DB_URL < knowwhere-schema.sql

# Run tests
python3 -m pytest test_pipeline.py test_outcome_loop.py test_cross_session.py -v
# Expect: 140 passed
```

### Running the pipeline manually

```bash
export KNOWWHERE_DB_URL="postgresql://..."
export DEEPSEEK_API_KEY="sk-..."

# Summarize today's sessions
python3 summarize_today.py --date $(date +%Y-%m-%d)

# Embed un-embedded summaries
python3 embed_summaries.py

# Test injection
python3 inject_subconscious.py --query "KnowWhere dashboard UCB" --top 3
```

---

## Provider Interfaces

### Storage Backend

To add a new storage backend (e.g., SQLite, Qdrant, Weaviate), implement:

```python
class StorageBackend(Protocol):
    def insert_source(self, session_id: str, content: str, metadata: dict) -> str: ...
    def upsert_summary(self, session_id: str, project: str, summary_text: str, embedding: np.ndarray | None) -> str: ...
    def search_similar(self, query_embedding: np.ndarray, top_k: int, min_score: float) -> list[dict]: ...
    def recall_deep(self, session_id: str | None, anchor_id: str | None) -> list[dict]: ...
    def health_check(self) -> dict: ...
```

Reference implementation: `knowwhere_db.py` (627 lines, PostgreSQL/pgvector).

### Embedding Provider

```python
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> np.ndarray | None: ...
    @property
    def dimension(self) -> int: ...
```

Reference implementations:
- Ollama: `summary_pipeline.py:embed_text()` (144 lines, nomic-embed-text 768d→256d)
- OpenAI/DeepSeek: ~50 lines (not yet implemented, but the interface is trivial)
- Local: `sentence-transformers` — coming in v0.8

### Summarization Provider

```python
class SummarizationProvider(Protocol):
    def summarize(self, messages: list[dict], max_chars: int) -> str: ...
```

Reference: `summary_pipeline.py` — instant (rule-based) + full (DeepSeek async).

---

## Testing

Three test tiers:

```bash
# Tier 1: Unit tests (no DB, no network)
python3 -m pytest test_pipeline.py -v
# 109 tests: DB CRUD, dedup, embedding validation, UCB math, guardrail filtering

# Tier 2: Integration tests (needs DB)
python3 -m pytest test_outcome_loop.py -v
# 14 tests: full pipeline with fixtures, debut injection, idempotency

# Tier 3: Cross-session outcome tests (needs DB + LLM)
python3 -m pytest test_cross_session.py -v
# 17 tests: inject decision → verify agent uses it without search tools
```

### Writing a cross-session test

```python
def test_agent_remembers_decision_without_search(db, embedding_provider):
    # 1. Simulate Session A: a decision was made
    db.upsert_summary(
        session_id="session-a-001",
        project="KnowWhere",
        summary_text="[WAS] Fixed UCB debut bypass bug in search_ucb(). [WARUM] New summaries scored below min_score. [HANDLUNG] Added OR debut_seen = FALSE to WHERE clause.",
        embedding=embedding_provider.embed("UCB debut bypass search_ucb bug fix")
    )
    
    # 2. Simulate Session B: agent queries about UCB
    query = "The UCB search isn't showing new summaries"
    embedding = embedding_provider.embed(query)
    results = db.search_similar(embedding, top_k=3, min_score=0.30)
    
    # 3. Assert: the fix explanation is injected
    assert any("debut_seen = FALSE" in r["summary_text"] for r in results)
    assert len(results) >= 1
```

---

## Project Structure

```
knowwhere-poc/
├── hermes-plugin/knowwhere/
│   └── __init__.py           # Hermes hook plugin (467 lines)
├── knowwhere_db.py           # PostgreSQL/pgvector CRUD (627 lines)
├── kw_injection.py           # Injection formatting, guardrails, query building (135 lines)
├── summary_pipeline.py       # Instant/full summaries, Ollama embeddings (144 lines)
├── summarize_today.py        # Nightly pipeline: state.db → DeepSeek → pgvector
├── embed_summaries.py        # Backfill: embed un-embedded summaries
├── inject_subconscious.py    # Manual injection for testing/preflight
├── resummarize_nightlies.py  # Quality backfill: re-process all summaries
├── hermes_env.py             # Read Hermes .env without mutating os.environ
├── test_pipeline.py          # 109 unit tests
├── test_outcome_loop.py      # 14 integration tests
├── test_cross_session.py     # 17 cross-session tests
├── scripts/
│   ├── install_plugin.sh     # Symlink plugin → ~/.hermes/plugins/knowwhere
│   └── eval_cross_session_outcome.py  # Live outcome evaluation harness
└── knowwhere-schema.sql      # PostgreSQL DDL
```

---

## Common Pitfalls

### pgvector

- **Pass numpy arrays directly, not bytes.** `register_vector(conn)` → `np.ndarray` to `cursor.execute()`. Using `.tobytes()` fails with type mismatch.
- **Dimension mismatch**: summaries are 256d, queries must be truncated to 256d. Full 768d → PostgreSQL error.
- **Commit discipline**: `psycopg2` autocommit is OFF. Every write needs explicit `conn.commit()`.
- **UUID casting**: use `id::text = ANY(%s)` — PostgreSQL sees Python UUID strings as `text`, not `uuid`.

### Hermes Plugin

- **Plugin imports from `~/Dev/knowwhere-poc/`**, not from the Rust repo. The Rust repo is dormant.
- **Cron jobs don't source `.zshrc`.** Parse `.zshrc` directly as fallback for `KNOWWHERE_DB_URL` and API keys.
- **Stale env vars** from `launchctl setenv` or `~/.hermes/.env` can block `.zshrc` fallback. Always compare env vs `.zshrc` and prefer `.zshrc` when they differ.
- **Toolset requirement**: The Hermes `knowwhere` toolset must be enabled for `kw_recall` to appear in the agent's tool list.

### Embeddings

- **Ollama batch limit**: 2048 token context window. Keep texts under 2000 chars.
- **NULL embeddings = silent failure**: Summaries written without embeddings → injection returns empty → agent has no memory. Always verify with `knowwhere health`.

---

## Roadmap for Contributors

Good first issues:

1. **SQLite/sqlite-vec backend** — Implement `StorageBackend` for SQLite. ~200 lines. Enables zero-dependency mode.
2. **OpenAI embedding provider** — `text-embedding-3-small` integration. ~50 lines. Trivial API wrapper.
3. **Health dashboard CLI** — `knowwhere health` + `knowwhere doctor`. ~200 lines. Read counts, check embedding NULLs, verify cron.
4. **Multi-user isolation** — Add `user_id` column + filter on all queries. ~100 lines. Required for SaaS deployments.

Architecture-level contributions:

5. **Dream Mode** — Nightly clustering, pattern separation, decay. The theoretical foundation is in `knowwhere-theory` skill. Start with decay (simplest) then pattern separation.
6. **Temporal decay scoring** — Hybrid `semantic * (1-w) + recency * w` scoring. Partial implementation in Rust repo (`apply_hybrid_temporal_scoring`); needs Python port.
7. **LangChain / OpenAI Assistants integration** — Make KnowWhere work without Hermes. Provide a Python SDK that wraps any agent's context window.

---

## Questions?

- Architecture deep-dive: `knowwhere-theory` skill (Hermes: `/load knowwhere-theory`)
- Operations handbook: `knowwhere-operations` skill
- Vision & design decisions: `knowwhere-vision` skill
- Open an issue: https://github.com/Jind0la/knowwhere/issues
