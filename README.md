# KnowWhere — Subconscious Memory for AI Agents

**Your agent shouldn't have amnesia every session.** KnowWhere gives Hermes (and any LLM agent) a persistent subconscious — it automatically remembers what you worked on, injects relevant context into every conversation, and surfaces past decisions before you repeat them.

```
pip install knowwhere && knowwhere init
```

→ 2 minutes later, your agent remembers across sessions. No manual search. No "let me check what we discussed last time."

---

## What Problem Does This Solve?

LLM agents have no memory between sessions. Each new conversation starts from zero — the agent doesn't know what you built yesterday, what bugs you fixed, or what decisions you made. You waste time re-explaining context.

**Existing solutions get it wrong:**
- **RAG / vector databases**: The agent has to actively search. That's conscious recall. Slow, misses things, requires you to know what to search for.
- **Summarization tools**: Summarize the chat after it ends. But the agent never SEES the summary unless you ask.
- **Manual context copy-paste**: You paste last session's summary. Breaks flow, feels like a workaround.

**KnowWhere is different**: It's subconscious injection. Before the agent responds, KnowWhere silently injects the most relevant memories from past sessions into the context window. The agent doesn't "search" — it just *knows*. Like human intuition.

---

## How It Works (30 seconds)

```
Every session:
  1. You type: "Let's continue the KnowWhere dashboard"
  2. KnowWhere embeds your query → searches 62 past session summaries
  3. Top 3 most relevant summaries are injected into the context
  4. Agent responds: "Last session we fixed the UCB debut bypass in search_ucb(). 
     The next step was the health dashboard. Want me to start there?"
```

The agent didn't search. It didn't call a tool. The memory was already there — injected before it thought.

---

## Architecture

KnowWhere is a **five-process pipeline** that runs alongside any LLM agent:

| Process | What it does | When |
|---------|-------------|------|
| **Ingest** | Captures every interaction, filters noise, stores raw source | Every turn |
| **Summarize** | Distills sessions into self-contained 250-500 char summaries | After session end (instant) → async LLM refinement |
| **Embed** | Converts summaries to 256d vectors via Matryoshka truncation | After summarization |
| **Inject** | Searches pgvector with UCB-weighted cosine similarity → injects top-N | Before first agent response |
| **Dream** (v0.2+) | Periodic pattern separation, cluster updates, decay | Nightly |

**Three storage layers, one truth:**
- **Source Store** (immutable): Original conversation text. The ground truth.
- **Summary Store** (pgvector): 256d embeddings + self-contained summaries. What gets injected.
- **Cluster Graph** (v0.2+): Emergent hierarchy from embedding geometry. Fractal zoom.

**Why this architecture works:**
- The agent treats injected summaries as **primary knowledge**, not search results. 5.6× faster decisions, 4.5× fewer tool calls (validated via A/B testing).
- Summaries are **self-contained**: the agent can act on them without fetching the original text.
- **Deep Recall** exists as a fallback: `kw_recall(session_id)` fetches verbatim source ± context window.

---

## Quickstart

### Prerequisites
- Python 3.11+
- PostgreSQL 16+ with pgvector extension (or let `knowwhere init` set it up)
- A Hermes Agent installation (for the agent plugin)

### 3 Commands to Running

```bash
# 1. Install
pip install knowwhere

# 2. Initialize (creates config, sets up DB, registers Hermes plugin)
knowwhere init

# 3. Verify
knowwhere health
# → KnowWhere v0.7.0 | DB: connected (62 summaries) | Embeddings: ollama/nomic-embed-text | Plugin: active
```

That's it. Your next Hermes session will have subconscious injection.

### What `knowwhere init` does

1. Creates `~/.knowwhere/config.toml` (database URL, embedding provider, LLM provider)
2. Runs `CREATE EXTENSION IF NOT EXISTS vector; CREATE TABLE IF NOT EXISTS summaries (...);`
3. Registers the Hermes plugin: symlinks `hermes-plugin/knowwhere` → `~/.hermes/plugins/knowwhere`
4. Installs the nightly cron job (summarize + embed pipeline)
5. Offers to install sentence-transformers for offline embeddings (no Ollama needed)

---

## Configuration

`~/.knowwhere/config.toml`:

```toml
[storage]
# PostgreSQL connection. SQLite coming in v0.8 for zero-dependency mode.
url = "postgresql://user:pass@localhost:5432/knowwhere"

[embedding]
# One of: "ollama", "openai", "local"
provider = "ollama"
# For ollama: model name on localhost:11434
ollama_model = "nomic-embed-text"
# For openai: model name + API key
# openai_model = "text-embedding-3-small"
# For local: uses sentence-transformers (pip install knowwhere[embeddings])
# local_model = "all-MiniLM-L6-v2"

[summarization]
# LLM for async full summaries (instant summaries are rule-based)
provider = "deepseek"  # or "openai", "ollama"
api_key = "${DEEPSEEK_API_KEY}"

[injection]
# Maximum characters injected per session
max_chars = 3000
# Minimum cosine similarity for a summary to be injected
min_score = 0.30
# UCB exploration constant (higher = more exploration of unproven memories)
ucb_c = 1.5
```

---

## Providers — Mix and Match

KnowWhere is provider-agnostic. Every layer is pluggable:

| Layer | Options |
|-------|---------|
| **Storage** | PostgreSQL/pgvector (now), SQLite/sqlite-vec (v0.8) |
| **Embeddings** | Ollama (local), OpenAI, sentence-transformers (local, no server) |
| **Summarization** | DeepSeek, OpenAI, Ollama, or rule-based only |
| **Agent** | Hermes (native), OpenAI Assistants, LangChain (v0.9+) |

---

## FAQ

### Why not just use RAG / a vector database?

RAG requires the agent to actively search. That's conscious recall — it only finds what you ask for. KnowWhere injects BEFORE the agent thinks, so the agent just *knows*. It surfaces things you forgot you needed.

### Does this replace Hindsight / Honcho / other memory providers?

No — it's complementary. Hindsight models *who you are* (user preferences, style). KnowWhere remembers *what you worked on* (project context, decisions, bugs). They coexist in the same Hermes instance.

### How is this different from ChatGPT's memory?

ChatGPT's memory is explicit save/recall — you tell it to remember, you ask what it knows. KnowWhere is **subconscious**: it observes every interaction, builds summaries automatically, and injects without being asked. No "do you remember..." prompts.

### What about privacy?

Everything runs locally by default. Summaries and embeddings are stored in YOUR database. No data leaves your machine unless you configure a cloud LLM for summarization. The `local` embedding mode uses `all-MiniLM-L6-v2` (~80MB download) — zero external calls.

### Does it work without Hermes?

The core pipeline (ingest → summarize → embed → inject) is a Python library. The Hermes plugin is one integration. v0.9+ will add LangChain and direct API integrations. For now: the `knowwhere` CLI works standalone for any text source.

---

## Development Status

| Feature | Status |
|---------|--------|
| Subconscious Injection (pgvector UCB search) | ✅ Production |
| Post-Session Auto-Summarization | ✅ Production |
| Noise Filtering (21% reduction) | ✅ Production |
| Debut Injection (new memories guaranteed visible) | ✅ Production |
| Deep Recall (`kw_recall` tool) | ✅ Production |
| Cross-Session Outcome Testing | ✅ 140 tests green |
| One-Command Install (`knowwhere init`) | 🚧 In Progress |
| SQLite Backend (zero-dependency mode) | 📋 v0.8 |
| Multi-User Isolation | 📋 v0.8 |
| Health Dashboard (`knowwhere doctor`) | 📋 v0.8 |
| Dream Mode (clustering, decay, pattern separation) | 📋 v0.9 |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture overview, development setup, and how to add new providers.

**Core principle**: KnowWhere is a pipeline, not a monolith. Every layer (storage, embeddings, summarization) has a defined interface. Adding a new embedding provider is ~50 lines of Python implementing the `EmbeddingProvider` protocol.

---

## License

Apache 2.0 — use it, modify it, ship it. If you build something cool with it, tell us.
