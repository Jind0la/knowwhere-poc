# KnowWhere v0.1 — Architecture

**Stand:** 2026-07-02 (dimtest-Erkenntnisse integriert)
**Scope:** Subconscious + Deep Recall. ~300 Zeilen Python + pgvector.
**Letzte Revision:** Subconscious-Rolle neu definiert — von Index zu primärer Wissensschicht.

## Executive Summary

KnowWhere ist der **Subconscious-Layer für KI-Agenten.** Kein Retrieval-System. Kein passiver Speicher. Sondern: kontinuierliche, proaktive Kontext-Injektion, die jeden Gedanken des Agenten einfärbt — ohne dass er sucht.

**Der dimtest hat die Architektur fundamental neu kalibriert.** Die ursprüngliche Annahme war: Subconscious = leichte Anker → Agent ruft Deep Recall auf. Die Realität: Der Agent behandelt injizierte Zusammenfassungen als **primäres Wissen.** Er sagt nicht „hier ist ein Hinweis, lass mich nachschauen" — er sagt „Nimar sagte X am Datum Y zu Thema Z, hier sind die Details." Die Zusammenfassung IST das Gedächtnis.

## Das dimtest-Ergebnis

Drei Subagents, drei Queries, je mit/ohne Subconscious-Injektion:

| Test | Ohne Injektion | Mit Injektion (80–150 Wörter) | Effekt |
|------|---------------|-------------------------------|--------|
| Moradbakhti Marketing | „Nicht gefunden" | Zitiert Datum + Nimar-Zitat + Zahlen | **Stark** |
| HomePod Streaming | „Keine Session" | Zitiert tvOS-Version + Root Cause | **Stark** |
| Leafgo PDF-Bug | „Keine Aufzeichnungen" | Zitiert CSS-Specificity + Quill.js | **Stark** |

**3/3 — die Grundprämisse ist validiert.** Aber mit entscheidender Einschränkung:

- ✅ 80–150-Wort-Zusammenfassungen wirken als primäre Wissensquelle
- ❌ 100-Zeichen-Snippets wirken NICHT (aus honcho-dimtest bekannt)
- ⚠️ Der Agent greift kaum auf Deep Recall zurück — die Injektion ist bereits ausreichend

**Implikation: Der Subconscious ist kein Index, sondern die primäre Gedächtnisschicht. Deep Recall ist Fallback, nicht Primärpfad.**

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      KNOWWHERE v0.1                              │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐ │
│  │ INGEST   │   │ SUMMARIZE    │   │ INJECT   │   │ RECALL   │ │
│  │          │   │              │   │          │   │          │ │
│  │ Raw →    │──→│ Instant      │──→│ Session  │←──│ Anchor → │ │
│  │ Store    │   │ (<1s, naive) │   │ Start:   │   │ Full     │ │
│  │ → Embed  │   │              │   │ Hot(500c) │   │ Text     │ │
│  │          │   │ Full         │   │ +Warm(200c│   │          │ │
│  │          │   │ (LLM, async, │   │ ) ≈3000c  │   │          │ │
│  │          │   │  <5min)      │   │           │   │          │ │
│  └────┬─────┘   └──────┬───────┘   └─────┬─────┘   └────┬─────┘ │
│       │                │                 │               │       │
│       ▼                ▼                 ▼               ▼       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    STORAGE LAYER                            │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │  │
│  │  │ Source Store  │  │ Vector Index │  │ Summary Store     │ │  │
│  │  │ (immutable)   │  │ (pgvector,   │  │ (pgvector, 256d)  │ │  │
│  │  │               │  │  256d HNSW)  │  │                   │ │  │
│  │  │ - full_text   │  │              │  │ - summary (500c)  │ │  │
│  │  │ - anchor_id   │  │ - embedding  │  │ - anchor_id (FK)  │ │  │
│  │  │ - metadata    │  │ - preview    │  │ - ucb_score       │ │  │
│  │  │               │  │ - metadata   │  │ - debut_seen      │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Vier Prozesse — drei Stores:**

| Prozess | Wann | Was |
|---------|------|-----|
| **Ingest** | Nach jedem Turn | Rohdaten speichern, embedden, Noise filtern |
| **Summarize** | Async nach Session | Instant-Summary (<1s) → Full Summary (LLM, <5min) |
| **Inject** | Session-Start + periodisch | Relevante Summaries in Context laden (Hot/Warm-Tiers) |
| **Recall** | On-Demand | Anchor → Volltext aus Source Store |

| Store | Rolle |
|-------|-------|
| **Source Store** | Unveränderliche Wahrheit. Original-Text jeder Interaktion. |
| **Vector Index** | Embedding-basierte Suche über Source Chunks (256d). Dient der Summary-Generierung und Deep Recall. |
| **Summary Store** | Angereicherte, selbsttragende Zusammenfassungen. DAS ist der Subconscious. 300–500 Zeichen, anchored. |

## Der entscheidende Paradigmenwechsel

**Vor dimtest:** Vector Index → Subconscious Injection (200-Zeichen-Pointer) → Agent ruft Deep Recall auf
**Nach dimtest:** Summary Store → Subconscious Injection (500-Zeichen-Zusammenfassung) → Agent antwortet direkt

```
ALT:  [200c Preview] → Agent: "Hmm, lass mich nachschauen" → kw_recall(anchor_id) → Volltext
NEU:  [500c Summary] → Agent: "Ich weiß das — Nimar sagte X am Y zu Z" → direkte Antwort
```

Deep Recall existiert weiterhin für:
- Verbatim-Zitate („Wortlaut der Aussage?")
- Kontext-Fenster („Was kam davor/danach?")
- Zusammenfassung reicht nicht für die Frage

## Dimension Decision (unverändert)

| Dimension | Use Case | Rationale |
|-----------|----------|-----------|
| **256d** | Summary Store + Vector Index (pgvector) | 70.8% Overlap mit 768d, 67% Speicherreduktion |
| **768d** | Deep Recall / exakter Abruf | Keine Qualitätsverluste |

**Evidence:** dimtest.py auf gefiltertem Korpus (280 Chunks, -29.8% Noise): 64d = 50.8% Overlap ❌, 256d = 70.8% Overlap ✅.

## Data Model

### Source Store (unverändert)

```sql
CREATE TABLE sources (
    id          SERIAL PRIMARY KEY,
    anchor_id   UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    full_text   TEXT NOT NULL,
    source      VARCHAR(255),        -- e.g. "2026-07-02.md"
    content_type VARCHAR(64),        -- "session", "discovery", "error_fix", "insight"
    created_at  TIMESTAMPTZ DEFAULT now(),
    metadata    JSONB DEFAULT '{}'
);
```

### Vector Index (angepasst)

```sql
CREATE TABLE vectors (
    id              SERIAL PRIMARY KEY,
    source_id       INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    embedding       VECTOR(256),           -- pgvector HNSW index
    content_preview VARCHAR(500),           -- für Deep Recall Identification (war 200)
    source_date     DATE,
    access_count    INTEGER DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    metadata        JSONB DEFAULT '{}'
);

CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

### Summary Store (NEU — die primäre Subconscious-Schicht)

```sql
CREATE TABLE summaries (
    id              SERIAL PRIMARY KEY,
    anchor_id       UUID UNIQUE NOT NULL,   -- FK zu sources.anchor_id
    summary_text    VARCHAR(500) NOT NULL,   -- die Zusammenfassung (300–500 Zeichen)
    embedding       VECTOR(256),            -- embed(summary_text)[:256]
    source_ids      INTEGER[] NOT NULL,      -- referenzierte sources.id(s) (1+)
    source_date     DATE,
    ucb_score       REAL DEFAULT 2.0,        -- Debut-Boost für neue Summaries
    access_count    INTEGER DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    debut_seen      BOOLEAN DEFAULT FALSE,   -- wurde diese Summary je injiziert?
    summary_type    VARCHAR(32) DEFAULT 'full', -- "instant" oder "full"
    superseded_by   INTEGER REFERENCES summaries(id), -- bei Updates: zeigt auf neuere Version
    created_at      TIMESTAMPTZ DEFAULT now(),
    metadata        JSONB DEFAULT '{}'
);

CREATE INDEX ON summaries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
CREATE INDEX ON summaries (debut_seen, source_date);
CREATE INDEX ON summaries (superseded_by) WHERE superseded_by IS NOT NULL;
```

**Design-Entscheidungen Summary Store:**
- `summary_text VARCHAR(500)` — 300–500 Zeichen. Genug Kontext für den Agenten, kompakt genug für's Context-Budget.
- `source_ids INTEGER[]` — eine Summary kann mehrere Source Chunks abdecken (z.B. Session-Zusammenfassung)
- `debut_seen BOOLEAN` — löst das Debut-Problem: neue Summaries werden garantiert mindestens einmal injiziert
- `superseded_by` — bei Updates wird die alte Summary auf die neue verlinkt, nicht gelöscht
- `summary_type` — „instant" (naive Extraktion, <1s) oder „full" (LLM, async). „instant" wird durch „full" ersetzt sobald verfügbar

### Noise Filter (Pre-Ingest, unverändert)

Siehe `filter_chunks.py`. Gefilterte Kategorien: cron-activity, skills-touched-generic, durable-facts, sense-inventory, system-health, naechste-session-short.

## Processes

### 1. Ingest

```python
def ingest(raw_text: str, source: str, content_type: str = "session") -> str | None:
    """Store raw text, generate anchor, embed, index. Returns anchor_id or None if noise."""
    # 1. Filter: skip noise chunks
    if filter_chunks.is_noise(raw_text)[0]:
        return None
    
    # 2. Store in Source Store → get anchor_id
    anchor_id = source_store.insert(full_text=raw_text, source=source, 
                                     content_type=content_type)
    
    # 3. Embed (nomic-embed-text via Ollama, 768d → truncate to 256d)
    embedding = embed(raw_text)[:256]
    
    # 4. Index in Vector Store with extended preview (500 chars for Deep Recall)
    vector_store.insert(
        source_id=source_id,
        embedding=embedding,
        content_preview=raw_text[:500],  # war 200
        source_date=parse_date(source),
    )
    
    return anchor_id
```

### 2. Summarize (NEU — ersetzt den alten Subconscious-Prozess)

```python
def summarize_session(anchor_ids: list[str]) -> list[str]:
    """Generate summaries for a finished session. Returns summary anchor_ids."""
    summary_ids = []
    
    for anchor_id in anchor_ids:
        # Instant summary: naive extraction (first + last 100 chars, no LLM)
        source = source_store.get_by_anchor(anchor_id)
        instant_summary = (
            f"[KnowWhere|aid={anchor_id}|type=instant] "
            f"{source.full_text[:250]}... {source.full_text[-250:]}"
        )[:500]
        
        summary_id = summary_store.insert(
            anchor_id=anchor_id,
            summary_text=instant_summary,
            embedding=embed(instant_summary)[:256],
            source_ids=[source.id],
            source_date=source.created_at.date(),
            summary_type="instant",
            debut_seen=False,
            ucb_score=2.0  # Debut-Boost
        )
        summary_ids.append(summary_id)
    
    # Async: trigger full LLM summarization (background, <5min)
    # When done → replaces instant with full, updates embedding
    trigger_llm_summarization(anchor_ids)
    
    return summary_ids


def trigger_llm_summarization(anchor_ids: list[str]):
    """Async: generate rich LLM summaries, replace instant versions."""
    # Runs in background via cron or delegate_task
    for anchor_id in anchor_ids:
        source = source_store.get_by_anchor(anchor_id)
        # Use cheapest model (deepseek-v4-flash) for summarization
        full_summary = llm_summarize(source.full_text)  # max 500 chars
        
        summary_store.upsert(
            anchor_id=anchor_id,
            summary_text=full_summary,
            embedding=embed(full_summary)[:256],
            summary_type="full",
            debut_seen=False,
            # Mark old instant summary as superseded
            supersede_previous=True
        )
```

**Warum Two-Tier Summarization?**
- **Instant**: Ermöglicht Subconscious-Injection auch bei Back-to-Back-Sessions. Naiv, aber besser als nichts.
- **Full**: LLM-generiert, kontextreich, innerhalb von Minuten. Überschreibt instant.

### 3. Inject (NEU — Session-Start + periodic)

```python
def inject_subconscious(session_context: str, max_chars: int = 3000) -> str:
    """Build the Subconscious injection block for session start.
    
    Tiered approach:
    - Hot (2-3 summaries, 500 char each): highly relevant, rich context
    - Warm (5-7 summaries, 200 char each): medium relevance, trigger recall
    - Debut: all unseen summaries (guaranteed first injection)
    
    Total budget: ~3000 chars.
    """
    query_vec = embed(session_context)[:256]
    
    # 1. Debut injection: summaries that have never been seen
    debuts = summary_store.get_debuts(limit=5)
    for d in debuts:
        summary_store.mark_seen(d.id)
    
    # 2. UCB-weighted retrieval for hot + warm
    hot_candidates = summary_store.search_ucb(
        query_vec, top_n=10, ucb_c=0.5,
        exclude_ids=[d.id for d in debuts]
    )
    
    # 3. Build tiered injection
    hot = hot_candidates[:3]    # Top-3: full 500-char summary
    warm = hot_candidates[3:10] # Next 7: truncated to 200 chars
    
    # 4. Assemble — debuts first, then hot, then warm
    blocks = []
    
    # Debuts (full length, marked as new)
    for d in debuts:
        blocks.append(f"[KnowWhere|NEW|aid={d.anchor_id}|{d.source_date}] {d.summary_text}")
    
    # Hot summaries (full 500 chars)
    for h in hot:
        blocks.append(f"[KnowWhere|aid={h.anchor_id}|{h.source_date}] {h.summary_text}")
    
    # Warm summaries (truncated to 200 chars)
    for w in warm:
        blocks.append(f"[KnowWhere|aid={w.anchor_id}|{w.source_date}] {w.summary_text[:200]}...")
    
    # 5. Budget enforcement: truncate if needed
    injection = "\n\n".join(blocks)
    if len(injection) > max_chars:
        # Drop warm summaries until budget fits
        while len(injection) > max_chars and len(blocks) > len(debuts) + len(hot):
            blocks.pop()
            injection = "\n\n".join(blocks)
    
    # 6. Update access counters
    summary_store.record_access([s.id for s in hot + warm])
    
    return injection
```

**Warum Session-Start statt jeder Turn?**
- Der dimtest zeigt: Summaries von 80–150 Wörtern wirken als primäres Wissen. Jeden Turn 3000 Zeichen zu injecten würde das Context-Budget sprengen.
- Session-Start-Injektion (~3000 chars) + Deep Recall on-demand während der Session ist effizienter.
- Periodischer Refresh alle N Turns (konfigurierbar, default N=10) als Kompromiss.

**Warum Hot/Warm-Tiers?**
- Hot (500c): Genug Kontext, dass der Agent direkt antworten kann. Für hochrelevante Themen.
- Warm (200c): Genug, um Deep Recall zu triggern wenn relevant. Spart Budget.
- Debut: Neue Summaries MÜSSEN beim ersten Session-Start injiziert werden — UCB allein findet nie-gesehene Summaries nicht.

**Format der Injektion:**
```
[KnowWhere|aid=abc123|2026-06-30] Moradbakhti ist gut aufgestellt für den Start. 
Nimar sagte am 30.06.: „Website + Kaltakquise starten in 2 Wochen." Client: Bäckerei 
Sundermann. Ziel: 10–15 KMU-Kunden in 6 Monaten. Website fast fertig, Kaltaquise-Texte 
in Arbeit.

[KnowWhere|aid=def456|2026-07-01] Leafgo PDF-Generierung: CSS-Bug in der Druckvorschau. 
Root Cause: Quill.js injiziert inline styles mit hoher Specificity, die das Print-CSS 
überschreiben. Fix: `!important` auf @media print-Regeln plus Quill-Theme-Override.
```

Das Format ist:
- Selbsttragend: kein externer Kontext nötig
- Anchored: `aid=` erlaubt Deep Recall für Details
- Datiert: `source_date` für zeitliche Einordnung
- Kontextreich: Namen, Zahlen, Zitate, Root Causes — alles was der Agent braucht

### 4. Deep Recall (angepasst)

```python
def deep_recall(anchor_id: str, context_window: int = 5) -> dict:
    """Retrieve full source text + surrounding context by anchor.
    
    Called when:
    - Agent needs verbatim quotes
    - Summary doesn't have enough context
    - Agent wants to see what came before/after
    """
    source = source_store.get_by_anchor(anchor_id)
    if not source:
        return {"error": f"Anchor {anchor_id} not found"}
    
    neighbors = source_store.get_neighbors(
        source.id, 
        before=context_window, 
        after=context_window
    )
    
    return {
        "anchor": anchor_id,
        "full_text": source.full_text,
        "source": source.source,
        "date": source.created_at.isoformat(),
        "context_before": [n.full_text for n in neighbors["before"]],
        "context_after": [n.full_text for n in neighbors["after"]],
    }
```

## Configuration

```yaml
# knowwhere_config.yaml
knowwhere:
  embedding:
    model: "nomic-embed-text"       # Ollama model
    full_dim: 768                   
    index_dim: 256                  # Summary + Vector Index dimension
    batch_size: 25                  # Ollama /api/embed batch limit
    max_chars: 1500                 # Truncate chunks to token limit
    
  ingest:
    content_preview_chars: 500      # Preview für Vector Index (war 200)
    noise_filter:
      enabled: true
      rules:
        - cron-activity
        - skills-touched-generic
        - durable-facts
        - sense-inventory
        - system-health
        - naechste-session-short
    
  summarize:
    instant_enabled: true            # Naive summary for immediate availability
    full_enabled: true               # LLM summary (async, replaces instant)
    full_model: "deepseek-v4-flash"  # Cheapest model for summarization
    max_summary_chars: 500
    full_delay_seconds: 300          # Max wait for full summary (5 min)
    
  inject:
    timing: "session_start"          # "session_start" | "every_n_turns"
    refresh_every_n_turns: 10        # Only for "every_n_turns" mode
    hot_count: 3                     # Full 500-char summaries
    warm_count: 7                    # Truncated 200-char summaries
    max_total_chars: 3000            # Total injection budget
    debut_limit: 5                   # Max debut summaries per injection
    ucb_c: 0.5                       # Exploration coefficient
    
  deep_recall:
    context_window: 5
    
  storage:
    source_store: "postgresql"
    connection: "${KNOWWHERE_DB_URL}"
```

## Hermes Integration (Memory Provider Plugin)

```python
# ~/.hermes/plugins/knowwhere/plugin.py
from hermes.memory.providers import MemoryProvider

class KnowWhereProvider(MemoryProvider):
    """Subconscious + Deep Recall memory provider."""
    
    async def prefetch(self) -> str:
        """Called at session init. Returns subconscious injection block."""
        session_context = self.get_recent_context()  # last N messages or topic hint
        return inject_subconscious(session_context)
    
    async def sync_turn(self, messages: list[dict]):
        """Called after each turn. Ingests new interactions."""
        for msg in messages:
            if msg["role"] in ("user", "assistant"):
                anchor_id = ingest(msg["content"], source=self.session_id)
                # Queue for summarization (batched at session end)
                self._pending_summaries.append(anchor_id)
    
    async def on_session_end(self):
        """Called when session closes. Triggers summarization."""
        if self._pending_summaries:
            summarize_session(self._pending_summaries)
    
    def get_tool_schemas(self) -> list[dict]:
        """Expose Deep Recall as a tool for the agent."""
        return [{
            "name": "kw_recall",
            "description": "Deep Recall: retrieve full source text by KnowWhere anchor ID.",
            "parameters": {
                "anchor_id": {"type": "string", "description": "KnowWhere anchor ID from injection block"},
                "context_window": {"type": "integer", "default": 5}
            }
        }]
```

## Evidence Base

| Quelle | Was | Status |
|--------|-----|--------|
| `dimtest_256d_results.json` | 256d = 70.8% Overlap vs. 768d auf gefiltertem Korpus | ✅ Go-Entscheidung |
| `dimtest_subconscious_effect.json` | 3/3 Queries: Subconscious-Injektion (80–150 Wörter) wirkt als primäres Wissen | ✅ Architektur-Validierung |
| `poc.py` | 397 Chunks, nomic-embed-text stabil via Ollama Batch | ✅ Technische Machbarkeit |
| `filter_chunks.py` | 29.8% Noise entfernt, 280 Chunks übrig | ✅ Pre-Ingest-Filter |
| Honcho Code-Audit | User-Model vs. Content-Similarity Gap bestätigt | ✅ Komplementär, nicht konkurrierend |

## v0.1 vs. v0.2+ Scope

| Feature | v0.1 (Jetzt) | v0.2+ |
|---------|-------------|-------|
| Subconscious Injection | ✅ Session-Start, Hot/Warm-Tiers, Debut | Cluster-basiert |
| Deep Recall | ✅ Anchor → Full Text | Mit Entity-Linking |
| Summarization | ✅ Instant + Full (LLM async) | Multi-Source Fusion |
| Noise Filter | ✅ Regex-basiert | ML-Klassifikator |
| Dream Mode | ❌ | ✅ Pattern Separation, Decay |
| Cluster Graph | ❌ | ✅ Emergente Hierarchie |
| Multimodales Scoring | ❌ | ✅ 0.5/0.3/0.2 Gewichtung |
| ColBERT Reranking | ❌ | ✅ Precision Boost |

## Next Steps

1. **Summary Store + Schema**: Railway DB mit summaries-Tabelle → `knowwhere-schema.sql`
2. **Summarization Pipeline**: `instant_summary()` + `trigger_llm_summarization()` → `summarize.py`
3. **Inject Engine**: `inject_subconscious()` mit Hot/Warm/Debut-Tiers → `inject.py`
4. **knowwhere-provider.py**: Hermes Memory Provider Plugin mit prefetch/sync_turn/on_session_end
5. **Validation Gate**: 10+ diverse Queries, messen ob Agent Summaries als primäres Wissen nutzt
