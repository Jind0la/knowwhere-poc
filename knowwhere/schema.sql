-- KnowWhere v0.1 — Summary Store Schema (PostgreSQL/pgvector)
-- Die primäre Subconscious-Schicht. Selbsttragende Summaries mit Anker.

-- Extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Source Store: Unveränderliche Wahrheit (immutable, append-only) — FIRST (referenced by summaries)
CREATE TABLE IF NOT EXISTS sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      TEXT NOT NULL,
    full_text       TEXT NOT NULL,                          -- Original-Text
    content_hash    TEXT NOT NULL UNIQUE,                   -- SHA-256 für Dedup
    char_count      INTEGER NOT NULL,
    user_id         TEXT NOT NULL DEFAULT 'default',        -- Multi-tenant isolation
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Summary Store: DAS ist der Subconscious
CREATE TABLE IF NOT EXISTS summaries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      TEXT NOT NULL,                          -- Hermes session ID
    project         TEXT NOT NULL,                          -- e.g. "knowwhere", "leafgo"
    summary_text    TEXT NOT NULL,                          -- 300-500 Zeichen, selbsttragend
    embedding       vector(256),                            -- nomic-embed-text, Matryoshka 256d
    anchor_id       UUID REFERENCES sources(id),            -- Anker zum Original-Text
    ucb_score       REAL NOT NULL DEFAULT 1.0,              -- Upper Confidence Bound
    debut_seen      BOOLEAN NOT NULL DEFAULT FALSE,         -- Debut-Injektion-Tracker
    view_count      INTEGER NOT NULL DEFAULT 0,             -- Wie oft injected
    last_injected   TIMESTAMPTZ,                            -- Letzte Injektion
    tier            TEXT NOT NULL DEFAULT 'warm',           -- 'hot' | 'warm' | 'cold'
    user_id         TEXT NOT NULL DEFAULT 'default',        -- Multi-tenant isolation
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique: one summary per session (idempotent — safe to re-run)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'uq_summaries_session'
    ) THEN
        ALTER TABLE summaries ADD CONSTRAINT uq_summaries_session UNIQUE (session_id);
    END IF;
END $$;

-- Vector Index: Embedding-basierte Suche (256d)
CREATE TABLE IF NOT EXISTS vector_index (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id       UUID REFERENCES sources(id) ON DELETE CASCADE,
    embedding       vector(256) NOT NULL,
    preview         TEXT NOT NULL,                          -- Erste 200 Zeichen
    project         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indizes
CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_sources_user ON sources(user_id);
CREATE INDEX IF NOT EXISTS idx_summaries_project ON summaries(project);
CREATE INDEX IF NOT EXISTS idx_summaries_ucb ON summaries(ucb_score DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_tier ON summaries(tier);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_sources_session ON sources(session_id);
CREATE INDEX IF NOT EXISTS idx_vector_index_source ON vector_index(source_id);

-- HNSW-Index für pgvector (schnelle Ähnlichkeitssuche)
CREATE INDEX IF NOT EXISTS idx_summaries_embedding 
    ON summaries USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_vector_index_embedding 
    ON vector_index USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 200);

-- UCB-Update-Funktion
CREATE OR REPLACE FUNCTION update_ucb()
RETURNS TRIGGER AS $$
DECLARE
    total_views INTEGER;
BEGIN
    SELECT COALESCE(SUM(view_count), 0) INTO total_views FROM summaries WHERE project = NEW.project;
    -- UCB = mean_reward + c * sqrt(ln(total_views) / (view_count + 1))
    -- c = 2.0 (exploration factor, adaptiv via Deep-Recall-Rate)
    NEW.ucb_score = 0.5 + 2.0 * SQRT(LN(GREATEST(total_views, 2)) / GREATEST(NEW.view_count, 1));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_ucb
    BEFORE UPDATE OF view_count ON summaries
    FOR EACH ROW
    EXECUTE FUNCTION update_ucb();

-- Injection-Query: Hot/Warm-Tiers + Debut
-- SELECT summary_text, anchor_id, ucb_score
-- FROM summaries 
-- WHERE project = $1 
--   AND (tier = 'hot' OR (tier = 'warm' AND view_count < 3) OR (debut_seen = FALSE))
-- ORDER BY ucb_score DESC
-- LIMIT $2;
