#!/bin/bash
# knowwhere-nightly.sh v3 — 23:00 Cron: Summarize + Embed Pipeline (pgvector)
# Source: ~/Dev/knowwhere-poc/knowwhere-nightly.sh
#
# Phase 3: Primary output path is PostgreSQL/pgvector on Railway.
# JSON/NPZ still written as fallback.
#
# Requires:
#   DEEPSEEK_API_KEY — for LLM summarization
#   KNOWWHERE_DB_URL — PostgreSQL connection string
#   Ollama running on localhost:11434 — for embeddings

set -e

VENV_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python3"
SCRIPT_DIR="$HOME/Dev/knowwhere-poc"
LOG_FILE="$HOME/.hermes/logs/knowwhere-nightly.log"

echo "=== KnowWhere Nightly Pipeline v3 $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_FILE"

# Source env vars from zshrc if not in environment
if [ -z "$DEEPSEEK_API_KEY" ]; then
    export DEEPSEEK_API_KEY=$(grep DEEPSEEK_API_KEY ~/.zshrc 2>/dev/null | head -1 | sed 's/.*=//')
fi
if [ -z "$KNOWWHERE_DB_URL" ]; then
    echo "❌ KNOWWHERE_DB_URL nicht gesetzt. Pipeline kann nicht auf pgvector schreiben." | tee -a "$LOG_FILE"
    echo "   Setze: export KNOWWHERE_DB_URL=\"postgresql://postgres:PASSWORD@hayabusa.proxy.rlwy.net:27590/railway\"" | tee -a "$LOG_FILE"
    exit 1
fi

echo "DB_URL set: ${KNOWWHERE_DB_URL:0:30}..." | tee -a "$LOG_FILE"

# Step 1: Summarize (state.db → pgvector + JSON)
echo "[1/3] summarize_today.py v3 → pgvector..." | tee -a "$LOG_FILE"
$VENV_PYTHON "$SCRIPT_DIR/summarize_today.py" --date "$(date +%Y-%m-%d)" 2>&1 | tee -a "$LOG_FILE"

# Step 2: Embed (read un-embedded summaries from pgvector, embed via Ollama, write back)
echo "[2/3] embed_summaries.py v3 → Ollama → pgvector..." | tee -a "$LOG_FILE"
$VENV_PYTHON "$SCRIPT_DIR/embed_summaries.py" 2>&1 | tee -a "$LOG_FILE"

# Step 3: Health check
echo "[3/3] DB health check..." | tee -a "$LOG_FILE"
$VENV_PYTHON "$SCRIPT_DIR/knowwhere_db.py" --health 2>&1 | tee -a "$LOG_FILE"

echo "=== Pipeline complete $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_FILE"
