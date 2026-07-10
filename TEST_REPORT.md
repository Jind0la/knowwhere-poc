# KnowWhere Phase 3 Pipeline — Test Report

**Date:** 2026-07-02  
**Test file:** `~/Dev/knowwhere-poc/test_pipeline.py`  
**Runner:** python3.11 (Hermes venv)  
**Result:** 109 ✅ / 109 ✅  (0 failures, 0 errors)

---

## 1. Test Coverage

| Test Class | # Tests | What's tested |
|---|---|---|
| `TestKnowWhereDB` | 23 | upsert_summary (6), search_ucb (8), mark_seen (3), record_access (3), get_debuts (3), insert_source (2), index_vector (2), health_check, get_db singleton, close (2) |
| `TestDetectProject` | 37 | KnowWhere (11), Era-Pet (5), Moradbakhti-KI (7), Infrastruktur (6), General fallback (2), edge cases (6) |
| `TestShouldIngest` | 23 | General threshold (3), Era-Pet noise (6), morning/social (8), news (4), non-General (1), edge cases (5) |
| `TestGetSessionContent` | 5 | Basic build, empty list, None skip, short-assistant SQL filter, truncation |
| `TestGenerateSessionSummaries` | 5 | Basic gen, short content skip, noise skip, multi-session, dedup |
| `TestGenerateCombinedSummary` | 2 | Basic, empty list |
| `TestWriteToPgvector` | 3 | Combined + per-session, empty list, error handling |
| `TestWriteToJson` | 1 | JSON fallback write |
| `TestGetTodaySessions` | 1 | Mocked sqlite3 |
| **Total** | **109** | |

---

## 2. Gefundene Bugs

### BUG 1 — Critical: `should_ingest` crashes on `None` title
**File:** `summarize_today.py`, line 151  
**Code:** `text = (session["title"] + " " + content).lower()`  
**Problem:** When `get_today_sessions()` returns a session with `title=NULL` (valid in sqlite3), the raw `None` from the DB causes `TypeError: can only concatenate str (not "NoneType") to str`.  
**Trigger:** Any session from state.db where the `title` column is NULL.  
**Fix:** Use `(session.get("title", "") or "") + " " + (content or "")`.  

### BUG 2 — Critical: `should_ingest` crashes on `None` content
**File:** `summarize_today.py`, line 151 (same line as BUG 1)  
**Problem:** Same concatenation also crashes when `content` (the function parameter) is `None`.  
**Fix:** Wrap with `(content or "")` — same fix as BUG 1.  

### BUG 3 — Dead code: `emb_bytes` in `upsert_summary`
**File:** `knowwhere_db.py`, line 66  
**Code:** `emb_bytes = embedding if isinstance(embedding, np.ndarray) else embedding.tobytes() if embedding is not None else None`  
**Problem:** The variable `emb_bytes` is computed and **never used**. The SQL query passes `embedding` (the original ndarray) directly. The ternary also has inverted logic: when `embedding` is an ndarray, it assigns the whole array (not `.tobytes()`). Since `emb_bytes` is unused, this doesn't cause runtime errors but is dead, confusing code.  
**Fix:** Remove the entire line.

### BUG 4 — Design: Morning-greeting keywords are overbroad
**File:** `summarize_today.py`, lines 164-166  
**Keywords:** `["guten morgen", "kaffee", "morgen", "gute nacht", "hallo"]`  
**Problems:**

| Keyword | False-positive scenario | Risk |
|---|---|---|
| `"kaffee"` (coffee) | A 15-msg session about "Kaffee project roadmap" → filtered as morning greeting | High — coffee is a legitimate topic |
| `"morgen"` (tomorrow/morning) | "Morgen deploy the update" (tomorrow's plan) → filtered as morning greeting | Medium — dual meaning in German |
| `"hallo"` (hello) | "Hallo, brauche Hilfe bei DB" (15-msg technical session) → filtered as greeting | High — every short conversation starts with hallo |

**Impact:** Substantial false-positive noise filtering. Short but legitimate technical sessions are silently dropped from the summary pipeline.

### BUG 5 — Design: Short assistant responses excluded from context
**File:** `summarize_today.py`, line 68  
**SQL:** `(role = 'assistant' AND length(content) > 80)`  
**Problem:** Assistant messages with ≤80 characters are excluded from the content fed to the summarizer. This silently drops short but potentially important responses:
- `"Ja, das ist richtig."` (18 chars) → lost
- `"API-Schlüssel aktualisiert."` (24 chars) → lost
- `"Erledigt."` (9 chars) → lost

Only verbose assistant responses survive, biasing summarization toward longer messages.

### BUG 6 — Design: Silent exception swallowing in `write_to_pgvector`
**File:** `summarize_today.py`, lines 286-287  
**Code:** `except Exception as e: print(f"⚠️  pgvector write failed: {e}")`  
**Problem:** If `upsert_summary` succeeds for the combined daily summary but fails mid-way through per-session summaries, pgvector is left in a partially written state. The exception is caught silently, and the function continues to JSON fallback. The caller (main()) has no way to know pgvector is incomplete.  
**Fix:** At minimum log the partial state; ideally wrap in a DB transaction with rollback on failure.

### BUG 7 — Minor: Token limit vs char limit mismatch
**File:** `summarize_today.py`, lines 195, 208  
**Problem:** The prompt asks the LLM for "max 200 Zeichen" (characters) but sets `max_tokens=150`. In German, 150 tokens ≈ 150 characters, not 200. The LLM may produce responses that get truncated at 150 tokens even though the instruction tells it to write up to 200 characters.  
**Fix:** Either increase `max_tokens` to ~250 or reduce the instruction to match 150 tokens.

---

## 3. Design-Empfehlungen

### Recommendation 1: Type-safety guards for DB-derived values
Add defensive handling in `should_ingest()` and any function that processes raw DB row values:

```python
title = (session.get("title") or "")
content_text = (content or "")
text = (title + " " + content_text).lower()
```

This prevents crashes from NULL/None values in any column (title, content, or future additions).

### Recommendation 2: Fine-tune the social-chat keyword filter
Replace the current broad keywords with more specific patterns:

- `"kaffee"` → require `"kaffee trinken"` or `"kaffee pause"` context
- `"morgen"` → require `"guten morgen"` (already covered) instead of bare `"morgen"`
- `"hallo"` → require `"hallo"` in the title with no additional substantive content

Or use a multi-stage approach: if the session looks like a greeting, check the remaining content for project keywords before deciding.

### Recommendation 3: Configurable assistant verbosity filter
Either:
- Make the 80-char threshold configurable (set to 0 to disable)
- Or remove the filter entirely (the summarizer handles short content fine)
- Or use a ratio-based filter: exclude assistant messages shorter than X% of the preceding user message

### Recommendation 4: Transaction safety for pgvector writes
Wrap the `write_to_pgvector` upsert calls in a DB transaction:

```python
with db.conn:
    db.upsert_summary(...)  # combined
    for s in session_summaries:
        db.upsert_summary(...)
```

This ensures atomicity — either all writes succeed or none do.

### Recommendation 5: Project detection edge cases
The current `detect_project` uses simple substring matching in priority order (KnowWhere → Era-Pet → Moradbakhti → Infrastruktur). Consider:
- Word-boundary matching to avoid false positives ("summarize" matching "summary")
- A confidence score rather than first-match-wins
- Detecting multi-project sessions and either splitting or tagging both

---

## 4. Key Strengths Verified

- ✅ **ON CONFLICT (session_id)** works correctly: double inserts become updates
- ✅ **UCB formula** present and correct in SQL: `(1.0 + %s * (ucb_score - 1.0))`
- ✅ **Debut bypass** works: `debut_seen=FALSE` rows bypass the min_score filter
- ✅ **mark_seen** only touches debuts, does NOT increment view_count (separation of concerns)
- ✅ **record_access** increments view_count, does NOT touch debut_seen
- ✅ **get_session_content** handles None content gracefully (skips iteration)
- ✅ **generate_session_summaries** passes through all project keywords for detection
- ✅ **generate_combined_summary** builds proper session lines with project metadata

---

## 5. Quick-Fix Patch

If you want to fix the critical bugs immediately, here are the minimal patches:

**`summarize_today.py` line 151:**
```python
# Before:
text = (session["title"] + " " + content).lower()
# After:
text = ((session.get("title") or "") + " " + (content or "")).lower()
```

**`knowwhere_db.py` line 66:**
```python
# Remove the entire line — emb_bytes is dead code.
```

---

*Generated by `test_pipeline.py` — 109 unit tests, 0 failures.*
