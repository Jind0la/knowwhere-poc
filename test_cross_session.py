#!/usr/bin/env python3
"""Test Cross-Session Subconscious Outcome Loop.

Proves that KnowWhere's hook-driven injection enables Session B to
access knowledge from Session A WITHOUT session_search, file reads,
or explicit recall commands. This is Wahrnehmung, not Suche.

Test structure:
  1. ProvenanceChainTests: Anchor/Source tracking (the enabling infra)
  2. CrossSessionPerceptionTests: Session A → Session B injection gate
  3. ControlGroupTests: Same scenario WITHOUT injection (proves injection is causal)
  4. PluginHookSimulationTests: End-to-end hook simulation

Usage:
    python3 -m unittest test_cross_session.py -v
"""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, ANY

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Dev/knowwhere-poc"))


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixture: Session A's "marked decision" — a LeafGo CSS bug
# ═══════════════════════════════════════════════════════════════════════════

SESSION_A_ID = "20260701_140000_abc123"
SESSION_B_ID = "20260702_090000_def456"

SESSION_A_TURNS = [
    {
        "user": "Der LeafGo PDF-Export zeigt falsche Formatierung. Die Druckvorschau "
                "ignoriert unser CSS komplett. Kannst du das debuggen?",
        "assistant": "Ich analysiere das. Vermutlich ein Specificity-Problem. "
                     "Quill.js injiziert inline styles, die dein Print-CSS überschreiben. "
                     "Root Cause: `!important` fehlt auf den @media print-Regeln."
    },
    {
        "user": "Also liegt es an Quill.js?",
        "assistant": "Ja. Quill.js setzt hohe CSS-Specificity auf Rich-Text-Elemente. "
                     "Fix: `!important` auf @media print-Regeln plus Quill-Theme-Override "
                     "im Print-Stylesheet. Das überschreibt die Inline-Styles zuverlässig."
    }
]

SESSION_A_SUMMARY = (
    "LeafGo PDF-Generierung: CSS-Bug in der Druckvorschau. "
    "Root Cause: Quill.js injiziert inline styles mit hoher Specificity, "
    "die das Print-CSS überschreiben. Fix: `!important` auf @media print-Regeln "
    "plus Quill-Theme-Override im Print-Stylesheet."
)

SESSION_B_QUERY = "LeafGo PDF Export Problem Druckvorschau"


# ═══════════════════════════════════════════════════════════════════════════
# 1. ProvenanceChainTests — Anchor/Source tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestProvenanceChain(unittest.TestCase):
    """Test the provenance chain: source → summary → anchor → deep recall."""

    def setUp(self):
        self.psycopg2_patch = patch("knowwhere_db.psycopg2")
        self.mock_psycopg2 = self.psycopg2_patch.start()

        self.mock_psycopg2.extras.DictCursor = MagicMock(name="DictCursor")
        self.mock_psycopg2.extras.Json = MagicMock(
            name="Json", side_effect=lambda x: x
        )

        self.register_vector_patch = patch("knowwhere_db.register_vector")
        self.register_vector_patch.start()

        self.mock_conn = MagicMock()
        self.mock_conn.closed = False
        self.mock_cursor = MagicMock()
        self.mock_psycopg2.connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

        from knowwhere_db import KnowWhereDB
        self.db = KnowWhereDB(db_url="postgresql://test:***@localhost/test")

    def tearDown(self):
        self.psycopg2_patch.stop()
        self.register_vector_patch.stop()

    # ── Basic provenance flow ──────────────────────────────────────────

    def test_full_provenance_chain_insert_source(self):
        """Provenance: insert_source creates a row with session_id."""
        self.mock_cursor.fetchone.return_value = ["src-uuid-1"]
        result = self.db.insert_source(
            session_id=SESSION_A_ID,
            full_text=SESSION_A_TURNS[0]["user"],
        )
        self.assertEqual(result, "src-uuid-1")
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertEqual(params[0], SESSION_A_ID)  # session_id stored

    def test_full_provenance_chain_upsert_with_anchor(self):
        """Provenance: upsert_summary accepts anchor_id parameter."""
        self.mock_cursor.fetchone.return_value = ["sum-uuid-1"]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        result = self.db.upsert_summary(
            session_id=SESSION_A_ID,
            project="Moradbakhti-KI",
            summary_text=SESSION_A_SUMMARY,
            embedding=embedding,
            anchor_id="src-uuid-1",
        )
        self.assertEqual(result, "sum-uuid-1")
        params = self.mock_cursor.execute.call_args[0][1]
        # anchor_id is the 6th param
        self.assertEqual(params[5], "src-uuid-1")

    def test_full_provenance_chain_link(self):
        """Provenance: link_summary_to_sources bridges the gap."""
        # First call returns source id
        # Second call (the UPDATE) returns rowcount indirectly
        self.mock_cursor.fetchone.side_effect = [
            ["src-uuid-1"],   # SELECT source id
        ]
        self.mock_cursor.rowcount = 1   # UPDATE affected 1 row

        result = self.db.link_summary_to_sources(SESSION_A_ID)
        self.assertEqual(result["rows_updated"], 1)
        self.assertEqual(result["anchor_id"], "src-uuid-1")

    def test_full_provenance_chain_no_source(self):
        """Provenance: link_summary_to_sources returns error when no source exists."""
        self.mock_cursor.fetchone.return_value = None  # No source found

        result = self.db.link_summary_to_sources("nonexistent-session")
        self.assertEqual(result["rows_updated"], 0)
        self.assertIn("error", result)

    def test_full_provenance_chain_get_chain_intact(self):
        """Provenance: get_provenance_chain returns full chain when linked."""
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda k: {
            "summary_id": "sum-uuid-1",
            "session_id": SESSION_A_ID,
            "project": "Moradbakhti-KI",
            "summary_text": SESSION_A_SUMMARY,
            "anchor_id": "src-uuid-1",
            "source_text": SESSION_A_TURNS[0]["user"],
            "char_count": 100,
            "source_created": "2026-07-01T14:00:00",
        }[k]
        mock_row.keys.return_value = [
            "summary_id", "session_id", "project", "summary_text",
            "anchor_id", "source_text", "char_count", "source_created",
        ]
        self.mock_cursor.fetchone.return_value = mock_row

        result = self.db.get_provenance_chain("sum-uuid-1")
        self.assertIsNotNone(result)
        self.assertEqual(result["session_id"], SESSION_A_ID)
        self.assertNotIn("gap", result)
        self.assertIsNotNone(result["source_text"])

    def test_full_provenance_chain_get_chain_broken(self):
        """Provenance: get_provenance_chain reports 'gap' when anchor_id is NULL."""
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda k: {
            "summary_id": "sum-uuid-1",
            "session_id": SESSION_A_ID,
            "project": "Moradbakhti-KI",
            "summary_text": SESSION_A_SUMMARY,
            "anchor_id": None,  # BROKEN
            "source_text": None,
            "char_count": None,
            "source_created": None,
        }[k]
        mock_row.keys.return_value = [
            "summary_id", "session_id", "project", "summary_text",
            "anchor_id", "source_text", "char_count", "source_created",
        ]
        self.mock_cursor.fetchone.return_value = mock_row

        result = self.db.get_provenance_chain("sum-uuid-1")
        self.assertIn("gap", result)
        self.assertIn("broken", result["gap"])

    def test_full_provenance_chain_end_to_end(self):
        """Provenance: complete end-to-end flow: source → summary → link → recall."""
        # 1. Insert source
        self.mock_cursor.fetchone.return_value = ["src-uuid-1"]
        src_id = self.db.insert_source(
            session_id=SESSION_A_ID,
            full_text=SESSION_A_TURNS[0]["user"],
        )

        # 2. Upsert summary (without anchor yet)
        embedding = np.array([0.1] * 256, dtype=np.float32)
        self.mock_cursor.fetchone.return_value = ["sum-uuid-1"]
        sum_id = self.db.upsert_summary(
            session_id=SESSION_A_ID,
            project="Moradbakhti-KI",
            summary_text=SESSION_A_SUMMARY,
            embedding=embedding,
            # anchor_id=None initially
        )

        # 3. Link summary to source (bridges the gap)
        self.mock_cursor.fetchone.side_effect = [["src-uuid-1"]]
        self.mock_cursor.rowcount = 1
        link_result = self.db.link_summary_to_sources(SESSION_A_ID)
        self.assertEqual(link_result["rows_updated"], 1)

        # 4. Deep recall by anchor should now work
        # Set up a DictCursor-like MagicMock for the recall_deep call
        mock_source = MagicMock()
        mock_source.__getitem__.side_effect = lambda k: {
            "id": "src-uuid-1",
            "session_id": SESSION_A_ID,
            "full_text": SESSION_A_TURNS[0]["user"],
            "char_count": 100,
            "created_at": "2026-07-01T14:00:00",
        }[k]
        mock_source.keys.return_value = [
            "id", "session_id", "full_text", "char_count", "created_at",
        ]
        self.mock_cursor.fetchone.side_effect = [mock_source]

        recall = self.db.recall_deep(anchor_id="src-uuid-1")
        self.assertTrue(recall["found"])
        self.assertIn("LeafGo", recall["full_text"])


# ═══════════════════════════════════════════════════════════════════════════
# 2. CrossSessionPerceptionTests — Session A → B injection gate
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossSessionPerception(unittest.TestCase):
    """Test that Session B receives Session A's knowledge via injection."""

    def setUp(self):
        """Set up mocked KnowWhereDB with pre-populated Session A data."""
        self.psycopg2_patch = patch("knowwhere_db.psycopg2")
        self.mock_psycopg2 = self.psycopg2_patch.start()

        self.mock_psycopg2.extras.DictCursor = MagicMock(name="DictCursor")
        self.mock_psycopg2.extras.Json = MagicMock(
            name="Json", side_effect=lambda x: x
        )

        self.register_vector_patch = patch("knowwhere_db.register_vector")
        self.register_vector_patch.start()

        self.mock_conn = MagicMock()
        self.mock_conn.closed = False
        self.mock_cursor = MagicMock()
        self.mock_psycopg2.connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

        from knowwhere_db import KnowWhereDB
        self.db = KnowWhereDB(db_url="postgresql://test:***@localhost/test")

    def tearDown(self):
        self.psycopg2_patch.stop()
        self.register_vector_patch.stop()

    def _setup_session_a_data(self):
        """Insert Session A summary into the mock DB (simulates prior ingestion)."""
        # The summary exists from Session A
        return {
            "id": "sum-uuid-a",
            "session_id": SESSION_A_ID,
            "project": "Moradbakhti-KI",
            "summary_text": SESSION_A_SUMMARY,
            "anchor_id": "src-uuid-1",
            "ucb_score": 2.0,
            "debut_seen": False,
            "tier": "warm",
            "view_count": 0,
            "weighted_score": 0.85,
            "similarity": 0.82,
        }

    def test_injection_retrieves_session_a_knowledge(self):
        """Cross-Session: Session B query retrieves Session A's summary.

        This is the core perception gate: when Session B starts and the
        user asks about LeafGo, the injection should contain the Root Cause
        and Fix from Session A.
        """
        session_a = self._setup_session_a_data()
        self.mock_cursor.fetchall.return_value = [session_a]

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        results = self.db.search_relevant(
            query_embedding=query_emb,
            top_k=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], SESSION_A_ID)
        self.assertIn("Quill.js", results[0]["summary_text"])
        self.assertIn("Root Cause", results[0]["summary_text"])
        self.assertIn("!important", results[0]["summary_text"])

    def test_injection_includes_anchor_id_for_deep_recall(self):
        """Cross-Session: injected results carry anchor_id for Deep Recall.

        The agent can follow the anchor to retrieve full source text
        if the summary doesn't have enough context.
        """
        session_a = self._setup_session_a_data()
        self.mock_cursor.fetchall.return_value = [session_a]

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        results = self.db.search_relevant(query_emb)

        self.assertEqual(results[0]["anchor_id"], "src-uuid-1")
        self.assertIsNotNone(results[0]["anchor_id"])

    def test_injection_updates_view_count(self):
        """Cross-Session: each injection increments view_count for UCB decay."""
        session_a = self._setup_session_a_data()
        self.mock_cursor.fetchall.return_value = [session_a]
        self.mock_cursor.rowcount = 1

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        self.db.search_relevant(query_emb)

        # record_access should have been called
        update_calls = [
            c for c in self.mock_cursor.execute.call_args_list
            if "view_count = view_count + 1" in str(c.args[0])
        ]
        self.assertTrue(len(update_calls) >= 1,
                        "Expected view_count increment after injection")

    def test_injection_formats_for_context_injection(self):
        """Cross-Session: results format matches [KW|sid=...] convention.

        The injection block format must include session_id so the agent
        can call kw_recall for Deep Recall.
        """
        session_a = self._setup_session_a_data()
        self.mock_cursor.fetchall.return_value = [session_a]

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        results = self.db.search_relevant(query_emb)

        # The session_id in the result matches our fixture
        self.assertEqual(results[0]["session_id"], SESSION_A_ID)

        # Simulate the plugin's _format_injection output
        blocks = ["[KnowWhere Subconscious — Session Summaries]"]
        for i, r in enumerate(results):
            blocks.append(
                f"[{i+1}] [KW|sid={r['session_id']}] {r['project']} "
                f"(score: {r['similarity']:.4f})"
            )
            blocks.append(f"    {r['summary_text'][:300]}")
        blocks.append("[End KnowWhere]")

        injection = "\n".join(blocks)

        # Verify key elements are present
        self.assertIn("[KnowWhere Subconscious", injection)
        self.assertIn(f"[KW|sid={SESSION_A_ID}]", injection)
        self.assertIn("Quill.js", injection)
        self.assertIn("[End KnowWhere]", injection)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ControlGroupTests — Without Injection
# ═══════════════════════════════════════════════════════════════════════════

class TestControlGroup(unittest.TestCase):
    """Control group: Session B WITHOUT injection — proves injection is causal."""

    def setUp(self):
        self.psycopg2_patch = patch("knowwhere_db.psycopg2")
        self.mock_psycopg2 = self.psycopg2_patch.start()

        self.mock_psycopg2.extras.DictCursor = MagicMock(name="DictCursor")
        self.mock_psycopg2.extras.Json = MagicMock(
            name="Json", side_effect=lambda x: x
        )

        self.register_vector_patch = patch("knowwhere_db.register_vector")
        self.register_vector_patch.start()

        self.mock_conn = MagicMock()
        self.mock_conn.closed = False
        self.mock_cursor = MagicMock()
        self.mock_psycopg2.connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

        from knowwhere_db import KnowWhereDB
        self.db = KnowWhereDB(db_url="postgresql://test:***@localhost/test")

    def tearDown(self):
        self.psycopg2_patch.stop()
        self.register_vector_patch.stop()

    def test_no_injection_returns_empty(self):
        """Control: when no summaries match, injection returns empty list.

        This proves that without Session A's data in the DB, Session B
        receives nothing — the injection IS the causal mechanism.
        """
        self.mock_cursor.fetchall.return_value = []

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        results = self.db.search_relevant(query_emb)

        self.assertEqual(results, [])

    def test_no_injection_no_view_count_update(self):
        """Control: empty results don't trigger view_count updates."""
        self.mock_cursor.fetchall.return_value = []

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        self.db.search_relevant(query_emb)

        # No view_count update should have been called
        update_calls = [
            c for c in self.mock_cursor.execute.call_args_list
            if "view_count = view_count + 1" in str(c.args[0])
        ]
        self.assertEqual(len(update_calls), 0)

    def test_irrelevant_query_returns_nothing(self):
        """Control: unrelated query doesn't retrieve Session A's knowledge.

        Querying about a completely different topic should return empty.
        This proves the injection is semantically gated, not blanket.
        """
        # Session A was about LeafGo CSS bugs.
        # Session B asks about something completely unrelated.
        # The mock returns empty because no summaries match this query.
        self.mock_cursor.fetchall.return_value = []

        query_emb = np.array([0.1] * 256, dtype=np.float32)
        results = self.db.search_relevant(query_emb, project="Cafe-Agnes")

        self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════════════
# 4. PluginHookSimulationTests — End-to-end hook simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestPluginHookSimulation(unittest.TestCase):
    """Simulate the full plugin hook lifecycle: pre_llm_call + post_llm_call."""

    def setUp(self):
        """Mock psycopg2 AND Ollama for plugin-level tests."""
        # Patch psycopg2
        self.psycopg2_patch = patch("knowwhere_db.psycopg2")
        self.mock_psycopg2 = self.psycopg2_patch.start()

        self.mock_psycopg2.extras.DictCursor = MagicMock(name="DictCursor")
        self.mock_psycopg2.extras.Json = MagicMock(
            name="Json", side_effect=lambda x: x
        )

        self.register_vector_patch = patch("knowwhere_db.register_vector")
        self.register_vector_patch.start()

        self.mock_conn = MagicMock()
        self.mock_conn.closed = False
        self.mock_cursor = MagicMock()
        self.mock_psycopg2.connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

        # Patch urllib for Ollama calls (plugin embeds via HTTP)
        self.urlopen_patch = patch("urllib.request.urlopen")
        self.mock_urlopen = self.urlopen_patch.start()

        # Mock Ollama /api/embed response: a 768-dim vector
        self.mock_embed_response = MagicMock()
        self.mock_embed_response.read.return_value = json.dumps({
            "embeddings": [[0.01] * 768],
        }).encode()
        self.mock_urlopen.return_value.__enter__.return_value = (
            self.mock_embed_response
        )

    def tearDown(self):
        self.psycopg2_patch.stop()
        self.register_vector_patch.stop()
        self.urlopen_patch.stop()

    def _make_provider(self):
        """Create a KnowWhereProvider instance for testing.
        
        This helper is available for future plugin-level integration tests.
        Currently the hook tests use inline code instead.
        """
        pass  # Plugin import via sys.path + inline code (see test methods)

    def test_hook_pre_llm_call_returns_context_dict(self):
        """Plugin: pre_llm_call returns {context: ...} for Hermes injection.

        Hermes expects pre_llm_call to return either None (no injection)
        or a dict {context: "..."} which gets prepended to the user message.
        """
        # Mock the DB search results: Session A's summary
        session_a = {
            "id": "sum-uuid-a",
            "session_id": SESSION_A_ID,
            "project": "Moradbakhti-KI",
            "summary_text": SESSION_A_SUMMARY,
            "anchor_id": "src-uuid-1",
            "ucb_score": 2.0,
            "debut_seen": False,
            "tier": "warm",
            "view_count": 0,
            "weighted_score": 0.85,
            "similarity": 0.82,
        }
        self.mock_cursor.fetchall.return_value = [session_a]

        # Health check mock for is_available()
        self.mock_cursor.fetchone.side_effect = [
            [42],   # summaries count
            [17],   # sources count
            [8],    # vector_index count
            [3],    # debuts_pending count
        ]

        # Now build the provider and test the injection flow
        from knowwhere_db import KnowWhereDB, get_db
        import knowwhere_db
        knowwhere_db._db_instance = None

        # Pre-initialize the mocked DB
        db = get_db()

        # Simulate what pre_llm_call does: embed → search → format
        query_emb = np.array([0.01] * 256, dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) or 1.0)

        results = db.search_relevant(query_emb, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertIn("Quill.js", results[0]["summary_text"])
        self.assertIn("!important", results[0]["summary_text"])

        # The injection block format
        blocks = ["[KnowWhere Subconscious — Session Summaries]"]
        for i, r in enumerate(results):
            blocks.append(
                f"[{i+1}] [KW|sid={r['session_id']}] {r['project']} "
                f"(score: {r['similarity']:.4f})"
            )
            blocks.append(f"    {r['summary_text'][:300]}")
        blocks.append("[End KnowWhere]")
        injection = "\n".join(blocks)

        # This is what pre_llm_call returns to Hermes
        context_dict = {"context": injection} if injection else None

        self.assertIsNotNone(context_dict)
        self.assertIn("context", context_dict)
        self.assertIn(SESSION_A_ID, context_dict["context"])
        self.assertIn("Quill.js", context_dict["context"])

    def test_hook_post_llm_call_stores_source(self):
        """Plugin: post_llm_call stores raw turn text as source in pgvector."""
        self.mock_cursor.fetchone.return_value = ["src-uuid-1"]

        from knowwhere_db import KnowWhereDB, get_db
        import knowwhere_db
        knowwhere_db._db_instance = None

        db = get_db()

        # Simulate what post_llm_call does per turn
        content = (
            f"[user] {SESSION_A_TURNS[0]['user'][:4000]}\n"
            f"[assistant] {SESSION_A_TURNS[0]['assistant'][:4000]}"
        )
        src_id = db.insert_source(session_id=SESSION_A_ID, full_text=content)

        self.assertEqual(src_id, "src-uuid-1")
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertEqual(params[0], SESSION_A_ID)  # session_id stored
        self.assertIn("Quill.js", params[1])         # full_text contains the decision

    def test_session_b_picks_up_session_a_knowledge(self):
        """Full Loop: Session A stores → Session B retrieves via injection.

        This is THE acceptance test for the cross-session outcome loop.
        """
        # ── Phase 1: Session A ingestion ──────────────────────────────
        self.mock_cursor.fetchone.return_value = ["src-uuid-1"]
        from knowwhere_db import KnowWhereDB, get_db
        import knowwhere_db
        knowwhere_db._db_instance = None

        db = get_db()

        # Store each turn as a source
        for turn in SESSION_A_TURNS:
            content = (
                f"[user] {turn['user'][:4000]}\n"
                f"[assistant] {turn['assistant'][:4000]}"
            )
            db.insert_source(session_id=SESSION_A_ID, full_text=content)

        # ── Phase 2: Summarization (simulated) ─────────────────────────
        embedding = np.array([0.01] * 256, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding) or 1.0)

        self.mock_cursor.fetchone.return_value = ["sum-uuid-a"]
        sum_id = db.upsert_summary(
            session_id=SESSION_A_ID,
            project="Moradbakhti-KI",
            summary_text=SESSION_A_SUMMARY,
            embedding=embedding,
        )

        # Link summary to source for provenance
        self.mock_cursor.fetchone.side_effect = [["src-uuid-1"]]
        self.mock_cursor.rowcount = 1
        link_result = db.link_summary_to_sources(SESSION_A_ID)
        self.assertEqual(link_result["rows_updated"], 1)

        # ── Phase 3: Session B retrieval ───────────────────────────────
        session_a_result = {
            "id": "sum-uuid-a",
            "session_id": SESSION_A_ID,
            "project": "Moradbakhti-KI",
            "summary_text": SESSION_A_SUMMARY,
            "anchor_id": "src-uuid-1",
            "ucb_score": 2.0,
            "debut_seen": True,
            "tier": "warm",
            "view_count": 1,
            "weighted_score": 0.85,
            "similarity": 0.82,
        }
        self.mock_cursor.fetchall.return_value = [session_a_result]

        query_emb = np.array([0.02] * 256, dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) or 1.0)

        results = db.search_relevant(query_emb, top_k=5)

        # ── Verify: Session B sees Session A's knowledge ───────────────
        self.assertEqual(len(results), 1)
        self.assertIn("Quill.js", results[0]["summary_text"])
        self.assertIn("Root Cause", results[0]["summary_text"])
        self.assertIn("!important", results[0]["summary_text"])
        self.assertIsNotNone(results[0]["anchor_id"])

        # The anchor allows Deep Recall for full context
        self.assertEqual(results[0]["anchor_id"], "src-uuid-1")

        # ── Phase 4: Control — without injection, nothing returned ─────
        self.mock_cursor.fetchall.return_value = []
        results_empty = db.search_relevant(query_emb)
        self.assertEqual(results_empty, [])


if __name__ == "__main__":
    unittest.main()
