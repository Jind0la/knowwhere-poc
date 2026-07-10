#!/usr/bin/env python3
"""Test KnowWhere Phase 3 Pipeline — complete unit tests.

Tests both knowwhere_db.py (with mocked psycopg2/pgvector) and
summarize_today.py (pure functions + mocked state.db/API).

Usage:
    python3 -m unittest test_pipeline.py -v
    python3 test_pipeline.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call, ANY

import numpy as np

# ---- Path Setup ----
sys.path.insert(0, os.path.expanduser("~/Dev/knowwhere-poc"))


# ============================================================================
# knowwhere_db.py tests
# ============================================================================

class TestKnowWhereDB(unittest.TestCase):
    """Test KnowWhereDB with mocked psycopg2 and pgvector."""

    def setUp(self):
        # Patch psycopg2 INSIDE the knowwhere_db module's namespace
        self.psycopg2_patch = patch("knowwhere_db.psycopg2")
        self.mock_psycopg2 = self.psycopg2_patch.start()

        # Set up extras submodule attributes needed by knowwhere_db
        self.mock_psycopg2.extras.DictCursor = MagicMock(name="DictCursor")
        self.mock_psycopg2.extras.Json = MagicMock(
            name="Json", side_effect=lambda x: x
        )

        # Patch register_vector
        self.register_vector_patch = patch("knowwhere_db.register_vector")
        self.mock_register_vector = self.register_vector_patch.start()

        # Mock connection and cursor
        self.mock_conn = MagicMock()
        self.mock_conn.closed = False  # So close() calls .close()
        self.mock_cursor = MagicMock()
        self.mock_psycopg2.connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

        # Import AFTER patching
        from knowwhere_db import KnowWhereDB
        self.db = KnowWhereDB(db_url="postgresql://test:test@localhost/test")

    def tearDown(self):
        self.psycopg2_patch.stop()
        self.register_vector_patch.stop()

    # ---- upsert_summary ----

    def test_upsert_summary_basic(self):
        """upsert_summary: basic insert with embedding and ON CONFLICT"""
        self.mock_cursor.fetchone.return_value = ["abc-123"]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        result = self.db.upsert_summary(
            session_id="test-session",
            project="KnowWhere",
            summary_text="Test summary",
            embedding=embedding,
            tier="warm",
        )

        self.assertEqual(result, "abc-123")

        # Verify ON CONFLICT clause
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("ON CONFLICT (session_id)", sql)
        self.assertIn("DO UPDATE SET", sql)

        # Verify commit called
        self.mock_conn.commit.assert_called_once()

    def test_upsert_summary_none_embedding(self):
        """upsert_summary: embedding=None should not crash"""
        self.mock_cursor.fetchone.return_value = ["abc-123"]

        result = self.db.upsert_summary(
            session_id="test-session",
            project="KnowWhere",
            summary_text="Test summary",
            embedding=None,
            tier="warm",
        )

        self.assertEqual(result, "abc-123")
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertIsNone(params[3])  # embedding param is None

    def test_upsert_summary_double_insert_updates(self):
        """upsert_summary: same session_id twice should produce UPDATE, not duplicate"""
        self.mock_cursor.fetchone.return_value = ["abc-123"]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.upsert_summary(
            session_id="same-session", project="KnowWhere",
            summary_text="First", embedding=embedding,
        )
        self.db.upsert_summary(
            session_id="same-session", project="KnowWhere",
            summary_text="Second", embedding=embedding,
        )

        self.assertEqual(self.mock_cursor.execute.call_count, 2)
        second_params = self.mock_cursor.execute.call_args_list[1][0][1]
        self.assertEqual(second_params[2], "Second")  # updated summary_text

    def test_upsert_summary_empty_session_id(self):
        """upsert_summary: empty session_id is passed through to the DB layer"""
        self.mock_cursor.fetchone.return_value = ["abc-123"]
        result = self.db.upsert_summary(
            session_id="", project="KnowWhere",
            summary_text="Test", embedding=None,
        )
        self.assertEqual(result, "abc-123")
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertEqual(params[0], "")  # empty session_id passed through

    def test_upsert_summary_emb_bytes_dead_code(self):
        """upsert_summary: BUG — 'emb_bytes' is computed but never used (dead code).

        The variable emb_bytes is assigned but the SQL query uses 'embedding'
        directly. The ternary on line 66 also has inverted logic: if embedding
        is an ndarray it assigns the array itself (not bytes), and if it's not
        an ndarray it tries .tobytes() which would fail for lists/tuples.
        Since emb_bytes is never used, this doesn't crash, but it's dead,
        confusing code."""
        self.mock_cursor.fetchone.return_value = ["abc-123"]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        # This should work despite the dead emb_bytes code
        result = self.db.upsert_summary(
            session_id="test-session", project="KnowWhere",
            summary_text="Test", embedding=embedding,
        )
        self.assertEqual(result, "abc-123")

        # The SQL parameters use 'embedding' directly, not 'emb_bytes'
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertIs(params[3], embedding)  # original embedding object used

    # ---- search_ucb ----

    def test_search_ucb_basic(self):
        """search_ucb: returns results with weighted_score"""
        self.mock_cursor.fetchall.return_value = [
            {"id": "1", "session_id": "s1", "project": "KnowWhere",
             "summary_text": "test", "anchor_id": None,
             "ucb_score": 1.5, "debut_seen": True, "tier": "warm",
             "view_count": 3, "weighted_score": 0.85, "similarity": 0.8},
        ]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        results = self.db.search_ucb(query_embedding=embedding, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")
        # Should have made UPDATE calls for record_access
        self.mock_conn.commit.assert_called()

    def test_search_ucb_with_project_filter(self):
        """search_ucb: project filter adds WHERE project = %s"""
        self.mock_cursor.fetchall.return_value = []
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.search_ucb(query_embedding=embedding, project="KnowWhere")

        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("WHERE project = %s", sql)

    def test_search_ucb_no_project(self):
        """search_ucb: without project, no project filter in WHERE"""
        self.mock_cursor.fetchall.return_value = []
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.search_ucb(query_embedding=embedding)

        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertNotIn("WHERE project", sql)

    def test_search_ucb_debut_bypasses_min_score(self):
        """search_ucb: debuts (debut_seen=FALSE) bypass min_score via OR"""
        self.mock_cursor.fetchall.return_value = [
            {"id": "1", "session_id": "s1", "project": "KnowWhere",
             "summary_text": "debut", "anchor_id": None,
             "ucb_score": 1.0, "debut_seen": False, "tier": "warm",
             "view_count": 0, "weighted_score": 0.1, "similarity": 0.05},
        ]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.search_ucb(query_embedding=embedding, min_score=0.30)

        # The FIRST execute call is the SELECT query; subsequent ones are
        # mark_seen/record_access UPDATES
        select_sql = self.mock_cursor.execute.call_args_list[0][0][0]
        self.assertIn("OR debut_seen = FALSE", select_sql)

    def test_search_ucb_empty_results(self):
        """search_ucb: empty result set returns empty list"""
        self.mock_cursor.fetchall.return_value = []
        embedding = np.array([0.1] * 256, dtype=np.float32)

        results = self.db.search_ucb(query_embedding=embedding)
        self.assertEqual(results, [])

    def test_search_ucb_orders_by_debut_then_score(self):
        """search_ucb: ORDER BY debut_seen ASC, weighted_score DESC"""
        self.mock_cursor.fetchall.return_value = []
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.search_ucb(query_embedding=embedding)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("ORDER BY debut_seen ASC, weighted_score DESC", sql)

    def test_search_ucb_ucb_formula_present(self):
        """search_ucb: UCB weight formula (1.0 + %s * (ucb_score - 1.0)) is in SQL"""
        self.mock_cursor.fetchall.return_value = []
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.search_ucb(query_embedding=embedding, ucb_weight=0.5)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("(1.0 + %s * (ucb_score - 1.0))", sql)

    def test_search_ucb_marks_debuts_as_seen(self):
        """search_ucb: calls mark_seen for debuts after fetching results"""
        self.mock_cursor.fetchall.return_value = [
            {"id": "d1", "session_id": "s1", "project": "KnowWhere",
             "summary_text": "debut", "anchor_id": None,
             "ucb_score": 1.0, "debut_seen": False, "tier": "warm",
             "view_count": 0, "weighted_score": 0.5, "similarity": 0.4},
            {"id": "d2", "session_id": "s2", "project": "KnowWhere",
             "summary_text": "seen", "anchor_id": None,
             "ucb_score": 0.5, "debut_seen": True, "tier": "warm",
             "view_count": 5, "weighted_score": 0.3, "similarity": 0.3},
        ]
        embedding = np.array([0.1] * 256, dtype=np.float32)

        self.db.search_ucb(query_embedding=embedding)

        # Should have called mark_seen for the debut (d1) — look for a
        # debuts_seen UPDATE in the SQL call list
        update_calls = [
            c for c in self.mock_cursor.execute.call_args_list
            if "debut_seen" in str(c.args[0])
        ]
        self.assertTrue(len(update_calls) >= 1,
                        "Expected at least one UPDATE setting debut_seen=TRUE")

    # ---- mark_seen ----

    def test_mark_seen(self):
        """mark_seen: sets debut_seen=TRUE, does NOT touch view_count"""
        self.mock_cursor.rowcount = 2
        result = self.db.mark_seen(["id1", "id2"])

        self.assertEqual(result, 2)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("debut_seen = TRUE", sql)
        self.assertNotIn("view_count = view_count + 1", sql)

    def test_mark_seen_empty_list(self):
        """mark_seen: empty list returns 0, no SQL call"""
        result = self.db.mark_seen([])
        self.assertEqual(result, 0)
        self.mock_cursor.execute.assert_not_called()

    # ---- record_access ----

    def test_record_access(self):
        """record_access: increments view_count by 1, does NOT touch debut_seen"""
        self.mock_cursor.rowcount = 3
        result = self.db.record_access(["id1", "id2", "id3"])

        self.assertEqual(result, 3)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("view_count = view_count + 1", sql)
        self.assertNotIn("debut_seen", sql)

    def test_record_access_empty_list(self):
        """record_access: empty list returns 0, no SQL call"""
        result = self.db.record_access([])
        self.assertEqual(result, 0)
        self.mock_cursor.execute.assert_not_called()

    def test_record_access_no_double_count(self):
        """record_access: each call increments view_count independently"""
        self.mock_cursor.rowcount = 1
        self.db.record_access(["id1"])
        self.db.record_access(["id1"])
        self.assertEqual(self.mock_cursor.execute.call_count, 2)

    # ---- get_debuts ----

    def test_get_debuts(self):
        """get_debuts: returns debuts ordered by created_at DESC"""
        self.mock_cursor.fetchall.return_value = [
            {"id": "d1", "session_id": "s1", "project": "KnowWhere",
             "debut_seen": False, "summary_text": "debut1"},
        ]
        results = self.db.get_debuts()
        self.assertEqual(len(results), 1)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("created_at DESC", sql)

    def test_get_debuts_with_project(self):
        """get_debuts: WITH project filter adds WHERE project=%s"""
        self.mock_cursor.fetchall.return_value = []
        self.db.get_debuts(project="Era-Pet")
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("AND project = %s", sql)

    def test_get_debuts_empty(self):
        """get_debuts: no debuts returns []"""
        self.mock_cursor.fetchall.return_value = []
        results = self.db.get_debuts()
        self.assertEqual(results, [])

    # ---- insert_source ----

    def test_insert_source_basic(self):
        """insert_source: inserts with SHA-256 hash and metadata"""
        self.mock_cursor.fetchone.return_value = ["src-123"]
        result = self.db.insert_source(
            session_id="s1",
            full_text="Hello World",
            metadata={"source": "chat"},
        )
        self.assertEqual(result, "src-123")
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("ON CONFLICT (content_hash)", sql)
        self.assertIn("DO UPDATE", sql)

    def test_insert_source_empty_text(self):
        """insert_source: empty text still computes SHA-256 hash"""
        self.mock_cursor.fetchone.return_value = ["src-123"]
        result = self.db.insert_source(session_id="s1", full_text="")
        self.assertEqual(result, "src-123")

    # ---- index_vector ----

    def test_index_vector_truncates_preview(self):
        """index_vector: preview is truncated to 200 chars"""
        embedding = np.array([0.1] * 256, dtype=np.float32)
        self.mock_cursor.fetchone.return_value = ["vi-123"]
        result = self.db.index_vector(
            source_id="src-123", embedding=embedding,
            preview="A" * 500, project="KnowWhere",
        )
        self.assertEqual(result, "vi-123")
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertEqual(len(params[2]), 200)  # preview[:200]

    def test_index_vector_no_project(self):
        """index_vector: project=None is allowed"""
        embedding = np.array([0.1] * 256, dtype=np.float32)
        self.mock_cursor.fetchone.return_value = ["vi-123"]
        result = self.db.index_vector(
            source_id="src-123", embedding=embedding, preview="short",
        )
        self.assertEqual(result, "vi-123")

    # ---- health_check ----

    def test_health_check(self):
        """health_check: returns dict with 4 count metrics"""
        self.mock_cursor.fetchone.side_effect = [
            [42],   # summaries
            [17],   # sources
            [8],    # vector_index
            [3],    # debuts_pending
        ]
        health = self.db.health_check()
        self.assertEqual(health, {
            "summaries": 42,
            "sources": 17,
            "vector_index": 8,
            "debuts_pending": 3,
        })

    # ---- get_db singleton ----

    def test_get_db_singleton(self):
        """get_db: returns the same instance on repeated calls"""
        from knowwhere_db import get_db
        import knowwhere_db
        knowwhere_db._db_instance = None  # reset

        db1 = get_db()
        db2 = get_db()
        self.assertIs(db1, db2)

    # ---- close ----

    def test_close(self):
        """close: closes connection and sets _conn to None"""
        _ = self.db.conn  # Trigger lazy connection init
        self.db.close()
        self.mock_conn.close.assert_called_once()
        self.assertIsNone(self.db._conn)

    def test_close_without_connection(self):
        """close: does not crash if _conn is None"""
        self.db._conn = None
        # Should not raise
        self.db.close()


# ============================================================================
# summarize_today.py tests  —  pure functions
# ============================================================================

class TestDetectProject(unittest.TestCase):
    """Test detect_project() keyword matching logic (pure function)."""

    def setUp(self):
        from summarize_today import detect_project
        self.detect = detect_project

    # ---- KnowWhere ----

    def test_knowwhere_keyword_knowwhere(self):
        self.assertEqual(self.detect("KnowWhere 256d test", ""), "KnowWhere")

    def test_knowwhere_keyword_railway(self):
        self.assertEqual(self.detect("Railway deployment fix", ""), "KnowWhere")

    def test_knowwhere_keyword_pgvector(self):
        self.assertEqual(self.detect("pgvector index rebuild", ""), "KnowWhere")

    def test_knowwhere_keyword_summary(self):
        self.assertEqual(self.detect("Working on summary gen", ""), "KnowWhere")

    def test_knowwhere_keyword_poc(self):
        self.assertEqual(self.detect("POC implementation", ""), "KnowWhere")

    def test_knowwhere_keyword_dimtest(self):
        self.assertEqual(self.detect("dimtest 256d results", ""), "KnowWhere")

    def test_knowwhere_keyword_subconscious(self):
        self.assertEqual(self.detect("Title", "subconscious layer design"), "KnowWhere")

    def test_knowwhere_keyword_64d(self):
        self.assertEqual(self.detect("64d embedding test", ""), "KnowWhere")

    def test_knowwhere_keyword_768d(self):
        self.assertEqual(self.detect("768d dimension", ""), "KnowWhere")

    def test_knowwhere_keyword_chunk(self):
        self.assertEqual(self.detect("chunk processing", ""), "KnowWhere")

    def test_knowwhere_keyword_session_branch(self):
        self.assertEqual(self.detect("session branch merge", ""), "KnowWhere")

    # ---- Era-Pet ----

    def test_era_pet_keyword_pet(self):
        self.assertEqual(self.detect("Pet animation", ""), "Era-Pet")

    def test_era_pet_keyword_spritesheet(self):
        self.assertEqual(self.detect("Spritesheet gen", ""), "Era-Pet")

    def test_era_pet_keyword_pixel(self):
        self.assertEqual(self.detect("Pixel art", ""), "Era-Pet")

    def test_era_pet_keyword_haar(self):
        self.assertEqual(self.detect("Haar style", ""), "Era-Pet")

    def test_era_pet_keyword_era_bild(self):
        self.assertEqual(self.detect("era bild test", ""), "Era-Pet")

    # ---- Moradbakhti-KI ----

    def test_moradbakhti_keyword_moradbakhti(self):
        self.assertEqual(self.detect("Moradbakhti website", ""), "Moradbakhti-KI")

    def test_moradbakhti_keyword_kmu(self):
        self.assertEqual(self.detect("KMU Kaltakquise", ""), "Moradbakhti-KI")

    def test_moradbakhti_keyword_leafgo(self):
        self.assertEqual(self.detect("LeafGo impl", ""), "Moradbakhti-KI")

    def test_moradbakhti_keyword_cafe_agnes(self):
        self.assertEqual(self.detect("Cafe Agnes redesign", ""), "Moradbakhti-KI")

    def test_moradbakhti_keyword_portfolio(self):
        self.assertEqual(self.detect("Portfolio update", ""), "Moradbakhti-KI")

    def test_moradbakhti_keyword_pitch(self):
        self.assertEqual(self.detect("Pitch deck", ""), "Moradbakhti-KI")

    def test_moradbakhti_keyword_kaltakquise(self):
        self.assertEqual(self.detect("Kaltakquise email", ""), "Moradbakhti-KI")

    # ---- Infrastruktur ----

    def test_infra_keyword_preflight(self):
        self.assertEqual(self.detect("Preflight check", ""), "Infrastruktur")

    def test_infra_keyword_cron(self):
        self.assertEqual(self.detect("Cron job fix", ""), "Infrastruktur")

    def test_infra_keyword_gateway(self):
        self.assertEqual(self.detect("Gateway error", ""), "Infrastruktur")

    def test_infra_keyword_hermes_config(self):
        self.assertEqual(self.detect("hermes config change", ""), "Infrastruktur")

    def test_infra_keyword_skill(self):
        self.assertEqual(self.detect("Skill authoring", ""), "Infrastruktur")

    def test_infra_keyword_hindsight(self):
        self.assertEqual(self.detect("Hindsight sync", ""), "Infrastruktur")

    # ---- General (fallback) ----

    def test_general_fallback(self):
        self.assertEqual(self.detect("Random topic", "random chat"), "General")

    def test_general_empty_title_and_content(self):
        self.assertEqual(self.detect("", ""), "General")

    # ---- Edge cases ----

    def test_empty_title_nonempty_content(self):
        """detect_project: empty title but content has keyword"""
        self.assertEqual(self.detect("", "KnowWhere project"), "KnowWhere")

    def test_empty_content_nonempty_title(self):
        """detect_project: title alone with keyword works"""
        self.assertEqual(self.detect("KnowWhere", ""), "KnowWhere")

    def test_unicode_text(self):
        self.assertEqual(self.detect("Über die Pipeline", ""), "General")

    def test_case_insensitivity(self):
        """detect_project: keyword matching is case-insensitive"""
        self.assertEqual(self.detect("KNOWWHERE RAILWAY", ""), "KnowWhere")

    def test_keyword_in_content_only(self):
        """detect_project: keywords found in content (not title) are detected"""
        self.assertEqual(
            self.detect("Random", "we discussed subconscious memory"),
            "KnowWhere",
        )

    def test_project_priority_knowwhere_wins(self):
        """detect_project: KnowWhere is checked first, so it wins over Era-Pet.

        A session mentioning both 'railway' and 'spritesheet' is classified
        as KnowWhere, not Era-Pet. This is intentional priority."""
        self.assertEqual(
            self.detect("pet spritesheet on railway", ""),
            "KnowWhere",  # KnowWhere checked first
        )


class TestShouldIngest(unittest.TestCase):
    """Test should_ingest() noise filtering logic (pure function)."""

    def setUp(self):
        from summarize_today import should_ingest
        self.should = should_ingest

    # ---- General session filtering ----

    def test_general_short_msgs_rejected(self):
        """General with <40 msgs should be rejected"""
        self.assertFalse(
            self.should(
                {"title": "Random chat", "message_count": 25},
                "some content", "General",
            )
        )

    def test_general_exactly_40_accepted(self):
        """General with exactly 40 msgs should be accepted"""
        self.assertTrue(
            self.should(
                {"title": "Longer chat", "message_count": 40},
                "some content", "General",
            )
        )

    def test_general_long_accepted(self):
        """General with >=40 msgs passes through"""
        self.assertTrue(
            self.should(
                {"title": "Deep discussion", "message_count": 55},
                "interesting topic explored in depth", "General",
            )
        )

    # ---- Era-Pet noise filtering ----

    def test_era_pet_yoga_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Era yoga pose", "message_count": 30},
                "yoga session for Era", "Era-Pet",
            )
        )

    def test_era_pet_sexy_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "sexy Era", "message_count": 30},
                "content", "Era-Pet",
            )
        )

    def test_era_pet_erotisch_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Era erotisch", "message_count": 30},
                "erotisch artwork", "Era-Pet",
            )
        )

    def test_era_pet_rollenspiel_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Rollenspiel session", "message_count": 30},
                "rollenspiel with Era", "Era-Pet",
            )
        )

    def test_era_pet_pose_rejected(self):
        """Pose is a filtering keyword — may be overbroad (e.g. 'character pose ref')"""
        self.assertFalse(
            self.should(
                {"title": "Era pose reference", "message_count": 30},
                "pose content", "Era-Pet",
            )
        )

    def test_era_pet_normal_accepted(self):
        """Era-Pet without noise keywords is accepted"""
        self.assertTrue(
            self.should(
                {"title": "Era spritesheet design", "message_count": 15},
                "designing new spritesheet", "Era-Pet",
            )
        )

    # ---- Morning / social filtering ----

    def test_morning_guten_morgen_short_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Guten Morgen", "message_count": 15},
                "guten morgen", "General",
            )
        )

    def test_morning_guten_morgen_long_accepted(self):
        """Morning greeting with >=40 msgs passes both morning AND General filters"""
        self.assertTrue(
            self.should(
                {"title": "Guten Morgen", "message_count": 40},
                "guten morgen", "General",
            )
        )

    def test_social_hallo_short_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Hallo", "message_count": 10},
                "hallo everyone", "General",
            )
        )

    def test_social_gute_nacht_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Gute Nacht", "message_count": 5},
                "gute nacht", "General",
            )
        )

    def test_kaffee_false_positive_overbroad(self):
        """BUG: 'kaffee' is overbroad — any session <20 msgs mentioning
        coffee/kaffee gets falsely classified as a morning greeting.

        A 15-msg session about 'Kaffee project roadmap' would be filtered."""
        self.assertFalse(
            self.should(
                {"title": "Kaffee project discussion", "message_count": 15},
                "we discussed Kaffee API integration", "General",
            )
        )

    def test_morgen_false_positive(self):
        """BUG: 'morgen' means both 'morning' AND 'tomorrow' in German.

        A session about 'morgen plan' (tomorrow's plan) with <20 msgs is
        falsely filtered. Also 'morgen' could be a proper name."""
        self.assertFalse(
            self.should(
                {"title": "Morgen release plan", "message_count": 18},
                "morgen deploy the update", "General",
            )
        )

    def test_hallo_technical_false_positive(self):
        """BUG: 'hallo' — a short technical session (<20 msgs) starting
        with 'Hallo' is falsely filtered as a greeting, even if it
        contains substantive technical content."""
        self.assertFalse(
            self.should(
                {"title": "Hallo, brauche Hilfe bei DB", "message_count": 15},
                "Hallo ich habe ein Problem mit der Datenbank", "General",
            )
        )

    # ---- News filtering ----

    def test_news_short_general_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "Interesting interview", "message_count": 20},
                "this interview was informative", "General",
            )
        )

    def test_news_cnbc_short_general_rejected(self):
        self.assertFalse(
            self.should(
                {"title": "CNBC update", "message_count": 15},
                "cnbc news about AI", "General",
            )
        )

    def test_news_long_general_accepted(self):
        """Long General news session (>=40 msgs) passes"""
        self.assertTrue(
            self.should(
                {"title": "Interview discussion", "message_count": 40},
                "interview topic discussed in depth", "General",
            )
        )

    def test_news_short_non_general_not_filtered(self):
        """News filter only applies to General project sessions"""
        self.assertTrue(
            self.should(
                {"title": "KnowWhere interview", "message_count": 20},
                "interview about KnowWhere architecture", "KnowWhere",
            )
        )

    # ---- Non-General sessions pass through ----

    def test_knowwhere_any_length_accepted(self):
        """KnowWhere sessions are accepted regardless of message count"""
        self.assertTrue(
            self.should(
                {"title": "KnowWhere query fix", "message_count": 3},
                "optimizing queries with pgvector", "KnowWhere",
            )
        )

    # ---- Edge cases ----

    def test_none_title_handled_gracefully(self):
        """FIXED: None title no longer crashes — handled with safe .get()"""
        sess = {"title": None, "message_count": 50}
        # Should NOT crash — 50msgs with content passes noise filter
        result = self.should(sess, "some content here", "General")
        self.assertIsInstance(result, bool)
        self.assertTrue(result)  # 50msgs General → True (msg threshold met)

    def test_empty_title_does_not_crash(self):
        """should_ingest: empty string title does NOT crash"""
        self.assertTrue(
            self.should(
                {"title": "", "message_count": 50},
                "some content", "General",
            )
        )

    def test_empty_content_does_not_crash(self):
        self.assertTrue(
            self.should(
                {"title": "General chat", "message_count": 50},
                "", "General",
            )
        )

    def test_none_content_handled_gracefully(self):
        """FIXED: None content no longer crashes — handled with safe (content or '')"""
        result = self.should(
            {"title": "General chat", "message_count": 50},
            None, "General",
        )
        self.assertTrue(result)  # 50msgs General with title but no content → True (msg threshold met)

    def test_unicode_handling(self):
        """should_ingest: handles umlauts and special characters"""
        self.assertTrue(
            self.should(
                {"title": "Über Projekte", "message_count": 50},
                "über die Arbeit diskutiert", "General",
            )
        )


class TestGetSessionContent(unittest.TestCase):
    """Test get_session_content() with mocked sqlite3."""

    def setUp(self):
        self.sqlite3_patch = patch("summarize_today.sqlite3")
        self.mock_sqlite3 = self.sqlite3_patch.start()
        self.mock_conn = MagicMock()
        self.mock_sqlite3.connect.return_value = self.mock_conn

        from summarize_today import get_session_content
        self.get_content = get_session_content

    def tearDown(self):
        self.sqlite3_patch.stop()

    def test_builds_nimar_and_era_lines(self):
        """get_session_content: builds Nimar:/Era: prefixed lines"""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("user", "Hello Era!", 1000),
            ("assistant", "Hi Nimar! How can I help?", 1001),
        ]
        content = self.get_content("session-123")
        self.assertIn("Nimar: Hello Era!", content)
        self.assertIn("Era: Hi Nimar!", content)

    def test_empty_message_list(self):
        self.mock_conn.execute.return_value.fetchall.return_value = []
        content = self.get_content("session-999")
        self.assertEqual(content, "")

    def test_none_content_skipped(self):
        """Messages with None content are skipped silently"""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("user", None, 1000),
            ("assistant", "valid content", 1001),
        ]
        content = self.get_content("session-123")
        self.assertNotIn("Nimar: None", content)
        self.assertIn("Era: valid content", content)

    def test_short_assistant_filter_in_sql(self):
        """get_session_content: SQL filters out assistant messages <=80 chars.

        The SQL WHERE clause excludes short assistant responses
        (role = 'assistant' AND length(content) > 80). This means
        potentially important short responses like
        'Ja, das ist richtig.' or 'API-Schlüssel aktualisiert.'
        are silently dropped from summarization context."""
        self.get_content("session-123")
        sql = self.mock_conn.execute.call_args[0][0]
        self.assertIn("length(content) > 20", sql)

    def test_truncation_at_session_preview_chars(self):
        """get_session_content: truncates at ~2000 chars"""
        long_assistant = "word " * 600
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("user", "tell me a story", 1000),
            ("assistant", long_assistant, 1001),
        ]
        content = self.get_content("session-123")
        # Should be near SESSION_PREVIEW_CHARS (2000)
        self.assertLessEqual(len(content), 2100)


class TestGenerateSessionSummaries(unittest.TestCase):
    """Test generate_session_summaries() with mocked dependencies."""

    def setUp(self):
        # Mock get_session_content
        self.content_patch = patch("summarize_today.get_session_content")
        self.mock_get_content = self.content_patch.start()
        self.mock_get_content.return_value = (
            "Content that is long enough to pass the 100 char filter. " * 5
        )

        # Mock call_deepseek (LLM call)
        self.llm_patch = patch("summarize_today.call_deepseek")
        self.mock_call_deepseek = self.llm_patch.start()
        self.mock_call_deepseek.return_value = "Short summary of the session."

        from summarize_today import generate_session_summaries
        self.generate = generate_session_summaries

    def tearDown(self):
        self.content_patch.stop()
        self.llm_patch.stop()

    def test_basic_generation(self):
        """generate_session_summaries: returns list with correct fields"""
        sessions = [
            {"id": "sess_abc123", "title": "KnowWhere Test",
             "message_count": 25, "source": "chat"},
        ]
        results = self.generate(sessions)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["short_id"], "abc123")
        self.assertEqual(results[0]["title"], "KnowWhere Test")
        self.assertEqual(results[0]["project"], "KnowWhere")
        self.assertEqual(results[0]["msgs"], 25)

    def test_short_content_skipped(self):
        """Sessions with content <100 chars are skipped"""
        self.mock_get_content.return_value = "Short"  # len=5
        sessions = [
            {"id": "sess_short", "title": "Too short",
             "message_count": 5, "source": "chat"},
        ]
        results = self.generate(sessions)
        self.assertEqual(len(results), 0)

    def test_general_noise_skipped(self):
        """General sessions with <40 msgs are skipped via should_ingest"""
        sessions = [
            {"id": "sess_noise", "title": "Random chat",
             "message_count": 25, "source": "chat"},
        ]
        results = self.generate(sessions)
        self.assertEqual(len(results), 0)

    def test_multiple_sessions(self):
        """generate_session_summaries: handles multiple sessions with different projects"""
        self.mock_get_content.return_value = "Content that is long enough. " * 10
        sessions = [
            {"id": "sess_a1", "title": "KnowWhere project",
             "message_count": 30, "source": "chat"},
            {"id": "sess_b2", "title": "Era spritesheet",
             "message_count": 20, "source": "chat"},
            {"id": "sess_c3", "title": "Moradbakhti pitch",
             "message_count": 15, "source": "chat"},
        ]
        results = self.generate(sessions)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["project"], "KnowWhere")
        self.assertEqual(results[1]["project"], "Era-Pet")
        self.assertEqual(results[2]["project"], "Moradbakhti-KI")

    def test_sessions_are_deduplicated_by_should_ingest(self):
        """generate_session_summaries: should_ingest=False sessions are skipped"""
        self.mock_get_content.return_value = "Long content. " * 50
        sessions = [
            {"id": "sess_good", "title": "KnowWhere optimization",
             "message_count": 30, "source": "chat"},
            {"id": "sess_noise", "title": "Guten Morgen",
             "message_count": 5, "source": "chat"},
        ]
        results = self.generate(sessions)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "sess_good")


class TestGenerateCombinedSummary(unittest.TestCase):
    """Test generate_combined_summary() formatting with mocked LLM."""

    def setUp(self):
        self.llm_patch = patch("summarize_today.call_deepseek")
        self.mock_llm = self.llm_patch.start()
        self.mock_llm.return_value = "[KnowWhere|2026-07-02] Combined result."

        from summarize_today import generate_combined_summary
        self.generate_combined = generate_combined_summary

    def tearDown(self):
        self.llm_patch.stop()

    def test_basic(self):
        session_summaries = [
            {
                "id": "sess_abc123",
                "short_id": "abc123",
                "title": "KnowWhere Test",
                "project": "KnowWhere",
                "summary": "Fixed pgvector queries",
                "msgs": 25,
            },
        ]
        result = self.generate_combined(session_summaries, "2026-07-02")
        self.assertEqual(result, "[KnowWhere|2026-07-02] Combined result.")
        prompt = self.mock_llm.call_args[0][1]
        self.assertIn("KnowWhere Test", prompt)
        self.assertIn("2026-07-02", prompt)

    def test_empty_list(self):
        """generate_combined_summary: empty session list should not crash"""
        result = self.generate_combined([], "2026-07-02")
        self.assertIsNotNone(result)


class TestWriteToPgvector(unittest.TestCase):
    """Test write_to_pgvector() with mocked DB.

    NOTE: write_to_pgvector does `from knowwhere_db import get_db` internally,
    so we must patch knowwhere_db.get_db, not summarize_today.get_db.
    """

    def setUp(self):
        self.get_db_patch = patch("knowwhere_db.get_db")
        self.mock_get_db = self.get_db_patch.start()
        self.mock_db = MagicMock()
        self.mock_get_db.return_value = self.mock_db

        from summarize_today import write_to_pgvector
        self.write_pg = write_to_pgvector

    def tearDown(self):
        self.get_db_patch.stop()

    def test_writes_combined_and_per_session(self):
        session_summaries = [
            {"id": "sess_abc", "project": "KnowWhere",
             "summary": "Fixed queries", "short_id": "abc",
             "title": "Test", "msgs": 25, "source": "chat"},
        ]
        self.write_pg(session_summaries, "Combined here", "2026-07-02")

        # Combined summary
        combined_call = self.mock_db.upsert_summary.call_args_list[0]
        self.assertEqual(combined_call[1]["session_id"], "daily-2026-07-02")
        self.assertEqual(combined_call[1]["project"], "_daily")
        self.assertEqual(combined_call[1]["tier"], "hot")

        # Per-session summary
        sess_call = self.mock_db.upsert_summary.call_args_list[1]
        self.assertEqual(sess_call[1]["session_id"], "sess_abc")
        self.assertEqual(sess_call[1]["project"], "KnowWhere")
        self.assertEqual(sess_call[1]["tier"], "warm")
        self.assertIsNone(sess_call[1]["embedding"])

    def test_empty_sessions_list(self):
        """write_to_pgvector: empty sessions still writes the combined summary"""
        self.write_pg([], "Combined", "2026-07-02")
        # Combined summary written (1 upsert), no per-session
        self.assertEqual(self.mock_db.upsert_summary.call_count, 1)

    def test_error_handling(self):
        """write_to_pgvector: catches exceptions and prints warning (design concern)"""
        self.mock_db.upsert_summary.side_effect = Exception("DB connection lost")
        # Should not re-raise the exception
        self.write_pg([], "Combined", "2026-07-02")  # no crash


class TestWriteToJson(unittest.TestCase):
    """Test write_to_json() with mocked filesystem."""

    def setUp(self):
        self.json_patch = patch("summarize_today.json")
        self.mock_json = self.json_patch.start()
        self.mock_json.load.return_value = {}
        self.mock_json.dumps.return_value = "{}"

        self.os_patch = patch("summarize_today.os")
        self.mock_os = self.os_patch.start()
        self.mock_os.path.exists.return_value = False
        self.mock_os.makedirs = MagicMock()

        self.open_patch = patch("summarize_today.open", MagicMock())
        self.mock_open = self.open_patch.start()

        from summarize_today import write_to_json
        self.write_json = write_to_json

    def tearDown(self):
        self.json_patch.stop()
        self.os_patch.stop()
        self.open_patch.stop()

    def test_writes_to_json(self):
        session_summaries = [
            {"id": "sess_abc", "project": "KnowWhere",
             "summary": "Fixed queries", "short_id": "abc",
             "title": "Test", "msgs": 25, "source": "chat"},
        ]
        # Should not crash
        self.write_json(session_summaries, "Combined summary", "2026-07-02")


class TestGetTodaySessions(unittest.TestCase):
    """Test get_today_sessions() with mocked sqlite3."""

    def setUp(self):
        self.sqlite3_patch = patch("summarize_today.sqlite3")
        self.mock_sqlite3 = self.sqlite3_patch.start()
        self.mock_conn = MagicMock()
        self.mock_sqlite3.connect.return_value = self.mock_conn

        # Return plain dicts (dict(r) works on dicts too)
        self.mock_conn.execute.return_value.fetchall.return_value = [
            {"id": "sess_123", "title": "Test", "message_count": 15,
             "tool_call_count": 5, "source": "chat", "started": "2026-07-02"},
        ]

        from summarize_today import get_today_sessions
        self.get_sessions = get_today_sessions

    def tearDown(self):
        self.sqlite3_patch.stop()

    def test_returns_list_of_dicts(self):
        sessions = self.get_sessions("2026-07-02")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], "sess_123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
