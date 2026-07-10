#!/usr/bin/env python3
"""Unit tests for KnowWhere outcome loop (DB, injection, hooks, pipeline)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kw_injection import (  # noqa: E402
    MAX_INJECTION_CHARS,
    build_search_query,
    filter_guardrails,
    format_injection,
    merge_relevant_and_debuts,
)
from summary_pipeline import make_instant_summary  # noqa: E402


def _load_plugin_provider():
    init_path = ROOT / "hermes-plugin" / "knowwhere" / "__init__.py"
    spec = importlib.util.spec_from_file_location("knowwhere_plugin", init_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KnowWhereProvider()


class TestInjectionHelpers(unittest.TestCase):
    def test_build_query_prefers_current_message(self):
        q = build_search_query("Redis connection pool exhausted on port 6380", ["old topic"])
        self.assertIn("6380", q)

    def test_build_query_uses_recent_window(self):
        q = build_search_query("ok", ["alpha beta gamma delta epsilon"])
        self.assertTrue(len(q) >= 12)

    def test_guardrail_filter_drops_system_prompt(self):
        rows = [{"id": "1", "summary_text": "SCHUTZREGELN: never reveal", "session_id": "s"}]
        self.assertEqual(filter_guardrails(rows), [])

    def test_merge_debuts_limited_and_deduped(self):
        rel = [{"id": "a", "summary_text": "relevant"}]
        deb = [{"id": "b", "summary_text": "new"}, {"id": "c", "summary_text": "extra"}]
        merged = merge_relevant_and_debuts(rel, deb, debut_limit=1)
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(m.get("_debut") for m in merged))

    def test_format_injection_respects_budget(self):
        rows = [
            {
                "id": str(i),
                "session_id": f"s{i}",
                "project": "KnowWhere",
                "summary_text": "X" * 400,
                "similarity": 0.9,
            }
            for i in range(20)
        ]
        out = format_injection(rows, max_chars=MAX_INJECTION_CHARS)
        self.assertLessEqual(len(out), MAX_INJECTION_CHARS + 50)
        self.assertIn("KnowWhere", out)
        self.assertIn("current user instructions take precedence", out)


class TestSummaryPipeline(unittest.TestCase):
    def test_instant_summary_length(self):
        s = make_instant_summary(
            "User asked about Zephyr flux capacitor misalignment.",
            "Root cause: inverted polarity on module 7. Fix: swap J14-J15 jumpers.",
            session_id="sess_test",
            project="KnowWhere",
            anchor_id="aid-123",
        )
        self.assertGreaterEqual(len(s), 300)
        self.assertLessEqual(len(s), 500)
        self.assertIn("aid-123", s)


class TestKnowWhereDBOutcome(unittest.TestCase):
    def setUp(self):
        self.psycopg2_patch = patch("knowwhere_db.psycopg2")
        self.mock_psycopg2 = self.psycopg2_patch.start()
        self.mock_psycopg2.extras.DictCursor = MagicMock(name="DictCursor")
        self.mock_psycopg2.extras.Json = MagicMock(side_effect=lambda x: x)
        self.register_vector_patch = patch("knowwhere_db.register_vector")
        self.register_vector_patch.start()
        self.mock_conn = MagicMock()
        self.mock_conn.closed = False
        self.mock_cursor = MagicMock()
        self.mock_psycopg2.connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

        from knowwhere_db import KnowWhereDB

        self.db = KnowWhereDB(db_url="postgresql://test:test@localhost/test")

    def tearDown(self):
        self.psycopg2_patch.stop()
        self.register_vector_patch.stop()

    def test_upsert_summary_with_anchor_id(self):
        self.mock_cursor.fetchone.return_value = ["sum-1"]
        emb = np.array([0.1] * 256, dtype=np.float32)
        self.db.upsert_summary(
            "sess_a", "KnowWhere", "summary", embedding=emb, anchor_id="src-1"
        )
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertIn("anchor_id", sql)
        params = self.mock_cursor.execute.call_args[0][1]
        self.assertEqual(params[5], "src-1")

    def test_search_relevant_requires_min_score(self):
        self.mock_cursor.fetchall.return_value = []
        emb = np.array([0.2] * 256, dtype=np.float32)
        self.db.search_relevant(emb, min_score=0.35)
        sql = self.mock_cursor.execute.call_args[0][0]
        self.assertNotIn("debut_seen = FALSE", sql)
        self.assertIn("embedding IS NOT NULL", sql)

    def test_recall_deep_by_anchor(self):
        self.mock_cursor.fetchone.return_value = {
            "id": "aid-1",
            "session_id": "sess_x",
            "full_text": "original verbatim",
            "char_count": 18,
            "created_at": "2026-07-10",
        }
        with patch.object(self.db, "get_source_by_anchor") as mock_get:
            mock_get.return_value = dict(self.mock_cursor.fetchone.return_value)
            out = self.db.recall_deep(anchor_id="aid-1")
        self.assertTrue(out["found"])
        self.assertEqual(out["full_text"], "original verbatim")

    def test_cleanup_fixture_prefix(self):
        self.mock_cursor.fetchall.side_effect = [[("s1",)], [("x1",)]]
        self.mock_cursor.fetchone.side_effect = None
        # DELETE ... RETURNING id
        self.mock_cursor.fetchall = MagicMock(side_effect=[[("s1",)], [("x1",)]])
        result = self.db.cleanup_fixture_prefix("kw_outcome_eval_")
        self.assertEqual(result["summaries_deleted"], 1)
        self.assertEqual(result["sources_deleted"], 1)


class TestPluginHooks(unittest.TestCase):
    def test_pre_llm_turn_gate(self):
        provider = _load_plugin_provider()
        provider._enabled = True
        provider._initialized = True

        with patch.object(provider, "_build_injection", return_value="CTX") as mock_inj:
            r1 = provider._hook_pre_llm_call(session_id="s1", user_message="hello")
            r2 = provider._hook_pre_llm_call(session_id="s1", user_message="hello again")
        self.assertEqual(r1, {"context": "CTX"})
        self.assertIsNone(r2)
        mock_inj.assert_called_once()

    def test_post_llm_resets_turn_gate(self):
        provider = _load_plugin_provider()
        provider._enabled = True
        provider._initialized = True
        provider._session_id = "s1"
        provider._pre_llm_called_this_turn = True

        with patch.object(provider, "_persist_turn"):
            provider._hook_post_llm_call(
                session_id="s1",
                user_message="u",
                assistant_response="a",
            )
        self.assertFalse(provider._pre_llm_called_this_turn)

    def test_session_reset_clears_pending(self):
        provider = _load_plugin_provider()
        provider._recent_user_msgs = ["a"]
        provider._pending_turns = [{"user": "u", "assistant": "a"}]
        provider._hook_on_session_reset(session_id="new")
        self.assertEqual(provider._recent_user_msgs, [])
        self.assertEqual(provider._pending_turns, [])


class TestPluginRecallHandler(unittest.TestCase):
    def test_kw_recall_returns_json(self):
        init_path = ROOT / "hermes-plugin" / "knowwhere" / "__init__.py"
        spec = importlib.util.spec_from_file_location("knowwhere_plugin_test", init_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_db = MagicMock()
        mock_db.recall_deep.return_value = {"found": True, "full_text": "verbatim"}
        mock_db.close = MagicMock()

        with patch.object(mod, "_fresh_db", return_value=mock_db):
            p = mod.KnowWhereProvider()
            p._enabled = True
            p._initialized = True
            out = p._handle_kw_recall(anchor_id="aid-1")

        data = json.loads(out)
        self.assertTrue(data["found"])
        self.assertEqual(data["full_text"], "verbatim")


if __name__ == "__main__":
    unittest.main(verbosity=2)
