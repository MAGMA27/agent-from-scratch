"""Unit tests for Session data model and SessionManager persistence."""

from datetime import datetime

import pytest

from myAgent.session.manager import Session, find_legal_message_start

# ── Session data model ────────────────────────────────────────────


class TestSession:
    def test_created_with_defaults(self):
        s = Session(key="abc")
        assert s.key == "abc"
        assert s.messages == []
        assert isinstance(s.created_at, datetime)

    def test_add_message_updates_attributes(self):
        s = Session(key="x")
        s.add_message("user", "Hello")
        assert len(s.messages) == 1
        msg = s.messages[0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        assert "timestamp" in msg

    def test_add_message_with_extra_kwargs(self):
        s = Session(key="x")
        s.add_message("tool", "result", tool_call_id="tc1", name="get_weather")
        msg = s.messages[0]
        assert msg["tool_call_id"] == "tc1"
        assert msg["name"] == "get_weather"

    def test_updated_at_changes_on_add(self):
        s = Session(key="x")
        ts1 = s.updated_at
        s.add_message("user", "msg")
        assert s.updated_at > ts1

    def test_get_history_slices_recent(self, sample_session):
        hist = sample_session.get_history(max_messages=2)
        # get_history drops orphan tool results at the front after slicing,
        # so with max_messages=2 we get only the trailing assistant message.
        assert len(hist) == 1
        assert hist[0]["role"] == "assistant"

    def test_get_history_skips_orphan_tool_at_front(self):
        s = Session(key="orphan-test")
        # Simulate orphan tool result (no matching assistant tool_call)
        s.add_message("tool", "orphan result", tool_call_id="missing_tc")
        s.add_message("user", "real message")
        hist = s.get_history()
        assert hist[0]["role"] == "user"

    def test_get_history_starts_at_user_after_channel_delivery(self):
        s = Session(key="cd-test")
        s.add_message("assistant", "bot msg", _channel_delivery=True)
        s.add_message("user", "reply")
        hist = s.get_history()
        # Should start at the assistant with delivery flag
        assert hist[0]["role"] == "assistant"
        assert hist[0].get("_channel_delivery")


# ── find_legal_message_start ──────────────────────────────────────


class TestFindLegalMessageStart:
    def test_returns_0_for_clean_history(self, sample_session):
        start = find_legal_message_start(sample_session.messages)
        assert start == 0

    def test_skips_orphan_tool_results(self):
        msgs = [
            {"role": "tool", "tool_call_id": "orphan_1"},
            {"role": "tool", "tool_call_id": "orphan_2"},
            {"role": "assistant", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1"},
        ]
        assert find_legal_message_start(msgs) == 2

    def test_no_tool_calls_returns_0(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert find_legal_message_start(msgs) == 0


# ── SessionManager persistence ────────────────────────────────────


class TestSessionManager:
    def test_get_or_create_new(self, session_manager):
        s = session_manager.get_or_create("new-session")
        assert s.key == "new-session"
        assert s.messages == []

    def test_get_or_create_cached(self, session_manager):
        s1 = session_manager.get_or_create("s1")
        s2 = session_manager.get_or_create("s1")
        assert s1 is s2

    @pytest.mark.asyncio
    async def test_save_and_reload_roundtrip(self, session_manager, sample_session):
        # Save a session with messages
        await session_manager.save(sample_session)

        # Clear cache so _load is exercised
        session_manager._cache.clear()

        loaded = session_manager.get_or_create(sample_session.key)
        assert loaded.key == sample_session.key
        assert len(loaded.messages) == len(sample_session.messages)
        for orig, loaded_msg in zip(sample_session.messages, loaded.messages):
            assert loaded_msg["role"] == orig["role"]
            assert loaded_msg["content"] == orig["content"]

    @pytest.mark.asyncio
    async def test_save_preserves_metadata(self, session_manager):
        s = Session(key="meta-test", metadata={"theme": "dark"})
        await session_manager.save(s)
        session_manager._cache.clear()
        loaded = session_manager.get_or_create("meta-test")
        assert loaded.metadata == {"theme": "dark"}

    @pytest.mark.asyncio
    async def test_save_preserves_last_consolidated(self, session_manager):
        s = Session(key="lc-test", last_consolidated=42)
        await session_manager.save(s)
        session_manager._cache.clear()
        loaded = session_manager.get_or_create("lc-test")
        assert loaded.last_consolidated == 42

    @pytest.mark.asyncio
    async def test_atomic_save_no_partial_file(self, session_manager, temp_workspace):
        """The tmp file is cleaned up; only the final .jsonl exists."""
        s = Session(key="atomic-test")
        await session_manager.save(s)

        tmp_files = list(temp_workspace.rglob("*.tmp"))
        assert len(tmp_files) == 0

        final = temp_workspace / "sessions" / "atomic-test.jsonl"
        assert final.exists()
