import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from myAgent.session.manager import Session, SessionManager


@pytest.fixture
def temp_workspace():
    """Temporary workspace directory for SessionManager tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session_manager(temp_workspace):
    """Fresh SessionManager with temp workspace."""
    return SessionManager(temp_workspace)


@pytest.fixture
def sample_session():
    """A session with a few preloaded messages."""
    s = Session(key="test-session")
    s.add_message("user", "Hello")
    s.add_message("assistant", "Hi there!")
    s.add_message("user", "What is the weather?")
    s.add_message("assistant", "Let me check.",
                  tool_calls=[{"id": "tc1", "type": "function",
                               "function": {"name": "get_weather",
                                            "arguments": '{"city": "Beijing"}'}}])
    s.add_message("tool", "Sunny, 22°C", tool_call_id="tc1", name="get_weather")
    s.add_message("assistant", "The weather in Beijing is sunny, 22°C.")
    return s


@pytest.fixture
def mock_provider():
    """Mock LLMProvider that returns a configurable response."""
    provider = MagicMock()
    provider.chat = AsyncMock()
    return provider
