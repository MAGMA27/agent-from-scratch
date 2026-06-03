"""Unit tests for AgentRunner using a mocked LLM provider."""

import pytest

from myAgent.agent.runner import AgentRunResult, AgentRunSpec, AgentRunner
from myAgent.providers.provider import LLMResponse, ToolCall
from myAgent.session.manager import Session


@pytest.fixture
def session():
    return Session(key="runner-test")


@pytest.fixture
def empty_tools():
    return []


@pytest.fixture
def weather_tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


class TestToolRegistration:
    def test_register_and_execute(self, mock_provider):
        runner = AgentRunner(mock_provider)
        runner.register_tool("echo", lambda **kw: kw.get("text", ""))

        result = runner._tool_handlers["echo"](text="hi")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_execute_registered_tool_async(self, mock_provider):
        runner = AgentRunner(mock_provider)

        async def double(x: int) -> int:
            return x * 2

        runner.register_tool("double", double)
        out = await runner.execute_tool("double", {"x": 3})
        assert out == "6"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_provider):
        runner = AgentRunner(mock_provider)
        out = await runner.execute_tool("no_such_tool", {})
        assert "Error" in out
        assert "unknown tool" in out

    @pytest.mark.asyncio
    async def test_execute_tool_handler_raises(self, mock_provider):
        runner = AgentRunner(mock_provider)

        async def bad_tool(**kw):
            raise ValueError("boom")

        runner.register_tool("bad", bad_tool)
        out = await runner.execute_tool("bad", {})
        assert "Error" in out
        assert "ValueError" in out
        assert "boom" in out


class TestAgentRunSimple:
    @pytest.mark.asyncio
    async def test_single_turn_no_tools(self, mock_provider, session, empty_tools):
        """Agent gets a text-only response on the first turn."""
        mock_provider.chat.return_value = LLMResponse(content="Hello back")

        runner = AgentRunner(mock_provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            session=session,
            tools=empty_tools,
            max_iterations=5,
        )

        result = await runner.run(spec)
        assert isinstance(result, AgentRunResult)
        assert result.final_content == "Hello back"
        assert result.error is None
        assert result.tools_used == []

    @pytest.mark.asyncio
    async def test_tool_call_loop(self, mock_provider, session, weather_tools):
        """Agent calls a tool, receives the result, then generates final text."""
        tc = ToolCall(id="call_1", type="function", name="get_weather",
                      arguments={"city": "Beijing"})

        # First call returns a tool call; second returns final content
        mock_provider.chat.side_effect = [
            LLMResponse(content=None, tool_calls=[tc]),
            LLMResponse(content="The weather is sunny"),
        ]

        runner = AgentRunner(mock_provider)

        async def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        runner.register_tool("get_weather", get_weather)

        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Weather?"}],
            session=session,
            tools=weather_tools,
            max_iterations=10,
        )

        result = await runner.run(spec)
        assert result.final_content == "The weather is sunny"
        assert "get_weather" in result.tools_used

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, mock_provider, session, empty_tools):
        """Agent that keeps calling tools forever hits max_iterations."""
        tc = ToolCall(id="loop", type="function", name="loop",
                      arguments={})
        # Always return a tool call — never final content
        mock_provider.chat.return_value = LLMResponse(
            content=None, tool_calls=[tc]
        )

        runner = AgentRunner(mock_provider)
        runner.register_tool("loop", lambda **kw: "looping")

        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "start"}],
            session=session,
            tools=empty_tools,
            max_iterations=3,
        )

        result = await runner.run(spec)
        assert result.final_content is None
        assert result.error == "Max iterations exceeded"
