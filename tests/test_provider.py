"""Unit tests for ToolCall, LLMResponse, and parse_openai_tool_calls.

These do NOT make any network calls — only data model logic.
"""

import json

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from myAgent.providers.provider import (
    LLMResponse,
    ToolCall,
    parse_openai_tool_calls,
)


class TestToolCall:
    def test_to_dict_serializes_arguments_as_json(self):
        tc = ToolCall(
            id="call_123",
            type="function",
            name="get_weather",
            arguments={"city": "Beijing"},
        )
        d = tc.to_dict()
        assert d["id"] == "call_123"
        assert d["type"] == "function"
        assert d["function"]["name"] == "get_weather"
        # arguments field should be a JSON string inside the dict
        parsed = json.loads(d["function"]["arguments"])
        assert parsed == {"city": "Beijing"}

    def test_to_dict_empty_arguments(self):
        tc = ToolCall(id="c1", type="function", name="ping", arguments={})
        d = tc.to_dict()
        assert json.loads(d["function"]["arguments"]) == {}


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.tool_calls == []

    def test_with_tool_calls(self):
        tc = ToolCall(id="tc1", type="function", name="f", arguments={})
        r = LLMResponse(content=None, tool_calls=[tc])
        assert r.content is None
        assert len(r.tool_calls) == 1

    def test_content_can_be_none(self):
        r = LLMResponse(content=None)
        assert r.content is None


class TestParseOpenAIToolCalls:
    def make_openai_tc(self, id_: str, name: str, args: dict) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=id_,
            type="function",
            function=Function(name=name, arguments=json.dumps(args)),
        )

    def test_none_returns_empty_list(self):
        assert parse_openai_tool_calls(None) == []

    def test_empty_list_returns_empty(self):
        assert parse_openai_tool_calls([]) == []

    def test_single_tool_call(self):
        raw = [self.make_openai_tc("id1", "get_weather", {"city": "Shanghai"})]
        result = parse_openai_tool_calls(raw)
        assert len(result) == 1
        assert result[0].id == "id1"
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"city": "Shanghai"}
        assert result[0].type == "function"

    def test_multiple_tool_calls(self):
        raw = [
            self.make_openai_tc("a", "f1", {"x": 1}),
            self.make_openai_tc("b", "f2", {"y": 2}),
        ]
        result = parse_openai_tool_calls(raw)
        assert len(result) == 2
        assert result[0].name == "f1"
        assert result[1].name == "f2"
