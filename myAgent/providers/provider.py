# Please install OpenAI SDK first: `pip3 install openai`
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, List, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

load_dotenv()
api_key = os.getenv("API_KEY")


@dataclass
class ToolCall:
    id: str
    type: str
    name: str
    arguments: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_openai_tool_calls(
        openai_tool_calls: Optional[List[ChatCompletionMessageToolCall]]
    ) -> List[ToolCall]:
        if not openai_tool_calls:
            return []

        parsed = []
        for tc in openai_tool_calls:
            args = json.loads(tc.function.arguments)
            parsed.append(ToolCall(
                id=tc.id,
                type=tc.type,
                name=tc.function.name,
                arguments=args
            ))
        return parsed


# Reusable callback type for streaming
OnDelta = Callable[[str], Awaitable[None]]


class LLMProvider:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    # ---- Non-streaming (keep for compatibility / fallback) ----

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or "deepseek-v4-flash",
            "messages": messages,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        response = await self.client.chat.completions.create(**kwargs)

        logger.debug("LLM call: model={}, messages={}, tools={}",
                     kwargs.get("model"), len(messages), len(tools or []))
        message = response.choices[0].message
        content = message.content
        tool_calls = parse_openai_tool_calls(message.tool_calls)

        return LLMResponse(content=content,
                           tool_calls=tool_calls)

    # ---- Streaming (the new hotness) ----

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        on_delta: OnDelta | None = None,
    ) -> LLMResponse:
        """Stream a chat completion, calling ``on_delta`` for each text token.

        Returns the same ``LLMResponse`` as :meth:`chat`, but fires
        ``on_delta`` incrementally as the model generates tokens.  This is the
        nanobot-style streaming pattern: the provider yields each token via a
        callback, and the caller (runner / CLI) renders it in real time.
        """
        kwargs: dict[str, Any] = {
            "model": model or "deepseek-v4-flash",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})

        logger.debug("LLM stream: model={}, messages={}, tools={}",
                     kwargs.get("model"), len(messages), len(tools or []))

        stream = await self.client.chat.completions.create(**kwargs)

        # ---- Accumulators for the final structured response ----
        content_parts: list[str] = []
        finish_reason: str | None = None
        tc_bufs: dict[int, dict[str, str]] = {}

        stream_iter = stream.__aiter__()
        idle_timeout_s = 90

        while True:
            try:
                chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=idle_timeout_s,
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning("Stream stalled after {}s", idle_timeout_s)
                return LLMResponse(content="[Error: stream stalled]")

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            # ---- content delta → fire callback immediately ----
            if delta.content:
                content_parts.append(delta.content)
                if on_delta:
                    await on_delta(delta.content)

            # ---- tool-call delta → accumulate incrementally ----
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index
                if idx not in tc_bufs:
                    tc_bufs[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tc_bufs[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tc_bufs[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tc_bufs[idx]["arguments"] += tc.function.arguments

        # ---- Build final response ----
        full_content = "".join(content_parts)

        tool_calls: list[ToolCall] = []
        for buf in sorted(tc_bufs.values(), key=lambda b: b.get("id", "")):
            try:
                args = json.loads(buf["arguments"]) if buf["arguments"].strip() else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=buf["id"],
                type="function",
                name=buf["name"],
                arguments=args,
            ))

        return LLMResponse(
            content=full_content or None,
            tool_calls=tool_calls,
        )
