# Please install OpenAI SDK first: `pip3 install openai`
import json
import os
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


class LLMProvider:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

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
