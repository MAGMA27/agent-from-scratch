# Please install OpenAI SDK first: `pip3 install openai`
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

load_dotenv()
api_key = os.getenv("API_KEY")

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_openai_tool_calls(
        openai_tool_calls: Optional[List[ChatCompletionMessageToolCall]]
    ) -> List[ToolCall]:
        """将 OpenAI SDK 的 tool_calls 转换为自定义 ToolCall 列表"""
        if not openai_tool_calls:
            return []

        parsed = []
        for tc in openai_tool_calls:
            # 解析 arguments（JSON 字符串 -> dict）
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

    async def chat(self, msg: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=msg,
            stream=False,
            tools=tools,
            extra_body={"thinking": {"type": "disabled"}},
        )

        message = response.choices[0].message
        content = message.content
        tool_calls = parse_openai_tool_calls(message.tool_calls)

        return LLMResponse(content=content,
                           tool_calls=tool_calls)
