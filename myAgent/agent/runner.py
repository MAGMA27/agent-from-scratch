from dataclasses import dataclass, field
from typing import Any

from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import Session


@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str, Any]]
    session: Session
    tools: list[dict[str, Any]]
    max_iterations: int


@dataclass(slots=True)
class AgentRunResult:
    messages: list[dict[str, Any]]
    final_content: str | None
    error: str | None
    tools_used: list[str] = field(default_factory=list)


class AgentRunner():
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        # 一个简单的工具注册表: name -> callable
        self._tool_handlers: dict[str, callable] = {}

    def register_tool(self, name: str, handler: callable) -> None:
        self._tool_handlers[name] = handler

    async def execute_tool(self, name: str, arguments: dict) -> str:
        handler = self._tool_handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = await handler(**arguments)
            return str(result) if result is not None else "(empty)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        run_result = AgentRunResult(
            messages=list(spec.initial_messages),
            final_content=None,
            error=None,
        )

        for _ in range(spec.max_iterations):
            response = await self.provider.chat(run_result.messages, spec.tools)

            if response.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [tc.to_dict() for tc in response.tool_calls],
                }
                run_result.messages.append(assistant_msg)
                spec.session.add_message(assistant_msg["role"], assistant_msg["content"], tool_calls=assistant_msg["tool_calls"])
                run_result.tools_used.extend(tc.name for tc in response.tool_calls)

                for tc in response.tool_calls:
                    result = await self.execute_tool(tc.name, tc.arguments)
                    # print(tc.name, tc.arguments)
                    run_result.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    })
                    spec.session.add_message(
                        "tool" ,
                        result,
                        tool_call_id=tc.id,
                        name=tc.name
                    )

                continue

            run_result.final_content = response.content
            return run_result

        run_result.error = "Max iterations exceeded"
        return run_result
