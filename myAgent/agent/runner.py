from myAgent.providers.provider import LLMProvider
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str, Any]]
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

    async def execute_tool(self):
        pass

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        run_result = AgentRunResult(messages=spec.initial_messages, final_content=None, error=None)

        for _ in range(20):
            response = await self.provider.chat(run_result.messages, spec.tools)

            if response.tool_calls:
                msg = {"role": "assistant", "content": response.content,
                       "tool_calls": [tc.to_dict() for tc in response.tool_calls]}
                run_result.messages.append(msg)

                for tc in response.tool_calls:
                    result = await self.execute_tool(tc.name, tc.arguments)
                    run_result.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            run_result.final_content = response.content
            return run_result

        run_result.error = "Max iterations exceeded"
        return run_result
