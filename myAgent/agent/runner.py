import asyncio
from dataclasses import dataclass, field
from typing import Any

from myAgent.agent.tools.read_file_tool import ReadFile
from myAgent.agent.tools.registry import ToolRegistry
from myAgent.providers.provider import LLMProvider, ToolCall
from myAgent.session.manager import Session


@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str, Any]]
    session: Session
    max_iterations: int
    concurrency_enabled: bool = False

@dataclass(slots=True)
class AgentRunResult:
    messages: list[dict[str, Any]]
    final_content: str | None
    error: str | None
    tools_used: list[str] = field(default_factory=list)


class AgentRunner:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.tools = ToolRegistry()
        self.tools.register(ReadFile())
        self.tool_spec = self.tools.tool_spec

    # -- concurrency control: partition + concurrency -------------------------------

    def _partition_tool_calls(self, tool_calls: list[ToolCall]) -> list[list[ToolCall] | ToolCall]:
        '''parititon tool_calls

        rule: The adjacent concurrency_safe tools can be executed in parallel if they belong to the same batch;
        Each non-safety tool occupies a batch exclusively and executes serially.
        The batches maintain their original order.

        example:
          [read(a), read(b), write(c), read(d)]
          → [[read(a), read(b)], [write(c)], [read(d)]]

        '''
        batches = []
        current_batch = []

        for tc in tool_calls:
            tool = self.tools.get(tc.name)
            can_batch = tool is not None and tool.concurrency_safe

            if can_batch:
                current_batch.append(tc)
            else:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([tc])

        if current_batch:
            batches.append(current_batch)
        return batches

    async def _run_tool(self, tc: ToolCall) -> tuple[ToolCall, Any]:
        '''execute single tool_call, return (tool_call, result_str)'''
        result = await self.tools.execute(tc.name, tc.arguments)
        return (tc, result)

    async def _execute_batch(self, batch: list[ToolCall] | ToolCall,
                             run_result: AgentRunResult, spec: AgentRunSpec):
        '''execute a batch of tool_calls。

        above one -> asyncio.gather parallel
        only one -> await
        '''
        if isinstance(batch, ToolCall):
            results = [await self._run_tool(batch)]
        else:
            results = await asyncio.gather(
                *(self._run_tool(tc) for tc in batch)
            )

        for tc, result in results:
            run_result.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result,
            })
            spec.session.add_message(
                "tool", result,
                tool_call_id=tc.id,
                name=tc.name,
            )

    # --------- MAIN LOOP ------------------------------------------------------

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        run_result = AgentRunResult(
            messages=list(spec.initial_messages),
            final_content=None,
            error=None,
        )

        for _ in range(spec.max_iterations):
            response = await self.provider.chat(run_result.messages, self.tool_spec)

            if response.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [tc.to_dict() for tc in response.tool_calls],
                }
                run_result.messages.append(assistant_msg)
                spec.session.add_message(
                    assistant_msg["role"], assistant_msg["content"],
                    tool_calls=assistant_msg["tool_calls"],
                )
                run_result.tools_used.extend(tc.name for tc in response.tool_calls)

                if spec.concurrency_enabled:
                    batches = self._partition_tool_calls(response.tool_calls)
                else:
                    batches = response.tool_calls

                for batch in batches:
                    await self._execute_batch(batch, run_result, spec)

                continue

            run_result.final_content = response.content
            return run_result

        run_result.error = "Max iterations exceeded"
        return run_result
