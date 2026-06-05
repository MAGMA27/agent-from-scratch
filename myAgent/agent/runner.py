import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from myAgent.agent.tools.loader import ToolLoader
from myAgent.agent.tools.registry import ToolRegistry
from myAgent.providers.provider import LLMProvider, OnDelta, ToolCall
from myAgent.session.manager import Session


@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str, Any]]
    session: Session
    max_iterations: int
    concurrency_enabled: bool = False
    # Streaming: if set, the runner calls provider.chat_stream(on_delta=...)
    # instead of provider.chat().  Each token fires this callback immediately.
    on_delta: OnDelta | None = None

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
        loader = ToolLoader()
        loader.load(ctx=None, registry=self.tools, scope="core")
        self.tool_spec = self.tools.tool_spec

    # -- concurrency control: partition + concurrency -------------------------------

    def _partition_tool_calls(self, tool_calls: list[ToolCall]) -> list[list[ToolCall] | ToolCall]:
        """Partition tool_calls into parallel-safe batches.

        Rule: adjacent concurrency_safe tools execute together in one batch;
        each non-safe tool occupies its own batch and runs serially.
        Batches maintain original order.

        Example:
          [read(a), read(b), write(c), read(d)]
          -> [[read(a), read(b)], [write(c)], [read(d)]]
        """
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
        """Execute single tool call, return (tool_call, result_str)."""
        result = await self.tools.execute(tc.name, tc.arguments)
        return (tc, result)

    async def _execute_batch(self, batch: list[ToolCall] | ToolCall,
                             run_result: AgentRunResult, spec: AgentRunSpec):
        """Execute a batch of tool_calls.

        Multiple calls -> asyncio.gather in parallel.
        Single call   -> await.
        """
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

    # ------------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------------

    async def _call_llm(self, messages: list[dict[str, Any]], spec: AgentRunSpec):
        """Call the LLM -- streaming if spec.on_delta is set, otherwise regular."""
        if spec.on_delta is not None:
            return await self.provider.chat_stream(
                messages, self.tool_spec,
                on_delta=spec.on_delta,
            )
        return await self.provider.chat(messages, self.tool_spec)

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        run_result = AgentRunResult(
            messages=list(spec.initial_messages),
            final_content=None,
            error=None,
        )

        for iteration in range(spec.max_iterations):
            logger.debug("Runner iteration {}/{}", iteration + 1, spec.max_iterations)
            response = await self._call_llm(run_result.messages, spec)

            if response.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [tc.to_dict() for tc in response.tool_calls],
                }
                logger.info("Tool calls: {}", [tc.name for tc in response.tool_calls])
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
            logger.info("Run complete, final content length: {}", len(run_result.final_content or ""))
            return run_result

        run_result.error = "Max iterations exceeded"
        logger.warning("Max iterations ({}) exceeded", spec.max_iterations)
        return run_result
