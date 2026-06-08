"""Simple lifecycle hook primitives for agent runs.

Usage::

    class LoggingHook(AgentHook):
        async def before_iteration(self, ctx):
            print(f"[iter {ctx.iteration}] {len(ctx.messages)} messages")

    result = await runner.run(spec, hooks=[LoggingHook()])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from myAgent.providers.provider import ToolCall


@dataclass(slots=True)
class AgentHookContext:
    """Per-iteration state snapshot exposed to hooks.

    The runner mutates ``messages`` in place across iterations, so
    hooks should treat the list as a live view, not a detached copy.
    """

    iteration: int
    messages: list[dict[str, Any]]
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    final_content: str | None = None
    error: str | None = None
    stop_reason: str | None = None


class AgentHook:
    """Minimal lifecycle surface for runner customization.

    Every method is a no-op by default; override only the callbacks
    you need.  All async methods are called with ``await``.
    """

    # -- Run-level -------------------------------------------------------

    async def before_run(self, messages: list[dict[str, Any]]) -> None:
        """Called once before the agent loop starts."""

    async def after_run(
        self,
        messages: list[dict[str, Any]],
        final_content: str | None,
        error: str | None,
    ) -> None:
        """Called once after the loop finishes (success or error)."""

    async def on_error(self, error: str) -> None:
        """Called when an unhandled exception bubbles out of the loop."""

    # -- Iteration-level -------------------------------------------------

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        """Called at the top of every loop iteration."""

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        """Called at the bottom of every loop iteration."""

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        """Called after the LLM returns tool calls but before they execute."""

    # -- Streaming -------------------------------------------------------

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        """Called for each text token during streaming."""

    async def on_stream_end(self, ctx: AgentHookContext) -> None:
        """Called when the streaming response is complete."""

    def wants_streaming(self) -> bool:
        """Return True if this hook needs per-token ``on_stream`` callbacks."""
        return False


class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks.

    Error isolation: async methods catch and log per-hook exceptions
    so a faulty hook cannot crash the agent loop.
    """

    def __init__(self, hooks: list[AgentHook]) -> None:
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    async def _for_each(self, method: str, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await getattr(h, method)(*args, **kwargs)
            except Exception:
                logger.exception("Hook.{}(…) error in {}", method, type(h).__name__)

    async def before_run(self, messages: list[dict[str, Any]]) -> None:
        await self._for_each("before_run", messages)

    async def after_run(
        self,
        messages: list[dict[str, Any]],
        final_content: str | None,
        error: str | None,
    ) -> None:
        await self._for_each("after_run", messages, final_content, error)

    async def on_error(self, error: str) -> None:
        await self._for_each("on_error", error)

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        await self._for_each("before_iteration", ctx)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        await self._for_each("after_iteration", ctx)

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        await self._for_each("before_execute_tools", ctx)

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        await self._for_each("on_stream", ctx, delta)

    async def on_stream_end(self, ctx: AgentHookContext) -> None:
        await self._for_each("on_stream_end", ctx)
