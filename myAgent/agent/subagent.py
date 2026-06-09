"""
Subagent manager — spawn background agent tasks and inject results
back into the main conversation session.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from myAgent.agent.tools.loader import ToolLoader
from myAgent.agent.tools.registry import ToolRegistry
from myAgent.session.manager import Session

if TYPE_CHECKING:
    from myAgent.agent.runner import AgentRunner
    from myAgent.bus.bus import MessageBus

SUBAGENT_MAX_ITERATIONS = 15
SUBAGENT_SYSTEM_PROMPT = (
    "You are a subagent spawned to complete a specific task. "
    "Stay focused, work step by step, and return a concise final answer. "
    "Your result will be reported back to the main agent."
)


OnSubagentResult = Callable[[str, str], Any]
"""Callback: (session_key, announce_text) -> None; called when subagent finishes."""


@dataclass
class SubagentStatus:
    """Lightweight status tracker for one subagent."""

    task_id: str
    label: str
    task_description: str
    phase: str = "running"  # running | done | error
    error: str | None = None


class SubagentManager:
    """Manages background subagent tasks.

    Subagents reuse the same ``AgentRunner`` (and therefore the same LLM
    provider) but with a fresh, isolated ``ToolRegistry`` and a lower
    iteration cap.
    """

    def __init__(
        self,
        runner: "AgentRunner",
        bus: "MessageBus",
        workspace,
        session_manager,
        on_result: OnSubagentResult | None = None,
    ):
        self.runner = runner
        self.bus = bus
        self.workspace = workspace
        self.session_manager = session_manager
        self._on_result = on_result
        self._running: dict[str, asyncio.Task[None]] = {}
        self._statuses: dict[str, SubagentStatus] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}
        self.max_concurrent = 5

    # -- public -----------------------------------------------------------

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        session_key: str = "default",
    ) -> str:
        """Start a subagent for *task* and return a short status message."""
        running = self.get_running_count()
        if running >= self.max_concurrent:
            return (
                f"Cannot spawn subagent: at concurrency limit "
                f"({running}/{self.max_concurrent})."
            )

        task_id = uuid.uuid4().hex[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        self._statuses[task_id] = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
        )

        bg = asyncio.create_task(
            self._run(task_id, task, display_label, session_key)
        )
        self._running[task_id] = bg
        self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_fut: asyncio.Task[None]) -> None:
            self._running.pop(task_id, None)
            self._statuses.pop(task_id, None)
            ids = self._session_tasks.get(session_key)
            if ids:
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    def get_running_count(self) -> int:
        return len(self._running)

    # -- internal ---------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build a fresh tool registry for the subagent."""
        registry = ToolRegistry()
        loader = ToolLoader()
        # Use the default scope so the subagent gets all tools.
        loader.load(ctx=None, registry=registry, scope="subagents")
        return registry

    async def _run(
        self,
        task_id: str,
        task: str,
        label: str,
        session_key: str,
    ) -> None:
        """Execute the subagent and announce the result."""
        from myAgent.agent.runner import AgentRunSpec

        logger.info("Subagent [{}] starting: {}", task_id, label)

        try:
            # tools = self._build_tools()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]

            sub_session = Session(key=f"subagent:{task_id}")
            spec = AgentRunSpec(
                initial_messages=messages,
                session=sub_session,
                max_iterations=SUBAGENT_MAX_ITERATIONS,
                concurrency_enabled=True,
            )
            result = await self.runner.run(spec)

            self._statuses[task_id].phase = "done"
            final = result.final_content or "Task completed — no final response."

        except Exception as exc:
            self._statuses[task_id].phase = "error"
            self._statuses[task_id].error = str(exc)
            logger.exception("Subagent [{}] failed", task_id)
            final = f"Error: {exc}"

        announce = (
            f"[Subagent '{label}' "
            f"{'completed' if self._statuses[task_id].phase == 'done' else 'failed'}]\n\n"
            f"Result:\n{final}"
        )

        # Route the result back to the main session.
        if self._on_result:
            await self._on_result(session_key, announce)
            logger.info("Subagent [{}] result delivered to session {}", task_id, session_key)
        else:
            # Fallback: publish to the message bus so it appears on next turn.
            from myAgent.bus.bus import InboundMessage

            await self.bus.publish_inbound(InboundMessage(content=announce))
            logger.info("Subagent [{}] result published to bus (fallback)", task_id)
