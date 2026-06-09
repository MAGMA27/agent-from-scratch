"""Spawn tool — allows the agent to create background subagents."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from myAgent.agent.tools.base import Tool, tool_parameters

if TYPE_CHECKING:
    from myAgent.agent.subagent import SubagentManager

# Context var set by AgentCore so SpawnTool knows which session it's in.
_current_session_key: ContextVar[str] = ContextVar(
    "spawn_session_key", default="default"
)


def set_spawn_session_key(key: str) -> None:
    _current_session_key.set(key)


@tool_parameters({
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task for the subagent to complete. Be specific and self-contained.",
        },
        "label": {
            "type": "string",
            "description": "Optional short label for display (e.g. 'fetch-docs').",
        },
    },
    "required": ["task"],
})
class SpawnTool(Tool):
    """Tool that spawns a background subagent to handle a task independently.

    The subagent runs in the background with its own tool set.  When it
    finishes, the result is injected into the current conversation so the
    main agent can continue.
    """

    _scopes = {"core"}
    _plugin_discoverable = False  # registered manually, needs SubagentManager ref

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(manager=ctx.subagent_manager)

    # Class-level attributes (used by ToolLoader + Tool base)
    name = "spawn"
    description = (
        "Spawn a subagent to handle a task in the background. "
        "Use this for complex, multi-step, or time-consuming work that "
        "can run independently. The subagent reports back when done."
    )

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        session_key = _current_session_key.get()
        return await self._manager.spawn(
            task=task,
            label=label,
            session_key=session_key,
        )
