from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from myAgent.agent.runner import AgentRunSpec
from myAgent.bus.bus import InboundMessage, OutboundMessage
from myAgent.session.manager import Session, SessionManager

SYSTEM_PROMPT = ""

class TurnState(Enum):
    RESTORE = auto()
    BUILD   = auto()
    RUN     = auto()
    SAVE    = auto()
    RESPOND = auto()
    DONE    = auto()


_TRANSITIONS = {
    (TurnState.RESTORE, "ok"): TurnState.BUILD,
    (TurnState.BUILD,   "ok"): TurnState.RUN,
    (TurnState.RUN,     "ok"): TurnState.SAVE,
    (TurnState.SAVE,    "ok"): TurnState.RESPOND,
    (TurnState.RESPOND, "ok"): TurnState.DONE,
}


@dataclass
class TurnContext:
    msg: InboundMessage
    state: TurnState = TurnState.RESTORE

    histories: list[dict[str, Any]] = field(default_factory=list)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    session: Session | None = None
    session_manager: SessionManager | None = None
    final_content: str | None = None
    outbound: OutboundMessage | None = None


class AgentCore:
    def __init__(self, bus, runner, tools):
        self.bus = bus
        self.runner = runner
        self.tools = tools

    async def process_message(self, msg: InboundMessage, session_manager: SessionManager, session_key: str) -> OutboundMessage | None:
        session = session_manager.get_or_create(session_key)
        ctx = TurnContext(msg=msg, session=session, session_manager=session_manager)

        while ctx.state != TurnState.DONE:
            handler = {
                TurnState.RESTORE: self._state_restore,
                TurnState.BUILD:   self._state_build,
                TurnState.RUN:     self._state_run,
                TurnState.SAVE:    self._state_save,
                TurnState.RESPOND: self._state_respond,
            }[ctx.state]

            event = await handler(ctx)
            next_state = _TRANSITIONS.get((ctx.state, event))
            if next_state is None:
                raise RuntimeError(f"No transition from {ctx.state} on {event!r}")
            ctx.state = next_state

        return ctx.outbound

    async def _state_restore(self, ctx: TurnContext) -> str:
        if ctx.session:
            ctx.histories = ctx.session.get_history()
        return "ok"

    async def _state_build(self, ctx: TurnContext) -> str:
        ctx.all_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *ctx.histories,
            {"role": "user", "content": ctx.msg.content},
        ]
        ctx.session.add_message("user", ctx.msg.content)
        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        spec = AgentRunSpec(
            initial_messages=ctx.all_messages,
            session=ctx.session,
            tools=self.tools,
            max_iterations=25,
        )
        result = await self.runner.run(spec)
        ctx.final_content = result.final_content
        ctx.session.add_message(
            'assistant', ctx.final_content
        )
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        await ctx.session_manager.save(ctx.session)
        return "ok"

    async def _state_respond(self, ctx: TurnContext) -> str:
        ctx.outbound = OutboundMessage(
            content=ctx.final_content or "",
        )
        return "ok"
