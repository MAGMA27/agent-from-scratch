from dataclasses import dataclass, field
from enum import Enum, auto

from myAgent.agent.runner import AgentRunSpec
from myAgent.bus.bus import InboundMessage, OutboundMessage

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
    session: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    final_content: str | None = None
    outbound: OutboundMessage | None = None


class AgentCore:
    def __init__(self, bus, runner, tools):
        self.bus = bus
        self.runner = runner
        self.tools = tools

    async def process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        ctx = TurnContext(msg=msg)

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
        ctx.session = ctx.msg.session
        return "ok"

    async def _state_build(self, ctx: TurnContext) -> str:
        ctx.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *ctx.session,
            {"role": "user", "content": ctx.msg.content},
        ]
        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        spec = AgentRunSpec(
            initial_messages=ctx.messages,
            tools=self.tools,
            max_iterations=25,
        )
        result = await self.runner.run(spec)
        ctx.final_content = result.final_content
        ctx.messages = result.messages
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        new_messages = ctx.messages[len(ctx.session) + 1:]
        ctx.session.extend(new_messages)
        # await save_session(ctx.msg.session, ctx.session)
        return "ok"

    async def _state_respond(self, ctx: TurnContext) -> str:
        ctx.outbound = OutboundMessage(
            content=ctx.final_content or "",
            session=ctx.session
        )
        return "ok"
