import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from myAgent.agent.memory import Consolidator, MemoryStore, get_memory_context
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

    history_messages: list[dict[str, Any]] = field(default_factory=list)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    session: Session | None = None
    session_manager: SessionManager | None = None
    final_content: str | None = None
    outbound: OutboundMessage | None = None


class AgentCore:
    def __init__(self, bus, runner, *,
                 consolidator: Consolidator | None = None,
                 memory_store: MemoryStore | None = None):
        self.bus = bus
        self.runner = runner
        self.consolidator = consolidator
        self.memory_store = memory_store
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._pending_queues: dict[str, asyncio.Queue] = {}

    async def handle_message(
        self, msg: InboundMessage,
        session_manager: SessionManager,
        session_key: str,
    ) -> OutboundMessage | None:
        """entry piont: maintain a turn or create a new one"""
        # exist a turn
        if session_key in self._pending_queues:
            try:
                self._pending_queues[session_key].put_nowait(msg)
            except asyncio.QueueFull:
                pass
            return None

        # new one
        return await self.process_message(msg, session_manager, session_key)

    async def process_message(self, msg: InboundMessage, session_manager: SessionManager, session_key: str) -> OutboundMessage | None:
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())

        # registration
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending

        try:
            async with lock:
                session = session_manager.get_or_create(session_key)
                ctx = TurnContext(msg=msg, session=session, session_manager=session_manager)

                # process the first msg
                await self._run_turn(ctx)

                # query if the queue is empty
                while True:
                    try:
                        next_msg = pending.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    # not empty
                    ctx.msg = next_msg
                    ctx.state = TurnState.RESTORE
                    ctx.histories = []
                    ctx.all_messages = []
                    ctx.final_content = None
                    await self._run_turn(ctx)

            return ctx.outbound

        finally:
            # logout queue
            self._pending_queues.pop(session_key, None)

    async def _compact_if_needed(self, session: Session) -> None:
        """Run consolidation before building context."""
        if self.consolidator is None or not session.messages:
            return
        await self.consolidator.compact(session)

    async def _run_turn(self, ctx: TurnContext) -> None:
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

    async def _state_restore(self, ctx: TurnContext) -> str:
        if ctx.session:
            await self._compact_if_needed(ctx.session)
            ctx.history_messages = ctx.session.get_history()
        return "ok"

    async def _state_build(self, ctx: TurnContext) -> str:
        # Build system prompt with memory context injected.
        system = SYSTEM_PROMPT
        if self.memory_store and ctx.session:
            mem = get_memory_context(self.memory_store, ctx.session)
            if mem:
                system = f"{system}\n\n{mem}" if system else mem

        ctx.all_messages = [
            {"role": "system", "content": system},
            *ctx.history_messages,
            {"role": "user", "content": ctx.msg.content},
        ]
        ctx.session.add_message("user", ctx.msg.content)
        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        spec = AgentRunSpec(
            initial_messages=ctx.all_messages,
            session=ctx.session,
            max_iterations=25,
            concurrency_enabled=True
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
