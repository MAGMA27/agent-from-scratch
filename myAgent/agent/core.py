import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from loguru import logger

from myAgent.agent.memory import Consolidator, MemoryStore, get_memory_context
from myAgent.agent.runner import AgentRunSpec
from myAgent.agent.skills import SkillLoader
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
    # Streaming: async callback that receives each token as it's generated.
    # Passed through from CLI -> core -> runner -> provider.
    on_delta: "Any | None" = None


class AgentCore:
    def __init__(self, bus, runner, *,
                 consolidator: Consolidator,
                 memory_store: MemoryStore,
                 skill_sys: SkillLoader):
        self.bus = bus
        self.runner = runner
        self.consolidator = consolidator
        self.memory_store = memory_store
        self.skill_sys = skill_sys
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._pending_queues: dict[str, asyncio.Queue] = {}

    async def handle_message(
        self, msg: InboundMessage,
        session_manager: SessionManager,
        session_key: str,
        on_delta=None,
    ) -> OutboundMessage | None:
        """Entry point: maintain a turn or create a new one.

        ``on_delta`` is an optional async callback(str) for streaming output.
        """
        if session_key in self._pending_queues:
            try:
                self._pending_queues[session_key].put_nowait(msg)
            except asyncio.QueueFull:
                pass
            return None

        return await self.process_message(
            msg, session_manager, session_key, on_delta=on_delta,
        )

    async def process_message(
        self, msg: InboundMessage,
        session_manager: SessionManager,
        session_key: str,
        *,
        on_delta=None,
    ) -> OutboundMessage | None:
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending

        try:
            async with lock:
                session = session_manager.get_or_create(session_key)
                ctx = TurnContext(
                    msg=msg,
                    session=session,
                    session_manager=session_manager,
                    on_delta=on_delta,
                )

                await self._run_turn(ctx)

                while True:
                    try:
                        next_msg = pending.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    ctx.msg = next_msg
                    ctx.state = TurnState.RESTORE
                    ctx.history_messages = []
                    ctx.all_messages = []
                    ctx.final_content = None
                    await self._run_turn(ctx)

            return ctx.outbound

        finally:
            self._pending_queues.pop(session_key, None)

    async def _compact_if_needed(self, session: Session) -> None:
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
        system = SYSTEM_PROMPT
        if self.memory_store and ctx.session:
            mem = get_memory_context(self.memory_store, session_key=ctx.session.key)

        if self.skill_sys:
            skills_prompt = {"role": "system",
                "content": "This is used for progressive loading - the agent can read the full"
                "skill content using read_file when needed." +
                self.skill_sys.build_skills_summary()
                }

            logger.debug("Skills prompt built: {}", skills_prompt.get("content", "")[:200])

        ctx.all_messages = [
            {"role": "system", "content": system},
            skills_prompt,
            *mem,
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
            concurrency_enabled=True,
            on_delta=ctx.on_delta,  # <-- streaming callback passed through
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
