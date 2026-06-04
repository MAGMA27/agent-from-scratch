"""Unit tests for AgentCore — state machine, message handling, session locking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from myAgent.agent.core import AgentCore, TurnContext, TurnState
from myAgent.agent.runner import AgentRunResult
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.session.manager import SessionManager


@pytest.fixture
def mock_runner():
    """Runner that returns a canned success."""
    r = MagicMock()
    r.run = AsyncMock(return_value=AgentRunResult(
        messages=[],
        final_content="mock response",
        error=None,
    ))
    return r


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def empty_tools():
    return []


@pytest.fixture
def core(bus, mock_runner, empty_tools):
    return AgentCore(bus=bus, runner=mock_runner, tools=empty_tools)


# ── State machine ─────────────────────────────────────────────────


class TestStateMachine:
    """Verify the state transition table and TurnContext lifecycle."""

    def test_turn_context_defaults(self):
        msg = InboundMessage(content="hi")
        ctx = TurnContext(msg=msg)
        assert ctx.state == TurnState.RESTORE
        assert ctx.histories == []
        assert ctx.session is None
        assert ctx.final_content is None

    def test_all_states_have_transitions(self):
        """Every state (except DONE) must have a defined 'ok' transition."""
        from myAgent.agent.core import _TRANSITIONS

        for state in TurnState:
            if state == TurnState.DONE:
                continue
            assert (state, "ok") in _TRANSITIONS, f"No transition from {state}"

    def test_transitions_chain(self):
        """Walk the full happy-path chain to DONE."""
        from myAgent.agent.core import _TRANSITIONS

        expected = [TurnState.BUILD, TurnState.RUN, TurnState.SAVE,
                     TurnState.RESPOND, TurnState.DONE]
        current = TurnState.RESTORE
        for target in expected:
            current = _TRANSITIONS[(current, "ok")]
            assert current == target


# ── AgentCore message handling ────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_new_session_returns_response(self, core, temp_workspace):
        """A fresh session_key creates a new turn and returns a response."""
        sm = SessionManager(temp_workspace)
        msg = InboundMessage(content="hello")
        result = await core.handle_message(msg, sm, "session-1")
        assert result is not None
        assert result.content == "mock response"

    @pytest.mark.asyncio
    async def test_concurrent_message_stays_pending(self, core, temp_workspace):
        """During a turn, concurrent messages return None (queued)."""
        sm = SessionManager(temp_workspace)
        # Wrap runner to delay so we can send a second message mid-turn
        async def slow_run(spec):
            await asyncio.sleep(0.05)
            return AgentRunResult([], "done", None)
        core.runner.run = AsyncMock(side_effect=slow_run)

        # Start first message
        t1 = asyncio.create_task(
            core.handle_message(InboundMessage("first"), sm, "s1")
        )
        await asyncio.sleep(0.01)  # let first turn begin

        # Concurrent message should return None
        result2 = await core.handle_message(InboundMessage("mid-turn"), sm, "s1")
        assert result2 is None

        out1 = await t1
        assert out1 is not None
        assert out1.content == "done"

    @pytest.mark.asyncio
    async def test_mid_turn_injection_processed(self, core, temp_workspace):
        """Messages injected mid-turn get processed when the turn finishes."""
        sm = SessionManager(temp_workspace)

        call_count = [0]

        async def fake_run(spec):
            call_count[0] += 1
            await asyncio.sleep(0.03)
            return AgentRunResult([], f"response-{call_count[0]}", None)

        core.runner.run = AsyncMock(side_effect=fake_run)

        # Start first message
        t1 = asyncio.create_task(
            core.handle_message(InboundMessage("msg-1"), sm, "s1")
        )
        await asyncio.sleep(0.01)

        # Inject mid-turn message (goes to queue)
        mid = await core.handle_message(InboundMessage("msg-2"), sm, "s1")
        assert mid is None  # queued

        out1 = await t1
        assert out1.content == "response-2"  # both messages processed
        assert call_count[0] == 2


# ── AgentCore _run_turn ──────────────────────────────────────────


class TestRunTurn:
    @pytest.mark.asyncio
    async def test_complete_turn_call_order(self, core, temp_workspace):
        """Verify that a full turn invokes all state handlers in order."""
        sm = SessionManager(temp_workspace)
        msg = InboundMessage("test")

        # Track which states were visited
        visited = []
        orig = {
            s: getattr(core, f"_state_{s.name.lower()}")
            for s in [TurnState.RESTORE, TurnState.BUILD, TurnState.RUN,
                       TurnState.SAVE, TurnState.RESPOND]
        }

        for state, handler in orig.items():
            async def tracker(ctx, s=state, h=handler):
                visited.append(s)
                return await h(ctx)
            setattr(core, f"_state_{state.name.lower()}", tracker)

        result = await core.handle_message(msg, sm, "order-session")
        assert result is not None
        assert visited == [
            TurnState.RESTORE, TurnState.BUILD, TurnState.RUN,
            TurnState.SAVE, TurnState.RESPOND,
        ]

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, core):
        """An unrecognized event should raise RuntimeError."""
        ctx = TurnContext(msg=InboundMessage("x"))
        ctx.state = TurnState.RESTORE

        async def bad_handler(ctx):
            return "bad_event"

        core._state_restore = bad_handler

        with pytest.raises(RuntimeError, match="No transition"):
            await core._run_turn(ctx)


# ── Individual state handlers ─────────────────────────────────────


class TestStateRestore:
    @pytest.mark.asyncio
    async def test_no_session_returns_empty_history(self, core):
        ctx = TurnContext(msg=InboundMessage("x"))
        event = await core._state_restore(ctx)
        assert event == "ok"
        assert ctx.histories == []

    @pytest.mark.asyncio
    async def test_with_session_loads_history(self, core, sample_session):
        ctx = TurnContext(msg=InboundMessage("x"), session=sample_session)
        event = await core._state_restore(ctx)
        assert event == "ok"
        assert len(ctx.histories) > 0


class TestStateBuild:
    @pytest.mark.asyncio
    async def test_builds_messages_with_system_prompt(self, core, temp_workspace):
        sm = SessionManager(temp_workspace)
        session = sm.get_or_create("build-test")
        ctx = TurnContext(msg=InboundMessage("hello"),
                          session=session,
                          session_manager=sm)

        event = await core._state_build(ctx)
        assert event == "ok"
        assert ctx.all_messages[0]["role"] == "system"
        assert ctx.all_messages[-1]["role"] == "user"
        assert ctx.all_messages[-1]["content"] == "hello"


class TestStateRun:
    @pytest.mark.asyncio
    async def test_runs_and_stores_result(self, core, temp_workspace):
        sm = SessionManager(temp_workspace)
        session = sm.get_or_create("run-test")
        ctx = TurnContext(msg=InboundMessage("q"),
                          session=session,
                          session_manager=sm)
        ctx.all_messages = [{"role": "user", "content": "q"}]

        event = await core._state_run(ctx)
        assert event == "ok"
        assert ctx.final_content == "mock response"


class TestStateRespond:
    @pytest.mark.asyncio
    async def test_produces_outbound_message(self, core):
        ctx = TurnContext(msg=InboundMessage("x"))
        ctx.final_content = "The answer"
        event = await core._state_respond(ctx)
        assert event == "ok"
        assert ctx.outbound is not None
        assert ctx.outbound.content == "The answer"

    @pytest.mark.asyncio
    async def test_null_content_defaults_to_empty_string(self, core):
        ctx = TurnContext(msg=InboundMessage("x"))
        ctx.final_content = None
        event = await core._state_respond(ctx)
        assert event == "ok"
        assert ctx.outbound.content == ""


# ── Session locking ───────────────────────────────────────────────


class TestSessionLocking:
    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_processing(self, core, temp_workspace):
        """Two turns on the same key are serialized by the lock."""
        sm = SessionManager(temp_workspace)

        execution_order = []

        async def slow_run(spec):
            execution_order.append("enter")
            await asyncio.sleep(0.05)
            execution_order.append("exit")
            return AgentRunResult([], "ok", None)

        core.runner.run = AsyncMock(side_effect=slow_run)

        async def send(msg_text, key):
            return await core.handle_message(
                InboundMessage(msg_text), sm, key
            )

        r1, r2 = await asyncio.gather(
            send("a", "locked"), send("b", "locked")
        )
        assert r1 is not None
        # r2 was queued during first turn, so it also got processed
        # The lock should serialize them
        assert execution_order == ["enter", "exit", "enter", "exit"]
