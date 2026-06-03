"""Unit tests for MessageBus and message dataclasses."""

import asyncio

import pytest

from myAgent.bus.bus import InboundMessage, MessageBus, OutboundMessage


class TestInboundMessage:
    def test_default_timestamp(self):
        msg = InboundMessage(content="hey")
        assert msg.content == "hey"
        assert msg.timestamp is not None


class TestOutboundMessage:
    def test_no_reply_to_by_default(self):
        msg = OutboundMessage(content="ok")
        assert msg.content == "ok"
        assert msg.reply_to is None

    def test_explicit_reply_to(self):
        msg = OutboundMessage(content="ok", reply_to="msg-1")
        assert msg.reply_to == "msg-1"


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_inbound_publish_consume(self):
        bus = MessageBus()
        sent = InboundMessage(content="ping")

        async def publisher():
            await bus.publish_inbound(sent)

        async def consumer():
            return await bus.consume_inbound()

        _, received = await asyncio.gather(publisher(), consumer())
        assert received is sent  # same object reference
        assert received.content == "ping"

    @pytest.mark.asyncio
    async def test_outbound_publish_consume(self):
        bus = MessageBus()
        sent = OutboundMessage(content="pong", reply_to="1")

        async def publisher():
            await bus.publish_outbound(sent)

        async def consumer():
            return await bus.consume_outbound()

        _, received = await asyncio.gather(publisher(), consumer())
        assert received is sent
        assert received.content == "pong"
        assert received.reply_to == "1"

    @pytest.mark.asyncio
    async def test_consume_inbound_blocks_until_published(self):
        """consume_inbound should await until a message arrives."""
        bus = MessageBus()

        async def delayed_publish():
            await asyncio.sleep(0.05)
            await bus.publish_inbound(InboundMessage("delayed"))

        t0 = asyncio.get_event_loop().time()
        _, msg = await asyncio.gather(delayed_publish(), bus.consume_inbound())
        elapsed = asyncio.get_event_loop().time() - t0
        assert msg.content == "delayed"
        # Should have waited at least a little
        assert elapsed >= 0.04

    @pytest.mark.asyncio
    async def test_multiple_messages_fifo(self):
        bus = MessageBus()
        for i in range(5):
            await bus.publish_inbound(InboundMessage(content=str(i)))

        for i in range(5):
            msg = await bus.consume_inbound()
            assert msg.content == str(i)
