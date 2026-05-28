import asyncio
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    content: str  # Message text
    session: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    content: str
    session: list[dict] = field(default_factory=list)
    reply_to: str | None = None


class MessageBus:
    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()
