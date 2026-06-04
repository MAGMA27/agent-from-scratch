import asyncio
from pathlib import Path

from myAgent.agent.core import AgentCore
from myAgent.agent.memory import Consolidator, MemoryStore
from myAgent.agent.runner import AgentRunner
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import SessionManager


async def main():
    workspace = Path("workspace")
    provider = LLMProvider()
    runner = AgentRunner(provider)
    bus = MessageBus()

    # Memory system
    memory_store = MemoryStore(workspace)
    consolidator = Consolidator(
        store=memory_store,
        provider=provider,
        model="deepseek-v4-flash",
        context_limit=65536,
    )

    core = AgentCore(
        bus, runner,
        consolidator=consolidator,
        memory_store=memory_store,
    )
    session_manager = SessionManager(workspace)
    session_key = '202606041133'

    print("Agent ready. Type your message (/exit to quit)")
    print("Available tools: get_weather, current_time")
    print()

    while True:
        user_input = input(">>> ")
        if user_input.lower() in ("/exit", "/quit", "exit"):
            break
        if not user_input.strip():
            continue

        msg = InboundMessage(content=user_input)
        response = await core.handle_message(msg, session_manager, session_key)

        if response:
            print(response.content)
        print()


if __name__ == "__main__":
    asyncio.run(main())
