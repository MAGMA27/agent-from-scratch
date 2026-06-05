import asyncio
import sys
from pathlib import Path

from loguru import logger

from myAgent.agent.core import AgentCore
from myAgent.agent.memory import Consolidator, MemoryStore
from myAgent.agent.runner import AgentRunner
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import SessionManager
from myAgent.agent.skills import SkillLoader

# Remove default handler and configure unified format
logger.remove()
_log_handler_id = logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=None,
)

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
    # Skill system
    skill_sys = SkillLoader(workspace)

    core = AgentCore(
        bus, runner,
        consolidator=consolidator,
        memory_store=memory_store,
        skill_sys=skill_sys,
    )
    session_manager = SessionManager(workspace)
    session_key = '202606041133'

    logger.info("Agent ready. Type your message (/exit to quit)")

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("/exit", "/quit", "exit"):
            break
        if not user_input.strip():
            continue

        msg = InboundMessage(content=user_input)
        response = await core.handle_message(msg, session_manager, session_key)

        if response:
            logger.info("[response] {}", response.content[:200])
            print(response.content)
        print()


if __name__ == "__main__":
    asyncio.run(main())
