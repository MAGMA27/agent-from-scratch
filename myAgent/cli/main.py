import asyncio
from datetime import datetime
from pathlib import Path

from myAgent.agent.core import AgentCore
from myAgent.agent.runner import AgentRunner
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import SessionManager


async def handle_get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 22°C"


async def handle_current_time() -> str:
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


async def main():
    provider = LLMProvider()
    runner = AgentRunner(provider)
    bus = MessageBus()

    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "current_time",
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]

    runner.register_tool("get_weather", handle_get_weather)
    runner.register_tool("current_time", handle_current_time)

    core = AgentCore(bus, runner, tool_definitions)
    session_manager = SessionManager(Path('workspace'))
    session_key = '202606012020'


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
