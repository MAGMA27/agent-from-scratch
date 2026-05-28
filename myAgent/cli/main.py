import asyncio
from datetime import datetime

from myAgent.agent.core import AgentCore
from myAgent.agent.runner import AgentRunner
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.providers.provider import LLMProvider


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
    session = []

    print("Agent ready. Type your message (/exit to quit)")
    print("Available tools: get_weather, current_time")
    print()

    while True:
        user_input = input(">>> ")
        if user_input.lower() in ("/exit", "/quit", "exit"):
            break
        if not user_input.strip():
            continue

        msg = InboundMessage(content=user_input, session=session)
        response = await core.process_message(msg)

        if response:
            session = response.session
            print(response.content)
        print()


if __name__ == "__main__":
    asyncio.run(main())
