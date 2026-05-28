import asyncio

from myAgent.agent.core import AgentCore
from myAgent.agent.runner import AgentRunner
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.providers.provider import LLMProvider

if __name__ == "__main__":
    provider = LLMProvider()
    runner = AgentRunner(provider)
    bus = MessageBus()
    tools = [{
        "type": "function",
        "function": {
            "name": "test",
            "description": "useless",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }]

    core = AgentCore(bus, runner, tools)
    session = []

    while True:
        user_input = input(">>> ")
        if user_input.lower() in ("/exit", "/quit", "exit"):
            break
        if not user_input.strip():
            continue
        msg = InboundMessage(content=user_input, session=session)
        asyncio.run(bus.publish_inbound(msg))
        a_message = asyncio.run(bus.consume_inbound())
        a_response = asyncio.run(core.process_message(a_message))
        asyncio.run(bus.publish_outbound(a_response))

        result = asyncio.run(bus.consume_outbound())
        session = result.session
        print(result.content)
