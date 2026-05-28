from myAgent.agent.runner import AgentRunner
from myAgent.providers.provider import LLMProvider
from myAgent.agent.core import AgentCore
from myAgent.bus.bus import InboundMessage, OutboundMessage, MessageBus
import asyncio


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
    asyncio.run(bus.publish_inbound(InboundMessage(content="你好")))
    a_message = asyncio.run(bus.consume_inbound())
    a_response = asyncio.run(core.process_message(a_message))
    asyncio.run(bus.publish_outbound(a_response))

    result = asyncio.run(bus.consume_outbound())
    print(result.content)