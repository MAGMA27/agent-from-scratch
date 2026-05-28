from myAgent.agent.runner import AgentRunSpec, AgentRunResult, AgentRunner
from myAgent.providers.provider import LLMProvider
import asyncio


if __name__ == "__main__":
    provider = LLMProvider()
    runner = AgentRunner(provider)

    messages = [{
        "role": "user",
        "content": "你好"
    }]

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

    result = asyncio.run(runner.run(messages, tools)) 
    if result.error:
        print(result.error)
    else:
        print(result.final_content)