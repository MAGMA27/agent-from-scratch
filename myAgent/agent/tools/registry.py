from typing import Any
import json
from myAgent.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """
    def __init__(self):
        self._tools: dict[str: Tool] = {}

    def register(self, tool: Tool) -> None:
        '''regist tool'''
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        '''get a tool'''
        return self._tools.get(name, None)
    
    @property
    def tool_spec(self) -> list[dict[str, Any]]:
        t_spec = []
        for _, tool in self._tools.items():
            t_spec.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return t_spec

    async def execute(self, name: str, params: dict) -> Any:
        tool = self._tools.get(name, None)
        if tool:
            return await tool.execute(**params)
        else:
            return None