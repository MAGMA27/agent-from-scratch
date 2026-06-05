from typing import Any

from myAgent.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        '''regist tool'''
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        '''get a tool'''
        return self._tools.get(name, None)

    def has(self, name: str) -> bool:
        '''check if a tool is registered'''
        return name in self._tools

    @property
    def tool_spec(self) -> list[dict[str, Any]]:
        t_spec = []
        for _, tool in self._tools.items():
            t_spec.append(tool.to_schema())
        return t_spec

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        tool = self._tools.get(name, None)
        if tool:
            params = tool.cast_params(params)
            validation = tool.validate_params(params)
            if validation:
                return f"json schema value validation failed: {validation}"
            return await tool.execute(**params)
        else:
            return None
