from pathlib import Path

from myAgent.agent.tools.base import Tool, tool_parameters


@tool_parameters({
    "type": "object",
    "properties": {"path": {"type": "string", "description": "File path to read"}},
    "required": ["path"],
})
class ReadFile(Tool):
    name = "read_file"
    description = "Read contents of a file"

    async def execute(self, path: str, **kwargs) -> str:
        try:
            return Path(path).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return Path(path).read_text(encoding='gbk')
