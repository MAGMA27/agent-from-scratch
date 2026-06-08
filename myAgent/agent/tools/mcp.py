"""Simplified MCP client: connects to stdio MCP servers and wraps their tools.

Inspired by nanobot's MCP system but stripped down to the essentials:
- stdio transport only
- One AsyncExitStack per server (prevents cancel-scope conflicts)
- Automatic reconnect on transient failures
- Schema normalization for nullable types
"""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from myAgent.agent.tools.base import Tool
from myAgent.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters allowed in tool names by LLM providers (Anthropic, OpenAI, etc.)
_SANITIZE_RE = re.compile(r"_+")

# Exception types that warrant a single retry (transient connection issues)
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset({
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "ConnectionAbortedError",
    "ConnectionError",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_name(name: str) -> str:
    """Sanitize an MCP-derived name for model API compatibility."""
    return _SANITIZE_RE.sub("_", re.sub(r"[^a-zA-Z0-9_-]", "_", name))


def _is_transient(exc: BaseException) -> bool:
    """Check if an exception looks like a transient connection error."""
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """Normalize nullable JSON Schema patterns for OpenAI/Anthropic compatibility.

    MCP servers often return ``oneOf [{"type": "null"}, {...}]`` which the
    provider APIs reject.  This converts them to ``{"type": "...", "nullable": true}``.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)

    # Handle ["string", "null"] → type: "string", nullable: true
    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    # Handle oneOf/anyOf with a null branch
    for key in ("oneOf", "anyOf"):
        branches = normalized.get(key)
        if isinstance(branches, list):
            non_null_branches = [
                b for b in branches if isinstance(b, dict) and b.get("type") != "null"
            ]
            has_null = any(
                isinstance(b, dict) and b.get("type") == "null" for b in branches
            )
            if has_null and len(non_null_branches) == 1:
                merged = {k: v for k, v in normalized.items() if k != key}
                merged.update(non_null_branches[0])
                normalized = merged
                normalized["nullable"] = True
                break

    # Recurse into properties
    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop) if isinstance(prop, dict) else prop
            for name, prop in normalized["properties"].items()
        }

    # Recurse into items
    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """Configuration for one MCP server (stdio only)."""

    command: str          # e.g. "npx" or "python"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    tool_timeout: int = 30  # seconds


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a native myAgent Tool.

    Delegates execution to the MCP session, with automatic reconnect on
    transient failures.
    """

    _plugin_discoverable = False

    def __init__(
        self,
        session: Any,
        server_name: str,
        tool_def: Any,
        tool_timeout: int = 30,
    ) -> None:
        self._session = session
        self._server_name = server_name
        self._original_name: str = tool_def.name
        self._name = _sanitize_name(f"mcp_{server_name}_{tool_def.name}")
        self._description: str = tool_def.description or tool_def.name
        raw_schema = getattr(tool_def, "inputSchema", None) or {
            "type": "object",
            "properties": {},
        }
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout

    # -- Tool interface -------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    # -- Execution with retry logic -------------------------------------------

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        retried_transient = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(self._original_name, arguments=kwargs),
                    timeout=self._tool_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP tool '{}' timed out after {}s",
                    self._name,
                    self._tool_timeout,
                )
                return f"(MCP tool call timed out after {self._tool_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
                return "(MCP tool call was cancelled)"
            except Exception as exc:
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP tool '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP tool '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP tool call failed after retry: {type(exc).__name__})"
                logger.exception("MCP tool '{}' failed: {}", self._name, exc)
                return f"(MCP tool call failed: {type(exc).__name__})"
            else:
                # Success — extract text content
                parts: list[str] = []
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP tool call failed)"


# ---------------------------------------------------------------------------
# MCP Manager
# ---------------------------------------------------------------------------


class MCPManager:
    """Manages connections to multiple MCP servers.

    Lifecycle::

        manager = MCPManager(servers={"filesystem": MCPServerConfig(...)})
        await manager.connect_all(registry)     # connect + register tools
        ...
        await manager.close_all()               # clean shutdown
    """

    def __init__(self, servers: dict[str, MCPServerConfig] | None = None) -> None:
        self._servers = servers or {}
        self._stacks: dict[str, AsyncExitStack] = {}

    @property
    def connected(self) -> bool:
        return bool(self._stacks)

    async def connect_all(self, registry: ToolRegistry) -> dict[str, AsyncExitStack]:
        """Connect all configured servers and register their tools.

        Returns a dict mapping server name → its AsyncExitStack.
        Each server gets its own stack to prevent cancel-scope conflicts
        when multiple servers are configured.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_stacks: dict[str, AsyncExitStack] = {}

        for name, cfg in self._servers.items():
            stack = AsyncExitStack()
            await stack.__aenter__()

            try:
                params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env=cfg.env or None,
                    cwd=cfg.cwd or None,
                )
                read, write = await stack.enter_async_context(
                    stdio_client(params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                # List and register tools
                tools_result = await session.list_tools()
                registered_count = 0
                for tool_def in tools_result.tools:
                    wrapper = MCPToolWrapper(
                        session,
                        name,
                        tool_def,
                        tool_timeout=cfg.tool_timeout,
                    )
                    registry.register(wrapper)
                    registered_count += 1
                    logger.debug(
                        "MCP: registered tool '{}' from server '{}'",
                        wrapper.name,
                        name,
                    )

                logger.info(
                    "MCP server '{}': connected, {} tools registered",
                    name,
                    registered_count,
                )
                server_stacks[name] = stack

            except Exception:
                logger.exception("MCP server '{}': connection failed", name)
                with suppress(Exception):
                    await stack.aclose()
                continue

        self._stacks = server_stacks
        return server_stacks

    async def close_all(self) -> None:
        """Close all MCP server connections."""
        for name, stack in self._stacks.items():
            try:
                await stack.aclose()
                logger.debug("MCP server '{}': closed", name)
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}': cleanup error (can be ignored)", name)
        self._stacks.clear()
