"""CLI commands for myAgent — interactive chat and status."""

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from myAgent import __logo__, __version__
from myAgent.agent.core import AgentCore
from myAgent.agent.hook import AgentHook, AgentHookContext
from myAgent.agent.memory import Consolidator, MemoryStore
from myAgent.agent.runner import AgentRunner
from myAgent.agent.skills import SkillLoader
from myAgent.agent.subagent import SubagentManager
from myAgent.agent.tools.mcp import MCPServerConfig
from myAgent.agent.tools.spawn import SpawnTool
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.cli.stream import StreamRenderer
from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import SessionManager

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        import os as _os
        _os.environ["PYTHONIOENCODING"] = "utf-8"
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

logger.remove()
_LOG_HANDLER_ID = logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<level>{message}</level>"
    ),
    level="WARNING",
    colorize=None,
)

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="myagent",
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION

    history_dir = Path.home() / ".myagent"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "history.txt"

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,
    )


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=sys.stdout.isatty(),
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


async def _print_interactive_line(text: str) -> None:
    """Print a dimmed progress line via prompt_toolkit-safe Rich."""

    def _write() -> None:
        ansi = _render_interactive_ansi(lambda c: c.print(f"  [dim]…{text}[/dim]"))
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display)."""
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def _is_exit_command(command: str) -> bool:
    return command.strip().lower() in EXIT_COMMANDS


# ---------------------------------------------------------------------------
# Example hook
# ---------------------------------------------------------------------------


class LoggingHook(AgentHook):
    """Log each iteration and tool execution to the console.

    Tweak or replace this to implement custom lifecycle callbacks --
    e.g. send progress to a channel plugin, emit structured events, etc.
    """

    async def before_run(self, messages: list[dict[str, Any]]) -> None:
        logger.info("Hook: before_run -- {} messages", len(messages))

    async def after_run(
        self,
        messages: list[dict[str, Any]],
        final_content: str | None,
        error: str | None,
    ) -> None:
        logger.info(
            "Hook: after_run -- {} messages, final={}chars, error={}",
            len(messages),
            len(final_content) if final_content else 0,
            error,
        )

    async def on_error(self, error: str) -> None:
        logger.info("Hook: on_error -- {}", error)

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        logger.info(
            "Hook: before_iteration #{} -- {} messages",
            ctx.iteration + 1,
            len(ctx.messages),
        )

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        names = [tc.name for tc in ctx.tool_calls]
        logger.info("Hook: before_execute_tools -- {}", names)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        logger.info(
            "Hook: after_iteration #{} -- stop={}",
            ctx.iteration + 1,
            ctx.stop_reason or "continue",
        )


# ---------------------------------------------------------------------------
# Agent bootstrap
# ---------------------------------------------------------------------------


def _build_agent(workspace: Path) -> tuple[AgentCore, SessionManager, LLMProvider]:
    """Wire up the agent subsystems and return (core, session_manager, provider)."""
    provider = LLMProvider()
    mcp_servers = _build_mcp_servers()
    runner = AgentRunner(provider, mcp_servers=mcp_servers)
    bus = MessageBus()

    memory_store = MemoryStore(workspace)
    consolidator = Consolidator(
        store=memory_store,
        provider=provider,
        model="deepseek-v4-flash",
        context_limit=65536,
    )
    skill_sys = SkillLoader(workspace)

    session_manager = SessionManager(workspace)

    # --- Subagent infra -----------------------------------------------
    subagent_mgr = SubagentManager(
        runner=runner,
        bus=bus,
        workspace=workspace,
        session_manager=session_manager,
    )
    spawn_tool = SpawnTool(manager=subagent_mgr)
    runner.tools.register(spawn_tool)
    runner.tool_spec = runner.tools.tool_spec
    logger.info("Registered spawn tool for subagent support")

    core = AgentCore(
        bus, runner,
        consolidator=consolidator,
        memory_store=memory_store,
        skill_sys=skill_sys,
        subagent_manager=subagent_mgr,
    )
    # --- Register hooks --------------------------------------------------
    core.hooks.append(LoggingHook())
    return core, session_manager, provider


def _build_mcp_servers():
    """Build MCP server configs from environment variables.

    Set env vars like:
      MCP_SERVERS=server1,server2
      MCP_SERVER1_COMMAND=npx
      MCP_SERVER1_ARGS=-y,@modelcontextprotocol/server-filesystem,/path
      MCP_SERVER1_TIMEOUT=30
    """
    import os

    server_names = os.environ.get("MCP_SERVERS", "")
    if not server_names:
        return {}

    servers = {}
    for name in server_names.split(","):
        name = name.strip()
        if not name:
            continue
        prefix = name.upper()
        command = os.environ.get(f"MCP_{prefix}_COMMAND", "")
        if not command:
            logger.warning(
                "MCP server '{}': no MCP_{}_COMMAND set, skipping",
                name, prefix,
            )
            continue
        args_raw = os.environ.get(f"MCP_{prefix}_ARGS", "")
        args = args_raw.split(",") if args_raw else []
        timeout_raw = os.environ.get(f"MCP_{prefix}_TIMEOUT", "30")
        servers[name] = MCPServerConfig(
            command=command,
            args=args,
            tool_timeout=int(timeout_raw),
        )
    return servers


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _version_callback(value: bool):
    if value:
        console.print(f"{__logo__} myAgent v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True,
    ),
    message: str | None = typer.Option(
        None, "--message", "-m", help="One-shot message (non-interactive mode)",
    ),
    session: str = typer.Option(
        "default", "--session", "-s", help="Session key for conversation persistence",
    ),
    workspace: str = typer.Option(
        "workspace", "--workspace", "-w", help="Workspace directory",
    ),
    no_md: bool = typer.Option(
        False, "--no-md", help="Disable markdown rendering in output",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-V", count=True, help="Increase log verbosity (-V for INFO, -VV for DEBUG)",
    ),
):
    """myAgent — a from-scratch AI agent framework.

    Run without arguments to start interactive chat.
    """
    # If a subcommand was given (e.g. 'myagent status'), let it handle itself.
    if ctx.invoked_subcommand is not None:
        return

    ws_path = Path(workspace).resolve()
    render_md = not no_md
    if message:
        _run_one_shot(message, session, ws_path, render_md, verbose=verbose)
    else:
        _run_interactive(session, ws_path, render_md, verbose=verbose)



def _run_one_shot(
    message: str, session_key: str, workspace: Path, render_markdown: bool,
    verbose: int = 0,
) -> None:
    """Send a single message and print the response, then exit."""

    _set_verbose(verbose)

    async def _run() -> None:
        core, session_mgr, _provider = _build_agent(workspace)

        await core.runner.connect_mcp()
        try:
            response = await core.handle_message(
                InboundMessage(content=message),
                session_mgr,
                session_key,
            )
        finally:
            await core.runner.close_mcp()

        if response and response.content:
            body = response.content
        else:
            body = "(no response)"

        from rich.markdown import Markdown
        from rich.text import Text

        console.print()
        console.print(f"[cyan]{__logo__} myAgent[/cyan]")
        console.print(Markdown(body) if render_markdown else Text(body))
        console.print()

    asyncio.run(_run())



def _set_verbose(verbose: int) -> None:
    """Adjust log level: 0 = WARNING, 1 = INFO, 2+ = DEBUG."""
    if verbose >= 2:
        level = "DEBUG"
    elif verbose == 1:
        level = "INFO"
    else:
        return  # keep WARNING
    logger.remove(_LOG_HANDLER_ID)
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=None,
    )

def _run_interactive(
    session_key: str, workspace: Path, render_markdown: bool,
    verbose: int = 0,
) -> None:
    """Run the interactive chat loop with prompt_toolkit and Rich output."""

    _set_verbose(verbose)
    _init_prompt_session()

    async def _loop() -> None:
        core, session_mgr, _provider = _build_agent(workspace)
        console.print(
            f"{__logo__} Interactive mode [bold]exit[/bold] "
            f"or [bold]Ctrl+C[/bold] to quit\n"
        )

        await core.runner.connect_mcp()
        try:
            # -- turn sync: block main loop until consumer finishes --
            turn_done = asyncio.Event()
            turn_done.set()  # first read unblocked

            # -- bus consumer: handles user messages AND subagent results --
            async def _print_oob_markdown(text: str) -> None:
                """Render out-of-band markdown safely via prompt_toolkit."""
                from rich.markdown import Markdown

                def _write():
                    c = Console(force_terminal=sys.stdout.isatty())
                    with c.capture() as capture:
                        c.print()
                        c.print(Markdown(text))
                        c.print()
                    print_formatted_text(ANSI(capture.get()), end="")

                await run_in_terminal(_write)

            async def _bus_consumer():
                """Consume messages from the bus and dispatch to AgentCore."""
                nonlocal turn_done
                while True:
                    try:
                        msg = await core.bus.consume_inbound()
                    except asyncio.CancelledError:
                        return

                    # Out-of-band? turn_done already set means main loop is
                    # not waiting for us -- e.g. subagent result arriving
                    # after the user's turn completed.  Process silently
                    # and render safely via run_in_terminal.
                    is_oob = turn_done.is_set()

                    if is_oob:
                        response = await core.handle_message(
                            msg, session_mgr, msg.session_key, on_delta=None,
                        )
                        if response and response.content:
                            await _print_oob_markdown(response.content)
                        continue

                    # User-initiated turn: normal streaming flow
                    renderer = StreamRenderer(
                        render_markdown=render_markdown,
                        show_spinner=True,
                        bot_name="myAgent",
                    )
                    try:
                        response = await core.handle_message(
                            msg,
                            session_mgr,
                            msg.session_key,
                            on_delta=renderer.on_delta,
                        )
                    finally:
                        await renderer.on_end()
                        await renderer.close()

                    if response is None:
                        continue  # mid-turn injection, nothing to render

                    if response.content:
                        renderer.render_complete(response.content)

                    turn_done.set()  # unblock main loop

            consumer = asyncio.create_task(_bus_consumer())

            # -- main loop: read user input, publish to bus --
            try:
                while True:
                    await turn_done.wait()  # wait for prev turn
                    try:
                        user_input = await _read_interactive_input_async()
                    except KeyboardInterrupt:
                        console.print("\nGoodbye!")
                        break

                    command = user_input.strip()
                    if not command:
                        continue
                    if _is_exit_command(command):
                        console.print("\nGoodbye!")
                        break

                    turn_done.clear()
                    await core.bus.publish_inbound(
                        InboundMessage(content=command, session_key=session_key)
                    )
            finally:
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer

        finally:
            await core.runner.close_mcp()


    asyncio.run(_loop())


@app.command()
def status(
    workspace: str = typer.Option(
        "workspace", "--workspace", "-w", help="Workspace directory",
    ),
):
    """Show myAgent status and configuration."""
    from myAgent.agent.tools.loader import ToolLoader
    from myAgent.agent.tools.registry import ToolRegistry

    ws = Path(workspace).resolve()

    console.print(f"\n{__logo__} myAgent v{__version__}\n")
    console.print(f"Workspace: {ws} {'[green]ok[/green]' if ws.exists() else '[red]missing[/red]'}")

    # Tools
    loader = ToolLoader()
    registry = ToolRegistry()
    loader.load(ctx=None, registry=registry, scope="core")
    console.print(f"Tools loaded: {len(registry.tool_spec)}")

    # Provider
    try:
        provider = LLMProvider()
        console.print(
            f"Provider: DeepSeek [dim](base: {provider.client.base_url})[/dim]"
        )
    except Exception as e:
        console.print(f"Provider: [red]error ({e})[/red]")

    # Sessions
    sessions_dir = ws / "sessions"
    if sessions_dir.exists():
        session_files = list(sessions_dir.glob("*.jsonl"))
        console.print(f"Sessions: {len(session_files)} file(s)")

    # Memory
    memory_dir = ws / "memory"
    if memory_dir.exists():
        history_file = memory_dir / "history.jsonl"
        if history_file.exists():
            entries = sum(1 for _ in open(history_file, encoding="utf-8"))
            console.print(f"Memory entries: {entries}")

    # Skills
    skill_sys = SkillLoader(ws)
    skills = skill_sys.list_skills(filter_unavailable=False)
    console.print(f"Skills available: {len(skills)}")

    console.print()


if __name__ == "__main__":
    app()
