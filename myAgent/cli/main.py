"""CLI commands for myAgent — interactive chat and status."""

import asyncio
import signal
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
from myAgent.bus.bus import InboundMessage, MessageBus
from myAgent.cli.stream import StreamRenderer
from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import SessionManager

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
    runner = AgentRunner(provider)
    bus = MessageBus()

    memory_store = MemoryStore(workspace)
    consolidator = Consolidator(
        store=memory_store,
        provider=provider,
        model="deepseek-v4-flash",
        context_limit=65536,
    )
    skill_sys = SkillLoader(workspace)

    core = AgentCore(
        bus, runner,
        consolidator=consolidator,
        memory_store=memory_store,
        skill_sys=skill_sys,
    )
    # --- Register hooks --------------------------------------------------
    core.hooks.append(LoggingHook())
    session_manager = SessionManager(workspace)
    return core, session_manager, provider


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

        try:
            response = await core.handle_message(
                InboundMessage(content=message),
                session_mgr,
                session_key,
            )
        finally:
            await core.runner.tools.close_all() if hasattr(core.runner.tools, "close_all") else None

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

        def _handle_signal(signum, _frame):
            sig_name = signal.Signals(signum).name
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while True:
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

                # Create a fresh renderer for this turn
                renderer = StreamRenderer(
                    render_markdown=render_markdown,
                    show_spinner=True,
                    bot_name="myAgent",
                )

                try:
                    response = await core.handle_message(
                        InboundMessage(content=command),
                        session_mgr,
                        session_key,
                        on_delta=renderer.on_delta,
                    )
                finally:
                    await renderer.on_end()
                    await renderer.close()

                # If streaming already rendered everything, skip
                if renderer.streamed:
                    pass
                elif response and response.content:
                    renderer.render_complete(response.content)
                else:
                    with renderer.pause_spinner():
                        pass

        finally:
            await core.runner.tools.close_all() if hasattr(core.runner.tools, "close_all") else None

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
