"""Streaming renderer for CLI output.

Uses Rich Live with ``transient=True`` for in-place markdown updates during
streaming. After the live display stops, a final clean render is printed so
the content persists on screen.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text


def _clear_current_line(console: Console) -> None:
    """Erase a transient status line before printing persistent output."""
    file = console.file
    isatty = getattr(file, "isatty", lambda: False)
    if not isatty():
        return
    file.write("\r\x1b[2K")
    file.flush()


def _make_console() -> Console:
    """Create a Console that emits plain text when stdout is not a TTY."""
    return Console(file=sys.stdout, force_terminal=sys.stdout.isatty())


class ThinkingSpinner:
    """Spinner that shows '<bot_name> is thinking...' with pause support."""

    def __init__(self, console: Console | None = None, bot_name: str = "myAgent"):
        c = console or _make_console()
        self._console = c
        self._spinner = c.status(f"[dim]{bot_name} is thinking...[/dim]", spinner="dots")
        self._active = False

    def __enter__(self):
        self._spinner.start()
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        self._spinner.stop()
        _clear_current_line(self._console)
        return False

    @contextmanager
    def pause(self):
        """Context manager: temporarily stop spinner for clean output."""
        if self._spinner and self._active:
            self._spinner.stop()
            _clear_current_line(self._console)
        try:
            yield
        finally:
            if self._spinner and self._active:
                self._spinner.start()


class StreamRenderer:
    """Streaming renderer with Rich Live for in-place updates.

    During streaming: updates content in-place via Rich Live.
    On end: stops Live (transient=True erases it), then prints final render.

    Also supports non-streaming mode via ``render_complete()`` which prints
    the full response directly with a header.
    """

    def __init__(
        self,
        render_markdown: bool = True,
        show_spinner: bool = True,
        bot_name: str = "myAgent",
    ):
        self._md = render_markdown
        self._show_spinner = show_spinner
        self._bot_name = bot_name
        self._buf = ""
        self.streamed = False
        self._console = _make_console()
        self._live: Live | None = None
        self._spinner: ThinkingSpinner | None = None
        self._header_printed = False
        self._start_spinner()

    def _renderable(self):
        """Create a renderable from the current buffer."""
        if self._md and self._buf:
            return Markdown(self._buf)
        return Text(self._buf or "")

    def _render_str(self) -> str:
        """Render current buffer to a plain string via Rich."""
        with self._console.capture() as cap:
            self._console.print(self._renderable())
        return cap.get()

    def _start_spinner(self) -> None:
        if self._show_spinner:
            self._spinner = ThinkingSpinner(bot_name=self._bot_name)
            self._spinner.__enter__()

    def _stop_spinner(self) -> None:
        if self._spinner:
            self._spinner.__exit__(None, None, None)
            self._spinner = None

    @property
    def console(self) -> Console:
        """Expose the console so external print functions can use it."""
        return self._console

    @property
    def header_printed(self) -> bool:
        """Whether this turn has already opened the assistant output block."""
        return self._header_printed

    def ensure_header(self) -> None:
        """Stop spinner and print the assistant header once per turn."""
        self._stop_spinner()
        if self._header_printed:
            return
        self._console.print()
        header = f"🤖 {self._bot_name}"
        self._console.print(f"[cyan]{header}[/cyan]")
        self._header_printed = True

    def pause_spinner(self):
        """Context manager: temporarily pause spinner for inline trace lines."""
        @contextmanager
        def _pause():
            live_was_active = self._live is not None
            if self._live:
                self._live.stop()
                self._live = None
            with self._spinner.pause() if self._spinner else nullcontext():
                yield
            if live_was_active:
                return
        return _pause()

    # -- streaming interface (for future provider streaming support) ----------

    async def on_delta(self, delta: str) -> None:
        """Feed an incremental text delta during streaming."""
        self.streamed = True
        self._buf += delta
        if self._live is None:
            if not self._buf.strip():
                return
            self.ensure_header()
            self._live = Live(
                self._renderable(),
                console=self._console,
                auto_refresh=False,
                transient=True,
            )
            self._live.start()
        else:
            self._live.update(self._renderable())
        self._live.refresh()

    async def on_end(self) -> None:
        """Stop the live display and print the final static render."""
        if self._live:
            self._live.refresh()
            self._live.update(self._renderable())
            self._live.refresh()
            self._live.stop()
            self._live = None
        self._stop_spinner()
        if self._buf.strip():
            out = sys.stdout
            out.write(self._render_str())
            out.flush()

    # -- non-streaming interface (current provider) ---------------------------

    def render_complete(self, content: str) -> None:
        """Print a full (non-streamed) response with header and markdown."""
        self.ensure_header()
        self._stop_spinner()
        if self._live:
            self._live.stop()
            self._live = None
        if not content.strip():
            return
        body = Markdown(content) if self._md else Text(content)
        self._console.print(body)
        self._console.print()

    def stop_for_input(self) -> None:
        """Stop spinner before user input to avoid prompt_toolkit conflicts."""
        self._stop_spinner()

    def pause(self):
        """Context manager: pause spinner for external output."""
        if self._spinner:
            return self._spinner.pause()
        return nullcontext()

    async def close(self) -> None:
        """Stop spinner/live without rendering."""
        if self._live:
            self._live.stop()
            self._live = None
        self._stop_spinner()
