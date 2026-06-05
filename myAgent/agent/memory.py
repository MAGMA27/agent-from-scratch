"""Memory system: history.jsonl append-only log + lightweight Consolidator."""

import json
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from myAgent.providers.provider import LLMProvider
from myAgent.session.manager import Session
from myAgent.utils.token_counter import count_message_tokens

# ── caps ────────────────────────────────────────────────────────────────────

_HISTORY_ENTRY_MAX_CHARS = 8_000    # per-entry cap in history.jsonl
_RECENT_HISTORY_MAX_ENTRIES = 30    # max entries injected into system prompt
_RECENT_HISTORY_MAX_CHARS = 16_000  # hard cap on "Recent History" section
_HISTORY_ENTRY_PREVIEW_MAX_CHARS = 4_000  # max chars per message in compact prompt


# ── MemoryStore ──────────────────────────────────────────────────────────────

class MemoryStore:
    """Manage history.jsonl: an append-only log of consolidated conversation
    summaries.  Each entry carries an auto-incrementing cursor, a timestamp,
    and a session_key so different sessions' memories stay isolated.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.memory_dir / "history.jsonl"
        self._cursor_file = self.memory_dir / ".cursor"

    # -- public api -----------------------------------------------------------

    def append_history(self, entry: str, *, session_key: str = "") -> int:
        """Append a summary entry and return its cursor."""
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = entry.rstrip()[:_HISTORY_ENTRY_MAX_CHARS]
        record = {
            "cursor": cursor, "timestamp": ts,
            "session_key": session_key, "content": content,
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def get_recent_history(self, *, session_key: str = "") -> str:
        """Return recent entries for *session_key* formatted for the system prompt.

        When session_key is empty (default), returns all entries (legacy / global).
        """
        entries = [
            e for e in self._read_entries()
            if not session_key or e.get("session_key") == session_key
        ]
        if not entries:
            return ""

        recent = entries[-_RECENT_HISTORY_MAX_ENTRIES:]
        lines: list[str] = []
        total = 0
        for e in reversed(recent):
            line = f"- [{e['timestamp']}] {e['content']}"
            if total + len(line) > _RECENT_HISTORY_MAX_CHARS:
                break
            lines.append(line)
            total += len(line)
        lines.reverse()
        return "# Recent History\n\n" + "\n".join(lines) if lines else ""

    # -- internal helpers -----------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            with open(self.history_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        with suppress(json.JSONDecodeError):
                            entries.append(json.loads(line))
        return entries

    def _next_cursor(self) -> int:
        if self._cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
        entries = self._read_entries()
        return max((e.get("cursor", 0) for e in entries), default=0) + 1


# ── Consolidator ─────────────────────────────────────────────────────────────

COMPACT_SYSTEM = """Summarize this conversation chunk. Keep:
- Key decisions and their rationale
- Facts the user shared about themselves
- Ongoing tasks or open questions
Output plain text, no preamble."""


class Consolidator:
    """Lightweight consolidation: when unconsolidated session tokens exceed
    budget, compress the oldest unconsolidated messages into a summary,
    append it to history.jsonl, and advance last_consolidated.

    Already-consolidated messages stay in session.messages (skipped by
    get_history) — they are never re-compressed.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        context_limit: int,
        completion_reserve: int = 4096,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.context_limit = context_limit
        self.budget = context_limit - completion_reserve - 1024

    async def compact(self, session: Session) -> str | None:
        """Check token usage; compress if over budget.  Returns the summary or None."""
        start = session.last_consolidated
        unconsolidated = session.messages[start:]
        total = sum(count_message_tokens(m) for m in unconsolidated)
        if total <= self.budget:
            return None

        removed = 0
        boundary_rel = 0
        for i, msg in enumerate(unconsolidated):
            if i > 0 and msg["role"] == "user":
                boundary_rel = i
                if removed >= total - int(self.budget * 0.5):
                    break
            removed += count_message_tokens(msg)

        if boundary_rel == 0:
            return None

        boundary = start + boundary_rel
        chunk = session.messages[start:boundary]
        formatted = "\n".join(
            f"[{m['role']}] {str(m.get('content', ''))[:_HISTORY_ENTRY_PREVIEW_MAX_CHARS]}"
            for m in chunk
        )

        summary = await self._summarize(formatted)
        if not summary:
            return None

        self.store.append_history(summary, session_key=session.key)
        session.last_consolidated = boundary
        return summary

    async def _summarize(self, text: str) -> str | None:
        try:
            response = await self.provider.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": COMPACT_SYSTEM},
                    {"role": "user", "content": text},
                ],
            )
            return response.content or "(empty)"
        except Exception:
            return None


def get_memory_context(store: MemoryStore, *, session_key: str = "") -> str:
    """Return recent history.jsonl entries for *session_key* formatted for
    the system prompt.
    """
    return store.get_recent_history(session_key=session_key)
