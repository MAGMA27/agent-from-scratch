"""Memory system: history.jsonl append-only log + lightweight Consolidator."""

import json
import os
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
    summaries.  Each entry carries an auto-incrementing cursor and a timestamp.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.memory_dir / "history.jsonl"
        self._cursor_file = self.memory_dir / ".cursor"

    # -- public api -----------------------------------------------------------

    def append_history(self, entry: str) -> int:
        """Append a summary entry and return its cursor."""
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = entry.rstrip()[:_HISTORY_ENTRY_MAX_CHARS]
        record = {"cursor": cursor, "timestamp": ts, "content": content}
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def get_recent_history(self) -> str:
        """Return recent entries as a formatted string for the system prompt."""
        entries = self._read_entries()
        if not entries:
            return ""

        # Take the most recent N entries, then truncate by char budget.
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
        # Fallback: scan the file and take max(cursor) + 1.
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
        # Only consider unconsolidated messages — already-compressed ones are
        # skipped by get_history() and should never drive further compaction.
        start = session.last_consolidated
        unconsolidated = session.messages[start:]
        total = sum(count_message_tokens(m) for m in unconsolidated)
        if total <= self.budget:
            return None

        # Locate a safe boundary within the *unconsolidated* region.
        # Stop at a user turn once enough tokens have been accounted for.
        removed = 0
        boundary_rel = 0  # offset from `start`
        for i, msg in enumerate(unconsolidated):
            if i > 0 and msg["role"] == "user":
                boundary_rel = i
                if removed >= total - int(self.budget * 0.5):
                    break
            removed += count_message_tokens(msg)

        if boundary_rel == 0:
            return None

        boundary = start + boundary_rel  # absolute index in session.messages
        chunk = session.messages[start:boundary]
        formatted = "\n".join(
            f"[{m['role']}] {str(m.get('content', ''))[:_HISTORY_ENTRY_PREVIEW_MAX_CHARS]}"
            for m in chunk
        )

        # Ask LLM to summarize.
        summary = await self._summarize(formatted)
        if not summary:
            return None

        # Persist to history.jsonl.
        self.store.append_history(summary)

        # Advance consolidation cursor so these messages are never touched again.
        session.last_consolidated = boundary

        # Store the summary in session metadata so get_memory_context() can
        # inject it as [Archived Context Summary] on the next turn.
        session.metadata["_last_summary"] = {
            "text": summary,
            "last_active": datetime.now().isoformat(),
        }

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


def get_memory_context(store: MemoryStore, session: Session) -> str:
    """Build the memory section for the system prompt: recent history.jsonl
    entries (from the Consolidator) plus any pending session summary.
    """
    parts: list[str] = []

    recent = store.get_recent_history()
    if recent:
        parts.append(recent)

    meta = session.metadata.get("_last_summary")
    if isinstance(meta, dict):
        text = meta.get("text")
        if text:
            parts.append(f"[Archived Context Summary]\n\n{text}")

    return "\n\n".join(parts)
