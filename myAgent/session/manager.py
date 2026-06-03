import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from myAgent.utils.token_counter import count_message_tokens


def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    """Find the first index whose tool results have matching assistant calls."""
    declared: set[str] = set()
    start = 0
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    declared.add(str(tc["id"]))
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid and str(tid) not in declared:
                start = i + 1
                declared.clear()
    return start


@dataclass
class Session():
    ''''''
    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_window(self, messages: list[dict[str, Any]], context_budget: int) -> list[dict[str, Any]]:
        """from end to head, comsuming cotext budget"""
        if context_budget <= 0:
            return []

        kept = []
        used = 0
        for msg in reversed(messages):
            tokens = count_message_tokens(msg)
            if kept and used + tokens > context_budget:
                break
            kept.append(msg)
            used += tokens
        kept.reverse()
        return kept

    def get_history(
        self,
        max_messages: int = 120,
        *,
        max_tokens: int = 0,
        include_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        '''Return unconsolidated messages for LLM input'''
        # get unconsolidated messages
        unconsolidated = self.messages[self.last_consolidated:]
        max_messages = max_messages if max_messages > 0 else 120
        sliced = unconsolidated[-max_messages:]

        # sliced = self.get_window(sliced, max_tokens)

        # begaining position alignment
        # if the message is deliveried by bot, and user may be replying to it
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        # Drop orphan tool results at the front.
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        # sanitization
        pass

        return sliced


class SessionManager:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(exist_ok=True)

        self.sessions_dir = self.workspace / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self._cache: dict[str, Session] = {}

    def get_or_create(self, key: str) -> Session:
        ''''''
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        '''load a session from disk'''
        path = self._get_session_path(key)

        if not path.exists():
            return None

        messages = []
        metadata = {}
        created_at = None
        updated_at = None
        last_consolidated = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                if data.get("_type") == "metadata":
                    metadata = data.get("metadata", {})
                    created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                    updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
                    last_consolidated = data.get("last_consolidated", 0)
                else:
                    messages.append(data)

        return Session(
            key=key,
            messages=messages,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=metadata,
            last_consolidated=last_consolidated
        )

    async def save(self, session: Session) -> None:
        ''''''
        path = self._get_session_path(session.key)
        tmp_path = path.with_suffix(".jsonl.tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                metadata_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            os.replace(tmp_path, path)

        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        self._cache[session.key] = session

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{key}.jsonl"

