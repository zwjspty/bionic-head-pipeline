from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID


HistoryRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ConversationTurn:
    role: HistoryRole
    content: str

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ConversationHistoryMetrics:
    enabled: bool
    turn_count: int
    message_count: int
    char_count: int


class ConversationHistoryStore:
    def __init__(
        self,
        *,
        enabled: bool = True,
        max_turn_pairs: int = 6,
        max_chars: int = 3000,
    ) -> None:
        if max_turn_pairs < 1:
            raise ValueError("max_turn_pairs must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.enabled = enabled
        self.max_turn_pairs = max_turn_pairs
        self.max_chars = max_chars
        self._messages_by_session: dict[UUID, list[ConversationTurn]] = {}

    def get(self, session_id: UUID) -> list[dict[str, str]]:
        if not self.enabled:
            return []
        return [turn.as_message() for turn in self._messages_by_session.get(session_id, [])]

    def append_pair(self, session_id: UUID, *, user: str, assistant: str) -> None:
        if not self.enabled:
            return
        messages = self._messages_by_session.setdefault(session_id, [])
        messages.extend(
            [
                ConversationTurn(role="user", content=user),
                ConversationTurn(role="assistant", content=assistant),
            ]
        )
        self._trim(session_id)

    def metrics(self, session_id: UUID) -> ConversationHistoryMetrics:
        if not self.enabled:
            return ConversationHistoryMetrics(
                enabled=False,
                turn_count=0,
                message_count=0,
                char_count=0,
            )
        messages = self._messages_by_session.get(session_id, [])
        return ConversationHistoryMetrics(
            enabled=True,
            turn_count=len(messages) // 2,
            message_count=len(messages),
            char_count=sum(len(turn.content) for turn in messages),
        )

    def _trim(self, session_id: UUID) -> None:
        messages = self._messages_by_session.get(session_id)
        if not messages:
            return

        max_messages = self.max_turn_pairs * 2
        while len(messages) > max_messages:
            del messages[:2]

        while len(messages) > 2 and sum(len(turn.content) for turn in messages) > self.max_chars:
            del messages[:2]
