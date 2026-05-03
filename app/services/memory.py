from __future__ import annotations

from collections import deque

from app.models.domain import ConversationTurn
from app.utils.tokens import TokenCounter


class ConversationMemory:
    """Short-term memory that retains the most recent grounded interactions."""

    def __init__(self, max_turns: int) -> None:
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add_turn(self, user_query: str, assistant_response: str) -> None:
        self._turns.append(
            ConversationTurn(
                user_query=user_query.strip(),
                assistant_response=assistant_response.strip(),
            )
        )

    def render(self, counter: TokenCounter, token_budget: int) -> str:
        if not self._turns or token_budget <= 0:
            return "No prior grounded conversation."

        selected_blocks: list[str] = []
        used_tokens = 0
        for turn in reversed(self._turns):
            block = f"User: {turn.user_query}\nAssistant: {turn.assistant_response}"
            block_tokens = counter.count(block)
            if selected_blocks and used_tokens + block_tokens > token_budget:
                break
            if not selected_blocks and block_tokens > token_budget:
                continue
            selected_blocks.append(block)
            used_tokens += block_tokens

        if not selected_blocks:
            return "No prior grounded conversation."

        return "\n\n".join(reversed(selected_blocks))

    @property
    def size(self) -> int:
        return len(self._turns)
