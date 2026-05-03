from __future__ import annotations

import re
from typing import Iterable, Protocol

from app.utils.text import split_into_sentences


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Estimate or count tokens for a string."""


class RegexTokenCounter:
    """A lightweight token estimator that does not depend on model-specific tokenizers."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(TOKEN_PATTERN.findall(text))

    def count_many(self, parts: Iterable[str]) -> int:
        return sum(self.count(part) for part in parts)


def take_tail_words(text: str, approx_token_budget: int) -> str:
    """Return the tail of a string for overlap preservation."""

    if approx_token_budget <= 0 or not text:
        return ""

    words = text.split()
    if len(words) <= approx_token_budget:
        return text
    return " ".join(words[-approx_token_budget:])


def truncate_text_to_budget(text: str, token_budget: int, counter: TokenCounter) -> str:
    """Truncate text to a target token budget with sentence-first behavior."""

    if token_budget <= 0 or not text:
        return ""

    if counter.count(text) <= token_budget:
        return text

    sentences = split_into_sentences(text)
    selected: list[str] = []
    used_tokens = 0
    for sentence in sentences:
        sentence_tokens = counter.count(sentence)
        if selected and used_tokens + sentence_tokens > token_budget:
            break
        if not selected and sentence_tokens > token_budget:
            break
        selected.append(sentence)
        used_tokens += sentence_tokens

    if selected:
        return " ".join(selected)

    words = text.split()
    while words and counter.count(" ".join(words)) > token_budget:
        words.pop()
    return " ".join(words)
