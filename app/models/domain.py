from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.memory import ConversationMemory


@dataclass(slots=True)
class ParsedPage:
    """A single parsed page or image frame."""

    page_number: int
    text: str
    token_count: int


@dataclass(slots=True)
class DocumentChunk:
    """A chunk retained for question-time ranking."""

    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    token_count: int

    @property
    def page_label(self) -> str:
        return str(self.page_start) if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"


@dataclass(slots=True)
class ParsedDocument:
    """Parsed and cleaned document content."""

    document_id: str
    name: str
    document_type: str
    pages: list[ParsedPage]
    cleaned_text: str
    token_count: int
    chunks: list[DocumentChunk]
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(slots=True)
class PreparedContext:
    """Context prepared for a specific user question."""

    text: str
    token_count: int
    selected_chunks: list[DocumentChunk]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConversationTurn:
    """A single conversation turn."""

    user_query: str
    assistant_response: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class GenerationResult:
    """Normalized model output."""

    answer: str
    grounded: bool
    fallback_used: bool
    backend: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    """Per-session documents and short-term memory."""

    session_id: str
    documents: list[ParsedDocument]
    memory: ConversationMemory
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
