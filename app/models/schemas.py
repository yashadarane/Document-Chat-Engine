from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import ProviderName


class DocumentMetadata(BaseModel):
    """Metadata shown after documents are processed."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    name: str
    document_type: str
    page_count: int
    token_count: int
    chunk_count: int
    warnings: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response returned after session creation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    documents: list[DocumentMetadata]
    combined_preview: str
    warnings: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response returned after a question is asked."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    answer: str
    grounded: bool
    fallback_used: bool
    backend: str
    requested_provider: ProviderName
    history_size: int
    context_excerpt: str
    warnings: list[str] = Field(default_factory=list)
