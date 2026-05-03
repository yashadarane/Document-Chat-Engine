from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings, load_settings
from app.core.logging import configure_logging
from app.services.chat_engine import ChatEngineService
from app.services.context_builder import ContextBuilder
from app.services.document_ingestion import DocumentIngestionService
from app.services.llm import BaseTextProvider, ProviderLLMService
from app.services.prompt_engine import PromptEngine
from app.services.session_store import SessionStore
from app.utils.tokens import RegexTokenCounter


def create_chat_engine(
    settings: Settings | None = None,
    *,
    ocr_engine: Any | None = None,
    local_backend: BaseTextProvider | None = None,
    gemini_backend: BaseTextProvider | None = None,
    groq_backend: BaseTextProvider | None = None,
) -> ChatEngineService:
    """Construct the chat engine service used by the Streamlit frontend and tests."""

    settings = settings or load_settings()
    configure_logging(settings.app.log_level)
    logger = logging.getLogger("document_chat_engine")
    logger.info("Initializing document chat engine.", extra={"event": "startup"})

    token_counter = RegexTokenCounter()
    prompt_engine = PromptEngine()
    context_builder = ContextBuilder(settings, token_counter)
    ingestion_service = DocumentIngestionService(settings, token_counter, context_builder, ocr_engine=ocr_engine)
    session_store = SessionStore(memory_turns=settings.context.memory_turns)
    llm_service = ProviderLLMService(
        settings,
        prompt_engine,
        token_counter,
        local_backend=local_backend,
        gemini_backend=gemini_backend,
        groq_backend=groq_backend,
    )
    chat_engine = ChatEngineService(
        settings,
        ingestion_service,
        context_builder,
        session_store,
        llm_service,
    )
    return chat_engine
