from __future__ import annotations

from app.core.config import ProviderName, Settings
from app.core.exceptions import GenerationTimeoutError, ModelUnavailableError, ProviderConfigurationError
from app.models.domain import GenerationResult
from app.models.schemas import ChatResponse, DocumentMetadata, UploadResponse
from app.services.context_builder import ContextBuilder
from app.services.document_ingestion import DocumentIngestionService, IncomingFile
from app.services.llm import ProviderLLMService
from app.services.session_store import SessionStore
from app.utils.text import WORD_PATTERN, split_into_sentences


class ChatEngineService:
    """Coordinates upload processing, session state, and local model calls."""

    def __init__(
        self,
        settings: Settings,
        ingestion_service: DocumentIngestionService,
        context_builder: ContextBuilder,
        session_store: SessionStore,
        llm_service: ProviderLLMService,
    ) -> None:
        self.settings = settings
        self.ingestion_service = ingestion_service
        self.context_builder = context_builder
        self.session_store = session_store
        self.llm_service = llm_service

    def create_session(self, files: list[IncomingFile]) -> UploadResponse:
        documents = self.ingestion_service.ingest_files(files)
        session = self.session_store.create_session(documents)
        warnings: list[str] = []
        for document in documents:
            warnings.extend(document.warnings)

        return UploadResponse(
            session_id=session.session_id,
            combined_preview=self.context_builder.preview_documents(documents),
            warnings=self._deduplicate(warnings),
            documents=[
                DocumentMetadata(
                    document_id=document.document_id,
                    name=document.name,
                    document_type=document.document_type,
                    page_count=document.page_count,
                    token_count=document.token_count,
                    chunk_count=len(document.chunks),
                    warnings=document.warnings,
                )
                for document in documents
            ],
        )

    def chat(
        self,
        session_id: str,
        query: str,
        *,
        provider_name: ProviderName | None = None,
        fallback_enabled: bool | None = None,
    ) -> ChatResponse:
        session = self.session_store.get_session(session_id)
        prepared_context = self.context_builder.prepare_context(session.documents, query)
        history = session.memory.render(
            self.llm_service.token_counter,
            self.settings.context.memory_token_budget,
        )
        selected_provider = provider_name or self.settings.routing.default_provider
        use_fallback = self.settings.routing.enable_fallback if fallback_enabled is None else fallback_enabled

        try:
            generation = self.llm_service.generate_response(
                prepared_context.text,
                query,
                history,
                provider_name=selected_provider,
                fallback_enabled=use_fallback,
                fallback_provider_name=self.settings.routing.fallback_provider,
            )
        except GenerationTimeoutError as exc:
            generation = self._timeout_fallback(selected_provider, exc.message)
        except (ModelUnavailableError, ProviderConfigurationError) as exc:
            generation = self._unavailable_fallback(selected_provider, exc.message, fallback_used=False)

        generation = self._maybe_apply_summary_fallback(
            generation=generation,
            query=query,
            provider_name=selected_provider,
            documents=session.documents,
        )
        generation = self._maybe_apply_extractive_context_fallback(
            generation=generation,
            query=query,
            context=prepared_context.text,
        )

        session.memory.add_turn(query, generation.answer)
        warnings = self._deduplicate([*prepared_context.warnings, *generation.warnings])

        return ChatResponse(
            session_id=session_id,
            answer=generation.answer,
            grounded=generation.grounded,
            fallback_used=generation.fallback_used,
            history_size=session.memory.size,
            backend=generation.backend,
            requested_provider=selected_provider,
            context_excerpt=prepared_context.text,
            warnings=warnings,
        )

    def prepare_provider(self, provider_name: ProviderName) -> None:
        """Warm the selected provider before the first question."""

        self.llm_service.prepare_provider(provider_name)

    def _timeout_fallback(self, provider_name: ProviderName, reason: str | None = None) -> GenerationResult:
        warning = reason or f"Provider '{provider_name}' response timed out."
        return GenerationResult(
            answer=self.llm_service.prompt_engine.TIMEOUT_TEXT,
            grounded=False,
            fallback_used=True,
            backend=provider_name,
            warnings=[warning],
        )

    def _unavailable_fallback(
        self,
        provider_name: ProviderName,
        reason: str | None = None,
        *,
        fallback_used: bool = True,
    ) -> GenerationResult:
        warning = reason or f"Provider '{provider_name}' is unavailable or misconfigured."
        return GenerationResult(
            answer=self.llm_service.prompt_engine.UNAVAILABLE_TEXT,
            grounded=False,
            fallback_used=fallback_used,
            backend=provider_name,
            warnings=[warning],
        )

    def _maybe_apply_summary_fallback(
        self,
        *,
        generation: GenerationResult,
        query: str,
        provider_name: ProviderName,
        documents,
    ) -> GenerationResult:
        if not self._looks_like_summary_query(query):
            return generation

        answer_text = (generation.answer or "").strip()
        if answer_text and answer_text != self.llm_service.prompt_engine.REFUSAL_TEXT:
            return generation

        summary = self.context_builder.preview_documents(
            documents,
            max_tokens=self.settings.context.extractive_summary_tokens,
        ).strip()
        if not summary:
            return generation

        return GenerationResult(
            answer=f"Here is a grounded summary of the uploaded documents:\n\n{summary}",
            grounded=True,
            fallback_used=True,
            backend=f"{generation.backend}:summary_fallback",
            warnings=[*generation.warnings, "Used extractive summary fallback because the selected provider did not return a usable summary."],
        )

    def _maybe_apply_extractive_context_fallback(
        self,
        *,
        generation: GenerationResult,
        query: str,
        context: str,
    ) -> GenerationResult:
        answer_text = (generation.answer or "").strip()
        if answer_text and answer_text != self.llm_service.prompt_engine.REFUSAL_TEXT and generation.grounded:
            return generation

        if not self._looks_like_exact_lookup_query(query):
            return generation

        evidence = self._best_evidence_sentence(query, context)
        if not evidence:
            return generation

        return GenerationResult(
            answer=evidence,
            grounded=True,
            fallback_used=True,
            backend=f"{generation.backend}:extractive_fallback",
            warnings=[*generation.warnings, "Used a direct document lookup because the model did not return a usable grounded answer."],
        )

    @classmethod
    def _best_evidence_sentence(cls, query: str, context: str) -> str:
        query_terms = cls._query_terms(query)
        if not query_terms:
            return ""

        candidates = split_into_sentences(context)
        if not candidates:
            candidates = [line.strip() for line in context.splitlines() if line.strip() and not line.startswith("[Document:")]

        best_sentence = ""
        best_score = 0
        for sentence in candidates:
            sentence_terms = {token.lower() for token in WORD_PATTERN.findall(sentence) if len(token) > 2}
            score = len(query_terms & sentence_terms)
            if score > best_score:
                best_sentence = sentence
                best_score = score

        minimum_score = 2 if len(query_terms) > 1 else 1
        if best_score < minimum_score:
            return ""
        return best_sentence[:900].strip()

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        stopwords = {
            "about",
            "also",
            "are",
            "can",
            "did",
            "does",
            "from",
            "give",
            "how",
            "into",
            "is",
            "me",
            "of",
            "please",
            "show",
            "tell",
            "that",
            "the",
            "this",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
        return {
            token.lower()
            for token in WORD_PATTERN.findall(query)
            if len(token) > 2 and token.lower() not in stopwords
        }

    @staticmethod
    def _looks_like_exact_lookup_query(query: str) -> bool:
        normalized = query.lower()
        exact_markers = (
            "number",
            "id",
            "date",
            "name",
            "amount",
            "premium",
            "status",
            "email",
            "phone",
            "address",
            "policy",
            "claim",
            "customer",
        )
        broad_markers = ("summarize", "summary", "overview", "explain", "describe", "activities", "implemented")
        return any(marker in normalized for marker in exact_markers) and not any(
            marker in normalized for marker in broad_markers
        )

    @staticmethod
    def _looks_like_summary_query(query: str) -> bool:
        normalized = query.lower()
        summary_markers = ("summary", "summarize", "summarise", "overview", "gist", "brief")
        return any(marker in normalized for marker in summary_markers)

    @staticmethod
    def _deduplicate(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered
