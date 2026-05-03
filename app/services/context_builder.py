from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from app.core.config import Settings
from app.models.domain import DocumentChunk, ParsedDocument, PreparedContext
from app.utils.text import extractive_summarize, split_into_paragraphs, split_into_sentences
from app.utils.tokens import TokenCounter, take_tail_words, truncate_text_to_budget


@dataclass(slots=True)
class ChunkUnit:
    """Small normalized source segment used to form chunks."""

    page_number: int
    text: str


class ContextBuilder:
    """Build document chunks and assemble question-specific prompt context."""

    def __init__(self, settings: Settings, token_counter: TokenCounter) -> None:
        self.settings = settings
        self.token_counter = token_counter

    def build_document_chunks(self, document: ParsedDocument) -> list[DocumentChunk]:
        units = self._build_units(document)
        if not units:
            return []

        chunk_size = self.settings.context.chunk_size_tokens
        overlap_tokens = self.settings.context.chunk_overlap_tokens
        chunks: list[DocumentChunk] = []
        current_parts: list[str] = []
        current_pages: list[int] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.token_counter.count(unit.text)
            if current_parts and current_tokens + unit_tokens > chunk_size:
                finalized = self._finalize_chunk(document, current_parts, current_pages)
                chunks.append(finalized)
                overlap_text = take_tail_words(finalized.text, overlap_tokens)
                current_parts = [overlap_text] if overlap_text else []
                current_pages = [finalized.page_end] if overlap_text else []
                current_tokens = self.token_counter.count(overlap_text)

            current_parts.append(unit.text)
            current_pages.append(unit.page_number)
            current_tokens += unit_tokens

        if current_parts:
            chunks.append(self._finalize_chunk(document, current_parts, current_pages))

        return chunks

    def preview_documents(self, documents: list[ParsedDocument], max_tokens: int = 600) -> str:
        sections: list[str] = []
        for document in documents:
            summary = extractive_summarize(
                document.cleaned_text,
                max_sentences=4,
                max_chars=max_tokens * 4,
            )
            sections.append(
                f"[Document: {document.name} | Type: {document.document_type} | Pages: {document.page_count}]\n"
                f"{summary}"
            )

        return truncate_text_to_budget("\n\n".join(sections).strip(), max_tokens, self.token_counter)

    def prepare_context(self, documents: list[ParsedDocument], query: str) -> PreparedContext:
        ranked_chunks = self._rank_chunks(documents, query)
        ranked_chunks = self._deduplicate_chunks(ranked_chunks)
        warnings: list[str] = []
        selected_chunks: list[DocumentChunk] = []
        sections: list[str] = []
        used_tokens = 0
        token_budget = self.settings.context.max_prompt_context_tokens

        for chunk in ranked_chunks[: self.settings.context.max_chunks_for_answer]:
            rendered = self._render_chunk(chunk)
            chunk_tokens = self.token_counter.count(rendered)
            if selected_chunks and used_tokens + chunk_tokens > token_budget:
                break
            if not selected_chunks and chunk_tokens > token_budget:
                rendered = truncate_text_to_budget(rendered, token_budget, self.token_counter)
                chunk_tokens = self.token_counter.count(rendered)
            selected_chunks.append(chunk)
            sections.append(rendered)
            used_tokens += chunk_tokens

        if not sections:
            warnings.append("No relevant document chunks matched the question.")
            fallback_text = "No relevant document context matched the question."
            return PreparedContext(
                text=fallback_text,
                token_count=self.token_counter.count(fallback_text),
                selected_chunks=[],
                warnings=warnings,
            )

        context_text = "\n\n".join(sections).strip()
        if len(ranked_chunks) > len(selected_chunks):
            warnings.append("Only the highest-priority document chunks were included to stay within token limits.")

        return PreparedContext(
            text=context_text,
            token_count=self.token_counter.count(context_text),
            selected_chunks=selected_chunks,
            warnings=warnings,
        )

    def _rank_chunks(self, documents: list[ParsedDocument], query: str) -> list[DocumentChunk]:
        query_terms = self._important_terms(query)
        chunks = [chunk for document in documents for chunk in document.chunks]
        if not query_terms:
            return chunks

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            score = self._score_chunk(chunk.text, query_terms)
            if score > 0:
                scored.append((score, chunk))

        if not scored:
            return chunks

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored]

    @staticmethod
    def _deduplicate_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        seen: set[str] = set()
        unique_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            text = chunk.text.strip()
            if text in seen:
                continue
            seen.add(text)
            unique_chunks.append(chunk)
        return unique_chunks

    def _build_units(self, document: ParsedDocument) -> list[ChunkUnit]:
        chunk_size = self.settings.context.chunk_size_tokens
        units: list[ChunkUnit] = []

        for page in document.pages:
            paragraphs = split_into_paragraphs(page.text) or ([page.text] if page.text else [])
            for paragraph in paragraphs:
                if self.token_counter.count(paragraph) <= chunk_size:
                    units.append(ChunkUnit(page_number=page.page_number, text=paragraph))
                else:
                    units.extend(self._split_large_text(paragraph, page.page_number, chunk_size))

        return units

    def _split_large_text(self, text: str, page_number: int, chunk_size: int) -> list[ChunkUnit]:
        sentences = split_into_sentences(text) or [text]
        units: list[ChunkUnit] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if current and self.token_counter.count(candidate) > chunk_size:
                units.append(ChunkUnit(page_number=page_number, text=current))
                current = sentence
            else:
                current = candidate

        if current:
            units.append(ChunkUnit(page_number=page_number, text=current))

        return units

    def _finalize_chunk(
        self,
        document: ParsedDocument,
        text_parts: Iterable[str],
        page_numbers: Iterable[int],
    ) -> DocumentChunk:
        pages = list(page_numbers)
        text = "\n".join(part for part in text_parts if part).strip()
        return DocumentChunk(
            chunk_id=uuid4().hex,
            document_id=document.document_id,
            document_name=document.name,
            page_start=min(pages),
            page_end=max(pages),
            text=text,
            token_count=self.token_counter.count(text),
        )

    @staticmethod
    def _render_chunk(chunk: DocumentChunk) -> str:
        return f"[Document: {chunk.document_name} | Pages: {chunk.page_label}]\n{chunk.text}"

    @staticmethod
    def _important_terms(text: str) -> set[str]:
        stopwords = {
            "about",
            "also",
            "and",
            "are",
            "can",
            "did",
            "does",
            "for",
            "from",
            "give",
            "how",
            "into",
            "last",
            "list",
            "many",
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
            "year",
        }
        terms = {
            token.lower().strip(".,?!:;()[]{}")
            for token in text.split()
            if len(token.strip(".,?!:;()[]{}")) > 2 and token.lower().strip(".,?!:;()[]{}") not in stopwords
        }
        expanded = set(terms)
        synonyms = {
            "days": {"leave", "leaves", "holiday", "holidays", "vacation", "off", "entitlement"},
            "off": {"leave", "leaves", "holiday", "holidays", "vacation", "days", "entitlement"},
            "leave": {"leaves", "vacation", "holiday", "holidays", "days", "off", "entitlement"},
            "leaves": {"leave", "vacation", "holiday", "holidays", "days", "off", "entitlement"},
            "holiday": {"holidays", "leave", "vacation", "days", "off"},
            "holidays": {"holiday", "leave", "vacation", "days", "off"},
            "vacation": {"leave", "leaves", "holiday", "holidays", "days", "off"},
            "salary": {"pay", "compensation", "wage", "income"},
            "pay": {"salary", "compensation", "wage", "income"},
            "premium": {"amount", "payment", "cost"},
            "policy": {"plan", "coverage", "document"},
            "claim": {"case", "request"},
        }
        for term in terms:
            expanded.update(synonyms.get(term, set()))
        return expanded

    def _score_chunk(self, text: str, query_terms: set[str]) -> float:
        chunk_terms = [token.lower().strip(".,?!:;()[]{}") for token in text.split()]
        counts = Counter(term for term in chunk_terms if term)
        overlap = sum(counts.get(term, 0) for term in query_terms)
        density_bonus = overlap / max(len(chunk_terms), 1)
        return overlap + density_bonus

    @staticmethod
    def _looks_like_summary_or_overview(query: str) -> bool:
        normalized = query.lower()
        markers = ("summary", "summarize", "summarise", "overview", "gist", "brief")
        return any(marker in normalized for marker in markers)
