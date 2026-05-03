from __future__ import annotations

from app.models.domain import DocumentChunk, ParsedDocument, ParsedPage
from app.services.context_builder import ContextBuilder


def test_context_builder_ranks_relevant_chunks(settings, token_counter):
    document = ParsedDocument(
        document_id="doc-1",
        name="policy.pdf",
        document_type="pdf",
        pages=[
            ParsedPage(page_number=1, text="Customer: Jane Doe\nPolicy Number: AZ-42", token_count=10),
            ParsedPage(page_number=2, text="Coverage: Comprehensive\nStatus: Active", token_count=10),
        ],
        cleaned_text="Customer: Jane Doe\nPolicy Number: AZ-42\n\nCoverage: Comprehensive\nStatus: Active",
        token_count=20,
        chunks=[],
    )

    builder = ContextBuilder(settings, token_counter)
    document.chunks = builder.build_document_chunks(document)
    prepared = builder.prepare_context([document], "What is the policy number?")

    assert prepared.text
    assert "Policy Number: AZ-42" in prepared.text
    assert prepared.token_count <= settings.context.max_prompt_context_tokens


def test_context_builder_uses_fallback_chunks_when_no_terms_match(settings, token_counter):
    document = ParsedDocument(
        document_id="doc-1",
        name="handbook.pdf",
        document_type="pdf",
        pages=[
            ParsedPage(page_number=1, text="Annual leave entitlement is 20 paid days per year.", token_count=10),
            ParsedPage(page_number=2, text="Sick leave is available for up to 10 paid days per year.", token_count=12),
        ],
        cleaned_text="Annual leave entitlement is 20 paid days per year.\n\nSick leave is available for up to 10 paid days per year.",
        token_count=22,
        chunks=[],
    )

    builder = ContextBuilder(settings, token_counter)
    document.chunks = builder.build_document_chunks(document)
    prepared = builder.prepare_context([document], "How many days off do employees get?")

    assert prepared.selected_chunks
    assert "Annual leave entitlement" in prepared.text

    unrelated = builder.prepare_context([document], "What is the cafeteria menu tomorrow?")
    assert unrelated.selected_chunks
    assert "Annual leave entitlement" in unrelated.text


def test_context_builder_deduplicates_repeated_chunk_text(settings, token_counter):
    document = ParsedDocument(
        document_id="doc-1",
        name="handbook.pdf",
        document_type="pdf",
        pages=[
            ParsedPage(page_number=1, text="Annual leave: 20 days. Sick Leave: 10 days.", token_count=8),
            ParsedPage(page_number=2, text="Annual leave: 20 days. Sick Leave: 10 days.", token_count=8),
        ],
        cleaned_text="Annual leave: 20 days. Sick Leave: 10 days.\n\nAnnual leave: 20 days. Sick Leave: 10 days.",
        token_count=16,
        chunks=[],
    )

    builder = ContextBuilder(settings, token_counter)
    duplicate_text = "Annual leave: 20 days. Sick Leave: 10 days."
    document.chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            document_name=document.name,
            page_start=1,
            page_end=1,
            text=duplicate_text,
            token_count=token_counter.count(duplicate_text),
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id=document.document_id,
            document_name=document.name,
            page_start=2,
            page_end=2,
            text=duplicate_text,
            token_count=token_counter.count(duplicate_text),
        ),
    ]
    prepared = builder.prepare_context([document], "How many leave days?")

    assert prepared.text.count("Annual leave: 20 days. Sick Leave: 10 days.") == 1
