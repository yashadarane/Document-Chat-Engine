from __future__ import annotations

import pytest

from app.core.exceptions import DocumentParsingError, OCRExtractionError
from app.services.context_builder import ContextBuilder
from app.services.document_ingestion import DocumentIngestionService, IncomingFile


def test_parse_pdf_extracts_text(settings, token_counter, sample_pdf_bytes):
    service = DocumentIngestionService(settings, token_counter, ContextBuilder(settings, token_counter))

    documents = service.ingest_files(
        [IncomingFile(filename="policy.pdf", content_type="application/pdf", payload=sample_pdf_bytes)]
    )

    assert len(documents) == 1
    assert documents[0].document_type == "pdf"
    assert "Policy Number: AZ-42" in documents[0].cleaned_text
    assert documents[0].page_count == 2
    assert documents[0].chunks


def test_parse_image_uses_ocr(settings, token_counter, sample_image_bytes, fake_ocr_engine):
    service = DocumentIngestionService(
        settings,
        token_counter,
        ContextBuilder(settings, token_counter),
        ocr_engine=fake_ocr_engine,
    )

    documents = service.ingest_files(
        [IncomingFile(filename="claim.png", content_type="image/png", payload=sample_image_bytes)]
    )

    assert len(documents) == 1
    assert documents[0].document_type == "image"
    assert "Claim Number: CLM-009" in documents[0].cleaned_text


def test_corrupt_pdf_raises(settings, token_counter):
    service = DocumentIngestionService(settings, token_counter, ContextBuilder(settings, token_counter))

    with pytest.raises(DocumentParsingError):
        service.ingest_files(
            [IncomingFile(filename="broken.pdf", content_type="application/pdf", payload=b"not-a-real-pdf")]
        )


def test_ocr_failure_is_wrapped(settings, token_counter, sample_image_bytes):
    def failing_ocr(_image, _language):
        raise RuntimeError("ocr failed")

    service = DocumentIngestionService(
        settings,
        token_counter,
        ContextBuilder(settings, token_counter),
        ocr_engine=failing_ocr,
    )

    with pytest.raises(OCRExtractionError):
        service.ingest_files(
            [IncomingFile(filename="claim.png", content_type="image/png", payload=sample_image_bytes)]
        )


def test_auto_ocr_falls_back_to_gemini_when_tesseract_returns_empty(settings, token_counter, sample_image_bytes, monkeypatch):
    service = DocumentIngestionService(settings, token_counter, ContextBuilder(settings, token_counter))
    monkeypatch.setenv(settings.gemini.api_key_env, "test-key")
    monkeypatch.setattr(service, "_extract_text_via_tesseract", lambda _image: "")
    monkeypatch.setattr(service, "_extract_text_via_gemini", lambda _image: "Policy Number: IMG-77")

    documents = service.ingest_files(
        [IncomingFile(filename="scan.png", content_type="image/png", payload=sample_image_bytes)]
    )

    assert documents[0].document_type == "image"
    assert "Policy Number: IMG-77" in documents[0].cleaned_text
