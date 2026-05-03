from __future__ import annotations

from app.main import create_chat_engine
from app.services.document_ingestion import IncomingFile


def test_full_pipeline_upload_then_chat(chat_engine, sample_pdf_bytes):
    upload_response = chat_engine.create_session(
        [IncomingFile(filename="policy.pdf", content_type="application/pdf", payload=sample_pdf_bytes)]
    )

    assert upload_response.session_id
    assert upload_response.documents[0].name == "policy.pdf"

    chat_response = chat_engine.chat(upload_response.session_id, "What is the policy number?", provider_name="local")

    assert "AZ-42" in chat_response.answer
    assert chat_response.grounded is True
    assert chat_response.requested_provider == "local"


def test_empty_file_rejected(chat_engine):
    try:
        chat_engine.create_session([IncomingFile(filename="empty.pdf", content_type="application/pdf", payload=b"")])
    except Exception as exc:
        assert exc.error_code == "empty_file"
    else:
        raise AssertionError("Expected empty file validation to fail.")


def test_corrupt_pdf_rejected(chat_engine):
    try:
        chat_engine.create_session(
            [IncomingFile(filename="broken.pdf", content_type="application/pdf", payload=b"not-a-real-pdf")]
        )
    except Exception as exc:
        assert exc.error_code == "document_parsing_failed"
    else:
        raise AssertionError("Expected corrupt PDF parsing to fail.")


def test_irrelevant_query_returns_refusal(chat_engine, sample_pdf_bytes):
    upload_response = chat_engine.create_session(
        [IncomingFile(filename="policy.pdf", content_type="application/pdf", payload=sample_pdf_bytes)]
    )

    chat_response = chat_engine.chat(upload_response.session_id, "What is the weather tomorrow?", provider_name="gemini")

    assert chat_response.answer == "I can't answer that from the provided documents."
    assert chat_response.grounded is False
    assert chat_response.requested_provider == "gemini"


def test_fallback_uses_secondary_provider(settings, failing_local_backend, gemini_dummy_backend, fake_ocr_engine, sample_pdf_bytes):
    settings.routing.enable_fallback = True
    chat_engine = create_chat_engine(
        settings,
        local_backend=failing_local_backend,
        gemini_backend=gemini_dummy_backend,
        ocr_engine=fake_ocr_engine,
    )
    upload_response = chat_engine.create_session(
        [IncomingFile(filename="policy.pdf", content_type="application/pdf", payload=sample_pdf_bytes)]
    )

    chat_response = chat_engine.chat(
        upload_response.session_id,
        "What is the policy number?",
        provider_name="local",
        fallback_enabled=True,
    )

    assert "AZ-42" in chat_response.answer
    assert chat_response.fallback_used is True
    assert chat_response.backend == "gemini:test"


def test_local_summary_fallback_uses_extractive_preview(chat_engine, sample_pdf_bytes):
    upload_response = chat_engine.create_session(
        [IncomingFile(filename="policy.pdf", content_type="application/pdf", payload=sample_pdf_bytes)]
    )

    chat_response = chat_engine.chat(
        upload_response.session_id,
        "Give me a summary of the document",
        provider_name="local",
    )

    assert "grounded summary" in chat_response.answer.lower()
    assert "Policy Number: AZ-42" in chat_response.answer
    assert chat_response.fallback_used is True
