from __future__ import annotations

import re
from io import BytesIO

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.core.config import Settings
from app.main import create_chat_engine
from app.services.chat_engine import ChatEngineService
from app.services.llm import BaseTextProvider
from app.services.prompt_engine import PromptEngine
from app.utils.tokens import RegexTokenCounter


class DummyProvider(BaseTextProvider):
    """Deterministic provider used by tests."""

    def __init__(self, provider_name: str, token_counter=None, *, fail: bool = False) -> None:
        super().__init__(token_counter=token_counter)
        self._provider_name = provider_name
        self._fail = fail

    @property
    def name(self) -> str:
        return self._provider_name

    def is_ready(self) -> bool:
        return True

    def load(self) -> None:
        if self._fail:
            raise RuntimeError(f"{self._provider_name} failed to load")

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        del max_new_tokens, temperature, top_p, repetition_penalty
        if self._fail:
            raise RuntimeError(f"{self._provider_name} failed to generate")

        query_match = re.search(r"USER QUESTION:\n(.*?)\n\nANSWER FORMAT:", prompt, flags=re.DOTALL)
        query = query_match.group(1).strip().lower() if query_match else ""

        if "weather" in query:
            return PromptEngine.REFUSAL_TEXT

        if "policy number" in query:
            match = re.search(r"Policy Number:\s*([A-Z0-9-]+)", prompt, flags=re.IGNORECASE)
            if match:
                value = match.group(1)
                return f"Answer: The policy number is {value}.\nEvidence: Policy Number: {value}"

        if "premium" in query:
            match = re.search(r"Premium:\s*([^\n]+)", prompt, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                return f"Answer: The premium is {value}.\nEvidence: Premium: {value}"

        if "customer" in query:
            match = re.search(r"Customer:\s*([^\n]+)", prompt, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                return f"Answer: The customer is {value}.\nEvidence: Customer: {value}"

        return PromptEngine.REFUSAL_TEXT


def create_pdf_bytes(page_texts: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    for page_text in page_texts:
        y_position = 750
        for line in page_text.splitlines():
            pdf.drawString(72, y_position, line)
            y_position -= 18
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def create_image_bytes() -> bytes:
    image = Image.new("RGB", (600, 200), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def settings() -> Settings:
    return Settings.model_validate(
        {
            "app": {
                "name": "Document Chat Engine Test",
                "version": "test",
                "environment": "test",
                "log_level": "INFO",
                "max_upload_files": 3,
                "max_pages_per_file": 5,
                "max_file_size_mb": 5,
                "allowed_extensions": [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
            },
            "context": {
                "chunk_size_tokens": 80,
                "chunk_overlap_tokens": 10,
                "max_prompt_context_tokens": 240,
                "max_chunks_for_answer": 4,
                "memory_turns": 3,
                "memory_token_budget": 80,
                "extractive_summary_tokens": 120,
            },
            "routing": {
                "default_provider": "local",
                "allow_ui_provider_override": True,
                "enable_fallback": False,
                "fallback_provider": "gemini",
            },
            "local_llm": {
                "backend": "transformers",
                "model_name": "dummy/model",
                "device": "cpu",
                "max_context_tokens": 512,
                "max_new_tokens": 128,
                "load_timeout_seconds": 5,
                "generation_timeout_seconds": 5,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            },
            "gemini": {
                "model_name": "gemini-test",
                "api_key_env": "GEMINI_API_KEY",
                "max_context_tokens": 512,
                "max_new_tokens": 128,
                "generation_timeout_seconds": 5,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "ocr": {
                "provider": "auto",
                "language": "eng",
                "tesseract_cmd": None,
                "gemini_model_name": None,
                "gemini_max_output_tokens": 2048,
            },
        }
    )


@pytest.fixture()
def token_counter() -> RegexTokenCounter:
    return RegexTokenCounter()


@pytest.fixture()
def local_dummy_backend(token_counter: RegexTokenCounter) -> DummyProvider:
    return DummyProvider("local:test", token_counter=token_counter)


@pytest.fixture()
def gemini_dummy_backend(token_counter: RegexTokenCounter) -> DummyProvider:
    return DummyProvider("gemini:test", token_counter=token_counter)


@pytest.fixture()
def failing_local_backend(token_counter: RegexTokenCounter) -> DummyProvider:
    return DummyProvider("local:test", token_counter=token_counter, fail=True)


@pytest.fixture()
def fake_ocr_engine():
    return lambda _image, _language: "Customer: Jane Doe\nClaim Number: CLM-009"


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    return create_pdf_bytes(
        [
            "Customer: Jane Doe\nPolicy Number: AZ-42\nPremium: 500 USD",
            "Coverage: Comprehensive\nStatus: Active",
        ]
    )


@pytest.fixture()
def sample_image_bytes() -> bytes:
    return create_image_bytes()


@pytest.fixture()
def chat_engine(settings: Settings, local_dummy_backend: DummyProvider, gemini_dummy_backend: DummyProvider, fake_ocr_engine) -> ChatEngineService:
    return create_chat_engine(
        settings,
        local_backend=local_dummy_backend,
        gemini_backend=gemini_dummy_backend,
        ocr_engine=fake_ocr_engine,
    )
