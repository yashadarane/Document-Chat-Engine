from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import (
    DocumentParsingError,
    FileCountLimitError,
    InvalidUploadError,
    OCRExtractionError,
    PageLimitError,
)
from app.models.domain import ParsedDocument, ParsedPage
from app.services.context_builder import ContextBuilder
from app.utils.files import (
    is_image_extension,
    is_pdf_extension,
    validate_file_metadata,
)
from app.utils.text import clean_extracted_text
from app.utils.tokens import TokenCounter


OCRCallable = Callable[[Any, str], str]


@dataclass(slots=True)
class IncomingFile:
    """Serializable file input passed from the API layer into the ingestion service."""

    filename: str | None
    content_type: str | None
    payload: bytes


class DocumentIngestionService:
    """Parses supported uploads into normalized document objects."""

    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        context_builder: ContextBuilder,
        *,
        ocr_engine: OCRCallable | None = None,
    ) -> None:
        self.settings = settings
        self.token_counter = token_counter
        self.context_builder = context_builder
        self._ocr_engine = ocr_engine
        self.logger = logging.getLogger(self.__class__.__name__)

    def ingest_files(self, files: Sequence[IncomingFile]) -> list[ParsedDocument]:
        if not files:
            raise InvalidUploadError("At least one file must be uploaded.")

        if len(files) > self.settings.app.max_upload_files:
            raise FileCountLimitError(
                f"A maximum of {self.settings.app.max_upload_files} files can be uploaded per request.",
                details={"max_upload_files": self.settings.app.max_upload_files},
            )

        documents: list[ParsedDocument] = []
        for file in files:
            validated = validate_file_metadata(
                file.filename,
                file.content_type,
                file.payload,
                max_size_bytes=self.settings.app.max_file_size_bytes,
                allowed_extensions=self.settings.app.allowed_extensions,
            )
            if is_pdf_extension(validated.extension):
                documents.append(self._parse_pdf(validated.filename, file.payload))
            elif is_image_extension(validated.extension):
                documents.append(self._parse_image(validated.filename, file.payload))

        return documents

    def _parse_pdf(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            import pdfplumber
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DocumentParsingError(
                "PDF dependencies are not installed. Install pdfplumber and pypdf to process PDFs.",
            ) from exc

        try:
            reader = PdfReader(io.BytesIO(payload))
            page_count = len(reader.pages)
        except Exception as exc:
            raise DocumentParsingError(f"Failed to open PDF '{filename}'.") from exc

        if page_count == 0:
            raise DocumentParsingError(f"PDF '{filename}' does not contain any pages.")

        if page_count > self.settings.app.max_pages_per_file:
            raise PageLimitError(
                f"PDF '{filename}' exceeds the {self.settings.app.max_pages_per_file} page limit.",
                details={"filename": filename, "page_count": page_count},
            )

        extracted_texts: list[str] = []
        try:
            with pdfplumber.open(io.BytesIO(payload)) as pdf:
                extracted_texts = [(page.extract_text() or "") for page in pdf.pages]
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse PDF '{filename}'.") from exc

        if not any(text.strip() for text in extracted_texts):
            extracted_texts = [(page.extract_text() or "") for page in reader.pages]

        warnings: list[str] = []
        pages: list[ParsedPage] = []
        for page_number, raw_text in enumerate(extracted_texts, start=1):
            cleaned = clean_extracted_text(raw_text)
            if not cleaned:
                warnings.append(f"No extractable text found on page {page_number}.")
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=cleaned,
                    token_count=self.token_counter.count(cleaned),
                )
            )

        if not any(page.text for page in pages):
            raise DocumentParsingError(f"No text could be extracted from PDF '{filename}'.")

        cleaned_text = "\n\n".join(page.text for page in pages if page.text).strip()
        document = ParsedDocument(
            document_id=uuid4().hex,
            name=filename,
            document_type="pdf",
            pages=pages,
            cleaned_text=cleaned_text,
            token_count=self.token_counter.count(cleaned_text),
            chunks=[],
            warnings=warnings,
        )
        document.chunks = self.context_builder.build_document_chunks(document)
        return document

    def _parse_image(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DocumentParsingError(
                "Pillow is not installed. Install pillow to process image documents.",
            ) from exc

        try:
            with Image.open(io.BytesIO(payload)) as image:
                frame_count = getattr(image, "n_frames", 1)
        except UnidentifiedImageError as exc:
            raise DocumentParsingError(f"Image '{filename}' is corrupt or unsupported.") from exc
        except Exception as exc:
            raise DocumentParsingError(f"Failed to open image '{filename}'.") from exc

        if frame_count > self.settings.app.max_pages_per_file:
            raise PageLimitError(
                f"Image '{filename}' exceeds the {self.settings.app.max_pages_per_file} frame limit.",
                details={"filename": filename, "page_count": frame_count},
            )

        warnings: list[str] = []
        pages: list[ParsedPage] = []

        try:
            with Image.open(io.BytesIO(payload)) as image:
                for index in range(frame_count):
                    if frame_count > 1:
                        image.seek(index)
                    frame = image.convert("RGB")
                    raw_text = self._extract_text_from_image(frame)
                    cleaned = clean_extracted_text(raw_text)
                    if not cleaned:
                        warnings.append(f"OCR produced no text on frame {index + 1}.")
                    pages.append(
                        ParsedPage(
                            page_number=index + 1,
                            text=cleaned,
                            token_count=self.token_counter.count(cleaned),
                        )
                    )
        except OCRExtractionError:
            raise
        except Exception as exc:
            raise DocumentParsingError(f"Failed to process image '{filename}'.") from exc

        if not any(page.text for page in pages):
            raise OCRExtractionError(f"No text could be extracted from image '{filename}'.")

        cleaned_text = "\n\n".join(page.text for page in pages if page.text).strip()
        document = ParsedDocument(
            document_id=uuid4().hex,
            name=filename,
            document_type="image",
            pages=pages,
            cleaned_text=cleaned_text,
            token_count=self.token_counter.count(cleaned_text),
            chunks=[],
            warnings=warnings,
        )
        document.chunks = self.context_builder.build_document_chunks(document)
        return document

    def _extract_text_from_image(self, image: Any) -> str:
        if self._ocr_engine is not None:
            try:
                return self._ocr_engine(image, self.settings.ocr.language)
            except Exception as exc:
                raise OCRExtractionError("The configured OCR engine failed while extracting text.") from exc

        provider = self.settings.ocr.provider
        if provider == "gemini":
            return self._extract_text_via_gemini(image)

        try:
            text = self._extract_text_via_tesseract(image)
        except OCRExtractionError as tesseract_error:
            if provider != "auto" or not self._has_gemini_key():
                raise
            self.logger.info(
                "Tesseract OCR failed; attempting Gemini OCR fallback.",
                extra={"event": "ocr_gemini_fallback"},
            )
            try:
                return self._extract_text_via_gemini(image)
            except OCRExtractionError as gemini_error:
                raise OCRExtractionError(
                    "OCR failed with Tesseract and Gemini. Install Tesseract, set GEMINI_API_KEY, "
                    "or configure ocr.provider explicitly.",
                    details={
                        "tesseract_error": tesseract_error.message,
                        "gemini_error": gemini_error.message,
                    },
                ) from gemini_error

        if clean_extracted_text(text):
            return text

        if provider == "auto" and self._has_gemini_key():
            self.logger.info(
                "Tesseract OCR returned no text; attempting Gemini OCR fallback.",
                extra={"event": "ocr_gemini_empty_fallback"},
            )
            return self._extract_text_via_gemini(image)

        return text

    def _extract_text_via_tesseract(self, image: Any) -> str:
        try:
            import pytesseract
            from pytesseract import TesseractNotFoundError
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OCRExtractionError(
                "OCR dependencies are not installed. Install pytesseract to process image documents.",
            ) from exc

        try:
            if self.settings.ocr.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.settings.ocr.tesseract_cmd
            prepared_image = self._prepare_image_for_ocr(image)
            text = pytesseract.image_to_string(
                prepared_image,
                lang=self.settings.ocr.language,
                config="--psm 6",
            )
            if clean_extracted_text(text):
                return text
            return pytesseract.image_to_string(
                prepared_image,
                lang=self.settings.ocr.language,
                config="--psm 11",
            )
        except TesseractNotFoundError as exc:
            raise OCRExtractionError("Tesseract OCR is not installed or not available on PATH.") from exc
        except Exception as exc:
            raise OCRExtractionError("Tesseract failed while extracting text from an image.") from exc

    def _extract_text_via_gemini(self, image: Any) -> str:
        api_key = os.getenv(self.settings.gemini.api_key_env) or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise OCRExtractionError(
                f"Gemini OCR requires an API key in '{self.settings.gemini.api_key_env}' or 'GOOGLE_API_KEY'."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OCRExtractionError("Gemini OCR requires the google-genai package.") from exc

        try:
            image_bytes = self._image_to_png_bytes(image)
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.settings.ocr.gemini_model_name or self.settings.gemini.model_name,
                contents=[
                    "Extract every visible text string from this document image. "
                    "Return only the extracted text in natural reading order.",
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_output_tokens=self.settings.ocr.gemini_max_output_tokens,
                ),
            )
            return response.text or ""
        except Exception as exc:
            raise OCRExtractionError("Gemini failed while extracting text from an image.") from exc

    @staticmethod
    def _image_to_png_bytes(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _prepare_image_for_ocr(image: Any) -> Any:
        from PIL import ImageOps

        prepared = image.convert("RGB")
        width, height = prepared.size
        if width < 1200:
            scale = max(2, min(4, 1200 // max(width, 1)))
            prepared = prepared.resize((width * scale, height * scale))
        grayscale = ImageOps.grayscale(prepared)
        return ImageOps.autocontrast(grayscale)

    def _has_gemini_key(self) -> bool:
        return bool(os.getenv(self.settings.gemini.api_key_env) or os.getenv("GOOGLE_API_KEY"))
