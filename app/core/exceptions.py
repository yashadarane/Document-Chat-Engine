from __future__ import annotations

from typing import Any


class ChatEngineError(Exception):
    """Base exception for application-specific errors."""

    status_code = 400
    error_code = "chat_engine_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidUploadError(ChatEngineError):
    """Raised for upload validation failures."""

    status_code = 400
    error_code = "invalid_upload"


class FileCountLimitError(InvalidUploadError):
    """Raised when too many files are uploaded."""

    error_code = "file_count_limit_exceeded"


class EmptyFileError(InvalidUploadError):
    """Raised when an uploaded file has no content."""

    error_code = "empty_file"


class FileTooLargeError(InvalidUploadError):
    """Raised when an uploaded file exceeds size limits."""

    error_code = "file_too_large"


class UnsupportedFileTypeError(InvalidUploadError):
    """Raised when an uploaded file type is not supported."""

    status_code = 415
    error_code = "unsupported_file_type"


class PageLimitError(InvalidUploadError):
    """Raised when a document exceeds the configured page limit."""

    error_code = "page_limit_exceeded"


class DocumentParsingError(ChatEngineError):
    """Raised when document parsing fails."""

    status_code = 422
    error_code = "document_parsing_failed"


class OCRExtractionError(ChatEngineError):
    """Raised when OCR extraction fails."""

    status_code = 422
    error_code = "ocr_extraction_failed"


class SessionNotFoundError(ChatEngineError):
    """Raised when a chat session cannot be found."""

    status_code = 404
    error_code = "session_not_found"


class ModelUnavailableError(ChatEngineError):
    """Raised when the configured model backend cannot be used."""

    status_code = 503
    error_code = "model_unavailable"


class ProviderConfigurationError(ChatEngineError):
    """Raised when a hosted or local provider is misconfigured."""

    status_code = 500
    error_code = "provider_configuration_error"


class GenerationTimeoutError(ChatEngineError):
    """Raised when generation times out."""

    status_code = 504
    error_code = "generation_timeout"
