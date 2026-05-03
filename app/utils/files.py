from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)


PDF_EXTENSION = ".pdf"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(slots=True)
class ValidatedFile:
    """Validated file metadata."""

    filename: str
    content_type: str | None
    extension: str
    size_bytes: int


def normalize_filename(filename: str | None) -> str:
    """Return a safe display filename."""

    if not filename:
        return "upload"
    return Path(filename).name


def get_extension(filename: str) -> str:
    """Return the normalized extension for a filename."""

    return Path(filename).suffix.lower()


def is_pdf_extension(extension: str) -> bool:
    """Check whether a file is a PDF."""

    return extension == PDF_EXTENSION


def is_image_extension(extension: str) -> bool:
    """Check whether a file is a supported image type."""

    return extension in SUPPORTED_IMAGE_EXTENSIONS


def validate_file_metadata(
    filename: str | None,
    content_type: str | None,
    payload: bytes,
    *,
    max_size_bytes: int,
    allowed_extensions: list[str],
) -> ValidatedFile:
    """Validate basic file constraints before parsing."""

    safe_name = normalize_filename(filename)
    extension = get_extension(safe_name)

    if not payload:
        raise EmptyFileError(f"Uploaded file '{safe_name}' is empty.")

    if len(payload) > max_size_bytes:
        raise FileTooLargeError(
            f"Uploaded file '{safe_name}' exceeds the {max_size_bytes} byte limit.",
            details={"filename": safe_name, "size_bytes": len(payload)},
        )

    if extension not in allowed_extensions:
        raise UnsupportedFileTypeError(
            f"Uploaded file '{safe_name}' uses unsupported extension '{extension or '<none>'}'.",
            details={"filename": safe_name, "extension": extension or None},
        )

    return ValidatedFile(
        filename=safe_name,
        content_type=content_type,
        extension=extension,
        size_bytes=len(payload),
    )

