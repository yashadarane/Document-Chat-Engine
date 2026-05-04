from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


ProviderName = Literal["local", "gemini", "groq"]
LocalBackendName = Literal["llama_cpp", "transformers"]
OCRProviderName = Literal["auto", "tesseract", "gemini"]


class AppSettings(BaseModel):
    """Top-level application settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "Document Chat Engine"
    version: str = "2.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_files: PositiveInt = 3
    max_pages_per_file: PositiveInt = 5
    max_file_size_mb: PositiveInt = 10
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    )

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, value: list[str]) -> list[str]:
        return [extension.lower().strip() for extension in value]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


class ContextSettings(BaseModel):
    """Context-window and chunking behavior."""

    model_config = ConfigDict(extra="forbid")

    chunk_size_tokens: PositiveInt = 360
    chunk_overlap_tokens: PositiveInt = 60
    max_prompt_context_tokens: PositiveInt = 2600
    max_chunks_for_answer: PositiveInt = 5
    memory_turns: PositiveInt = 4
    memory_token_budget: PositiveInt = 120
    extractive_summary_tokens: PositiveInt = 220


class ProviderRoutingSettings(BaseModel):
    """Provider routing and fallback settings."""

    model_config = ConfigDict(extra="forbid")

    default_provider: ProviderName = "groq"
    allow_ui_provider_override: bool = True
    enable_fallback: bool = False
    fallback_provider: ProviderName = "local"


class LocalModelOption(BaseModel):
    """Named local model profile exposed in the UI."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    backend: LocalBackendName | None = None
    model_name: str | None = None
    model_path: str | None = None
    device: str | None = None
    max_context_tokens: PositiveInt | None = None
    max_new_tokens: PositiveInt | None = None
    load_timeout_seconds: PositiveInt | None = None
    generation_timeout_seconds: PositiveInt | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)


class LocalLLMSettings(BaseModel):
    """Local model settings."""

    model_config = ConfigDict(extra="forbid")

    active_model: str | None = None
    display_name: str | None = None
    backend: LocalBackendName = "llama_cpp"
    model_name: str | None = None
    model_path: str | None = "./app/models/phi-3-mini-4k-instruct-q4_k_m.gguf"
    device: str = "cpu"
    max_context_tokens: PositiveInt = 2048
    max_new_tokens: PositiveInt = 300
    load_timeout_seconds: PositiveInt = 240
    generation_timeout_seconds: PositiveInt = 90
    temperature: float = Field(default=0.05, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    model_options: list[LocalModelOption] = Field(default_factory=list)

    @property
    def model_label(self) -> str:
        if self.display_name:
            return self.display_name
        if self.model_name:
            return self.model_name
        if self.model_path:
            return Path(self.model_path).stem
        return "local-model"

    def with_active_model(self, model_id: str | None = None) -> "LocalLLMSettings":
        """Return settings with the selected named profile applied."""

        selected_id = model_id or self.active_model
        if not selected_id:
            return self

        selected = next((option for option in self.model_options if option.id == selected_id), None)
        if selected is None:
            return self

        data = self.model_dump(exclude={"model_options"})
        data["active_model"] = selected.id
        for key, value in selected.model_dump(exclude={"id"}).items():
            if value is not None:
                data[key] = value

        data["model_options"] = self.model_options
        return LocalLLMSettings.model_validate(data)


class GeminiSettings(BaseModel):
    """Gemini provider settings."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    max_context_tokens: PositiveInt = 32768
    max_new_tokens: PositiveInt = 300
    generation_timeout_seconds: PositiveInt = 60
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class GroqSettings(BaseModel):
    """Groq provider settings."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "llama-3.3-70b-versatile"
    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    max_context_tokens: PositiveInt = 32768
    max_new_tokens: PositiveInt = 300
    generation_timeout_seconds: PositiveInt = 60
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class OCRSettings(BaseModel):
    """OCR settings."""

    model_config = ConfigDict(extra="forbid")

    provider: OCRProviderName = "auto"
    language: str = "eng"
    tesseract_cmd: str | None = None
    gemini_model_name: str | None = None
    gemini_max_output_tokens: PositiveInt = 2048


class Settings(BaseModel):
    """Validated application settings."""

    model_config = ConfigDict(extra="forbid")

    app: AppSettings = Field(default_factory=AppSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    routing: ProviderRoutingSettings = Field(default_factory=ProviderRoutingSettings)
    local_llm: LocalLLMSettings = Field(default_factory=LocalLLMSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)

    def with_local_model(self, model_id: str | None) -> "Settings":
        """Return a settings copy with the requested local model profile selected."""

        if not model_id:
            return self.model_copy(update={"local_llm": self.local_llm.with_active_model()})
        return self.model_copy(update={"local_llm": self.local_llm.with_active_model(model_id)})


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a top-level mapping.")

    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load configuration from YAML."""

    resolved_path = Path(config_path or os.getenv("CHAT_ENGINE_CONFIG", DEFAULT_CONFIG_PATH))
    payload = _deep_merge({}, _read_yaml_file(resolved_path))
    return Settings.model_validate(payload)
