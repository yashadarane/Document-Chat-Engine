from __future__ import annotations

import logging
import os
import re
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import GeminiSettings, GroqSettings, LocalLLMSettings, ProviderName, Settings
from app.core.exceptions import GenerationTimeoutError, ModelUnavailableError, ProviderConfigurationError
from app.models.domain import GenerationResult
from app.services.prompt_engine import PromptEngine
from app.utils.tokens import RegexTokenCounter, TokenCounter, truncate_text_to_budget


class BaseTextProvider(ABC):
    """Shared interface for provider-specific text generation backends."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.token_counter = token_counter or RegexTokenCounter()
        self._last_error: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        """Generate a completion for the provided prompt."""

    def estimate_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def count(self, text: str) -> int:
        return self.estimate_tokens(text)

    def is_ready(self) -> bool:
        return False

    def load(self) -> None:
        """Load or validate the provider if needed."""

    @property
    def last_error(self) -> str | None:
        return self._last_error


class TransformersBackend(BaseTextProvider):
    """Transformers backend for Hugging Face local models."""

    def __init__(self, settings: LocalLLMSettings, token_counter: TokenCounter | None = None) -> None:
        super().__init__(token_counter=token_counter)
        self.settings = settings
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = settings.device
        self._load_lock = Lock()

    @property
    def name(self) -> str:
        return f"local:transformers:{self.settings.model_label}"

    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def estimate_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return super().estimate_tokens(text)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        self.load()
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with self._torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_length = encoded["input_ids"].shape[1]
        generated = output[0][input_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def load(self) -> None:
        if self.is_ready():
            return

        with self._load_lock:
            if self.is_ready():
                return

            if not self.settings.model_name:
                raise ModelUnavailableError("No Hugging Face model name was configured for the transformers backend.")

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "Transformers backend requires torch and transformers to be installed.",
                ) from exc

            try:
                self._torch = torch
                self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name, use_fast=True)
                if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype="auto",
                )
                if self._device.startswith("cuda") and torch.cuda.is_available():
                    self._model.to(self._device)
                else:
                    self._device = "cpu"
                    self._model.to(self._device)
                self._model.eval()
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                raise ModelUnavailableError(
                    f"Failed to load transformers model '{self.settings.model_name}'.",
                    details={"backend": self.name},
                ) from exc


class LlamaCppBackend(BaseTextProvider):
    """llama.cpp backend for GGUF models."""

    def __init__(self, settings: LocalLLMSettings, token_counter: TokenCounter | None = None) -> None:
        super().__init__(token_counter=token_counter)
        self.settings = settings
        self._client = None

    @property
    def name(self) -> str:
        return f"local:llama_cpp:{self.settings.model_label}"

    def is_ready(self) -> bool:
        return self._client is not None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        self.load()
        completion = self._client.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stop=[
                "\n\nUSER QUESTION:",
                "\n\nANSWER:",
                "\nQUESTION:",
                "\nANSWER FORMAT:",
                "\nSHORT CONVERSATION MEMORY:",
                "\nDOCUMENT CONTEXT:",
            ],
        )
        return completion["choices"][0]["text"].strip()

    def load(self) -> None:
        if self.is_ready():
            return

        if not self.settings.model_path:
            raise ModelUnavailableError("No GGUF model path was configured for the llama.cpp backend.")

        model_path = Path(self.settings.model_path)
        if not model_path.exists():
            raise ModelUnavailableError(
                f"Configured GGUF model path '{model_path}' does not exist.",
                details={"backend": self.name},
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                "llama.cpp backend requires llama-cpp-python to be installed.",
            ) from exc

        try:
            cpu_count = os.cpu_count() or 2
            self._client = Llama(
                model_path=str(model_path),
                n_ctx=self.settings.max_context_tokens,
                n_threads=max(2, cpu_count - 1),
                n_batch=128,
                verbose=False,
            )
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            raise ModelUnavailableError(
                f"Failed to load llama.cpp model from '{model_path}'.",
                details={"backend": self.name},
            ) from exc


class GeminiBackend(BaseTextProvider):
    """Gemini backend using a Google AI Studio API key."""

    def __init__(self, settings: GeminiSettings, token_counter: TokenCounter | None = None) -> None:
        super().__init__(token_counter=token_counter)
        self.settings = settings
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    def is_ready(self) -> bool:
        return self._client is not None

    def load(self) -> None:
        if self.is_ready():
            return

        api_key = os.getenv(self.settings.api_key_env) or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                f"Gemini API key was not found in environment variable '{self.settings.api_key_env}'.",
                details={"env_var": self.settings.api_key_env},
            )

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError("Gemini backend requires the google-genai package.") from exc

        try:
            self._client = genai.Client(api_key=api_key)
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            raise ModelUnavailableError("Failed to initialize the Gemini client.") from exc

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        del repetition_penalty
        self.load()

        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError("Gemini backend requires the google-genai package.") from exc

        response = self._client.models.generate_content(
            model=self.settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_new_tokens,
            ),
        )
        return (response.text or "").strip()


class GroqBackend(BaseTextProvider):
    """Groq backend using the OpenAI-compatible chat completions API."""

    def __init__(self, settings: GroqSettings, token_counter: TokenCounter | None = None) -> None:
        super().__init__(token_counter=token_counter)
        self.settings = settings
        self._api_key: str | None = None

    @property
    def name(self) -> str:
        return f"groq:{self.settings.model_name}"

    def is_ready(self) -> bool:
        return bool(self._api_key)

    def load(self) -> None:
        api_key = os.getenv(self.settings.api_key_env)
        if api_key:
            api_key = api_key.strip().strip('"').strip("'")
        if not api_key:
            raise ProviderConfigurationError(
                f"Groq API key was not found in environment variable '{self.settings.api_key_env}'.",
                details={"env_var": self.settings.api_key_env},
            )
        self._api_key = api_key
        self._last_error = None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        del repetition_penalty
        self.load()
        payload = {
            "model": self.settings.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        request = Request(
            self.settings.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "DocChatEngine/2.1.0 Python urllib",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.settings.generation_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self._last_error = body or str(exc)
            detail = self._format_api_error(exc.code, body)
            raise ModelUnavailableError(
                f"Groq API returned an error while generating a response: {detail}",
                details={"status": exc.code, "provider": self.name},
            ) from exc
        except (URLError, TimeoutError) as exc:
            self._last_error = str(exc)
            raise ModelUnavailableError(
                "Groq API could not be reached while generating a response.",
                details={"provider": self.name},
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()

    @staticmethod
    def _format_api_error(status_code: int, body: str) -> str:
        body = (body or "").strip()
        if not body:
            return f"HTTP {status_code}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return f"HTTP {status_code}: {body[:500]}"

        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or str(error)
            error_type = error.get("type")
            if error_type:
                return f"HTTP {status_code} {error_type}: {message}"
            return f"HTTP {status_code}: {message}"
        return f"HTTP {status_code}: {body[:500]}"


class ProviderLLMService:
    """Provider-agnostic generation service with optional fallback routing."""

    def __init__(
        self,
        settings: Settings,
        prompt_engine: PromptEngine,
        token_counter: TokenCounter,
        *,
        local_backend: BaseTextProvider | None = None,
        gemini_backend: BaseTextProvider | None = None,
        groq_backend: BaseTextProvider | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_engine = prompt_engine
        self.token_counter = token_counter
        self.logger = logging.getLogger(self.__class__.__name__)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-chat-provider")
        self._provider_overrides: dict[ProviderName, BaseTextProvider | None] = {
            "local": local_backend,
            "gemini": gemini_backend,
            "groq": groq_backend,
        }
        self.providers: dict[ProviderName, BaseTextProvider] = {}

    def generate_response(
        self,
        context: str,
        query: str,
        history: str,
        *,
        provider_name: ProviderName,
        fallback_enabled: bool,
        fallback_provider_name: ProviderName,
    ) -> GenerationResult:
        attempted: list[str] = []
        primary = provider_name
        fallback = fallback_provider_name if fallback_enabled and fallback_provider_name != provider_name else None

        try:
            result = self._generate_with_provider(primary, context, query, history)
            return result
        except (GenerationTimeoutError, ModelUnavailableError, ProviderConfigurationError) as exc:
            attempted.append(primary)
            if not fallback:
                raise
            self.logger.warning(
                "Primary provider failed; attempting fallback.",
                extra={"event": "provider_fallback", "provider": primary, "fallback_provider": fallback},
            )
            fallback_result = self._generate_with_provider(fallback, context, query, history)
            fallback_result.fallback_used = True
            fallback_result.warnings.append(f"Primary provider '{primary}' failed: {exc.message}")
            return fallback_result

    def prepare_provider(self, provider_name: ProviderName) -> None:
        provider = self.get_provider(provider_name)
        self._prepare_backend(provider, provider_name)

    def get_provider(self, provider_name: ProviderName) -> BaseTextProvider:
        if provider_name not in self.providers:
            self.providers[provider_name] = self._build_provider(provider_name)
        return self.providers[provider_name]

    def _generate_with_provider(
        self,
        provider_name: ProviderName,
        context: str,
        query: str,
        history: str,
    ) -> GenerationResult:
        provider = self.get_provider(provider_name)
        provider_settings = self._provider_settings(provider_name)
        fitted_context, fitted_history, warnings = self._fit_prompt_inputs(
            provider,
            provider_settings.max_context_tokens,
            context,
            query,
            history,
            provider_name=provider_name,
        )
        prompt = self._build_provider_prompt(provider_name, fitted_context, query, fitted_history)

        try:
            if not provider.is_ready():
                self._prepare_backend(provider, provider_name)
            future = self._executor.submit(
                provider.generate,
                prompt,
                max_new_tokens=provider_settings.max_new_tokens,
                temperature=provider_settings.temperature,
                top_p=provider_settings.top_p,
                repetition_penalty=getattr(provider_settings, "repetition_penalty", 1.0),
            )
            raw_answer = future.result(timeout=provider_settings.generation_timeout_seconds)
        except FutureTimeoutError as exc:
            self.logger.warning("Provider generation timed out.", extra={"event": "generation_timeout", "provider": provider_name})
            raise GenerationTimeoutError(f"{provider_name.capitalize()} response timed out.") from exc
        except (ModelUnavailableError, ProviderConfigurationError):
            raise
        except Exception as exc:
            provider._last_error = str(exc)
            raise ModelUnavailableError(
                f"Provider '{provider_name}' failed while generating a response.",
                details={"provider": provider_name},
            ) from exc

        answer = self._sanitize_answer(raw_answer)
        if not answer:
            return GenerationResult(
                answer=self.prompt_engine.REFUSAL_TEXT,
                grounded=False,
                fallback_used=False,
                backend=provider.name,
                warnings=[*warnings, "Provider returned an empty response."],
            )

        return GenerationResult(
            answer=answer,
            grounded=not self.prompt_engine.is_refusal(answer),
            fallback_used=False,
            backend=provider.name,
            warnings=warnings,
        )

    def _build_local_backend(self) -> BaseTextProvider:
        local_settings = self.settings.local_llm.with_active_model()
        self.settings = self.settings.model_copy(update={"local_llm": local_settings})
        if local_settings.backend == "transformers":
            return TransformersBackend(local_settings, token_counter=self.token_counter)
        if local_settings.backend == "llama_cpp":
            return LlamaCppBackend(local_settings, token_counter=self.token_counter)
        raise ModelUnavailableError(f"Unsupported local backend '{local_settings.backend}'.")

    def _build_provider(self, provider_name: ProviderName) -> BaseTextProvider:
        override = self._provider_overrides[provider_name]
        if override is not None:
            return override
        if provider_name == "local":
            return self._build_local_backend()
        if provider_name == "gemini":
            return GeminiBackend(self.settings.gemini, token_counter=self.token_counter)
        return GroqBackend(self.settings.groq, token_counter=self.token_counter)

    def _provider_settings(self, provider_name: ProviderName):
        if provider_name == "local":
            return self.settings.local_llm
        if provider_name == "gemini":
            return self.settings.gemini
        return self.settings.groq

    def _fit_prompt_inputs(
        self,
        provider: BaseTextProvider,
        max_context_tokens: int,
        context: str,
        query: str,
        history: str,
        *,
        provider_name: ProviderName,
    ) -> tuple[str, str, list[str]]:
        warnings: list[str] = []
        prompt_budget = max(256, max_context_tokens - 192)
        prompt = self._build_provider_prompt(provider_name, context, query, history)
        if provider.estimate_tokens(prompt) <= prompt_budget:
            return context, history, warnings

        trimmed_history = "No prior grounded conversation."
        prompt = self._build_provider_prompt(provider_name, context, query, trimmed_history)
        if provider.estimate_tokens(prompt) <= prompt_budget:
            warnings.append("Conversation history was trimmed to fit the provider token budget.")
            return context, trimmed_history, warnings

        prompt_without_context = self._build_provider_prompt(provider_name, "", query, trimmed_history)
        remaining_budget = max(64, prompt_budget - provider.estimate_tokens(prompt_without_context))
        trimmed_context = truncate_text_to_budget(context, remaining_budget, provider)
        warnings.append("Document context was trimmed further at generation time to fit the provider token budget.")
        return trimmed_context, trimmed_history, warnings

    def _build_provider_prompt(self, provider_name: ProviderName, context: str, query: str, history: str) -> str:
        if provider_name == "local":
            return self.prompt_engine.build_qwen_prompt(context=context, query=query, history=history)
        return self.prompt_engine.build_prompt(context=context, query=query, history=history)

    def _prepare_backend(self, provider: BaseTextProvider, provider_name: ProviderName) -> None:
        timeout = self._provider_settings(provider_name).load_timeout_seconds if provider_name == "local" else 30
        future = self._executor.submit(provider.load)
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            self.logger.warning("Provider loading timed out.", extra={"event": "provider_load_timeout", "provider": provider_name})
            raise GenerationTimeoutError(f"{provider_name.capitalize()} model loading timed out.") from exc
        except (ModelUnavailableError, ProviderConfigurationError):
            raise
        except Exception as exc:
            provider._last_error = str(exc)
            raise ModelUnavailableError(
                f"Provider '{provider_name}' failed while preparing.",
                details={"provider": provider_name},
            ) from exc

    @staticmethod
    def _sanitize_answer(answer: str) -> str:
        cleaned = answer.strip()
        if not cleaned:
            return ""
        if "ANSWER:" in cleaned:
            cleaned = cleaned.split("ANSWER:", maxsplit=1)[-1].strip()
        leakage_markers = [
            "\nUSER QUESTION:",
            "\nQUESTION:",
            "\nANSWER FORMAT:",
            "\nSHORT CONVERSATION MEMORY:",
            "\nDOCUMENT CONTEXT:",
            "\nRules:",
        ]
        for marker in leakage_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker, maxsplit=1)[0].strip()

        answer_match = re.search(r"Answer:\s*(.+?)(?:\nEvidence:|\Z)", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if answer_match:
            cleaned = answer_match.group(1).strip()

        cleaned = re.sub(r"^(Answer:\s*)+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\n?Evidence:\s*.+\Z", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned
