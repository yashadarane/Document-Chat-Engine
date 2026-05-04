# Document Chat Engine

This project implements the supplied PDF assignment as a lightweight **non-RAG** document chat engine. It reads uploaded PDFs or images, extracts text, chunks and ranks the extracted content in memory, builds a grounded prompt, and answers strictly from the uploaded documents.

The app supports three answer providers behind the same document pipeline:

- `groq`: Groq through `GROQ_API_KEY`, used as the default provider
- `gemini`: Google AI Studio through `GEMINI_API_KEY`
- `local`: optional laptop-hosted `llama.cpp` model using Qwen2.5 1.5B Instruct

## What It Does

1. Upload up to 3 PDFs or images.
2. Parse PDFs with `pdfplumber` and `pypdf`.
3. OCR images with `pytesseract`.
4. Clean and manually chunk the extracted text.
5. Rank chunks for the current question.
6. Build a strict document-grounded prompt.
7. Answer with local Qwen, Gemini, or Groq.

This stays within the assignment's non-RAG requirement because it does **not** use embeddings, a vector database, or retrieval infrastructure. It only performs in-memory lexical ranking over uploaded chunks.

## Model Decision

Groq is the default answer provider. The local model path remains available as an optional **Qwen2.5 1.5B Instruct GGUF** setup, configured directly instead of exposed through a local-model dropdown.

Models considered during development:

- **Qwen2.5 1.5B Instruct GGUF**: retained as the optional local model because it gives the best balance of CPU usability, instruction following, and document Q&A quality among the available small local models.
- **Phi-3 Mini Instruct GGUF**: evaluated but removed from the UI/config because it was heavier and less consistent in this local document-Q&A setup.
- **TinyLlama 1.1B Chat GGUF**: evaluated but removed from the UI/config because answer quality was too weak for grounded document Q&A.
- **Gemini 2.5 Flash**: retained as an optional cloud provider and fallback target.
- **Groq llama-3.3-70b-versatile**: selected as the default cloud provider through Groq's OpenAI-compatible chat-completions API.

The model files may still exist under `app/models`, but only Qwen is exposed by the active config.

## Configuration

Main config sections:

- `routing`: provider selection and fallback behavior
- `local_llm`: local Qwen configuration
- `gemini`: Google AI Studio settings
- `groq`: Groq settings
- `ocr`: OCR behavior

Optional local model:

```yaml
local_llm:
  active_model: "qwen25_15b_gguf"
  display_name: "Qwen2.5 1.5B Instruct"
  model_path: "./app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

## API Keys

Gemini:

```powershell
$env:GEMINI_API_KEY="your-key"
```

Groq:

```powershell
$env:GROQ_API_KEY="your-key"
```

The Groq integration uses Groq's OpenAI-compatible chat completions endpoint:

```text
https://api.groq.com/openai/v1/chat/completions
```

The configured Groq model is `llama-3.3-70b-versatile`, chosen as a fast hosted option for direct document Q&A.

Official Groq docs describe the API key environment variable as `GROQ_API_KEY`, bearer authentication, and the `/openai/v1/chat/completions` endpoint.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR on your machine.
4. For local inference, install `llama-cpp-python`:

```bash
pip install -r requirements-llama-cpp.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

5. Place the Qwen GGUF model at the configured path or update `config.yaml`.

## Run

```bash
streamlit run streamlit_app.py
```

The app usually opens at:

```text
http://localhost:8501
```

## UI Features

- Provider selector: defaults to `groq`; choose `local` or `gemini` when needed
- Fallback checkbox: retry with the configured fallback provider if the primary provider fails
- Warm selected provider button
- Extracted text preview
- Per-answer context viewer for debugging

## Testing

Run:

```bash
pytest
```

The tests cover:

- PDF parsing
- OCR extraction
- Prompt guardrails
- Chunk ranking/context preparation
- Local/Gemini provider routing
- Fallback behavior

## Design Notes

- The prompt is strict and anti-hallucination oriented.
- The same parsing, chunking, memory, and prompt pipeline is shared across all providers.
- Groq is the default provider for faster hosted responses.
- Qwen remains available as an optional local model for offline or fallback testing.
