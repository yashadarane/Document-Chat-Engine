from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import streamlit as st

from app.core.config import ProviderName, Settings, load_settings
from app.core.exceptions import ChatEngineError
from app.main import create_chat_engine
from app.services.document_ingestion import IncomingFile


def get_settings(local_model_id: str | None = None) -> Settings:
    return load_settings().with_local_model(local_model_id)


@st.cache_resource(show_spinner=False)
def get_chat_engine(local_model_id: str | None = None):
    return create_chat_engine(get_settings(local_model_id))


def sync_session_state(settings: Settings) -> None:
    st.session_state.setdefault("active_session_id", None)
    st.session_state.setdefault("uploaded_preview", "")
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("selected_provider", settings.routing.default_provider)
    st.session_state.setdefault("fallback_enabled", settings.routing.enable_fallback)


def reset_chat_state() -> None:
    st.session_state["active_session_id"] = None
    st.session_state["uploaded_preview"] = ""
    st.session_state["documents"] = []
    st.session_state["chat_messages"] = []


def uploaded_to_incoming(files: Iterable) -> list[IncomingFile]:
    return [IncomingFile(filename=f.name, content_type=f.type, payload=f.getvalue()) for f in files]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #101418;
            --panel: #171d23;
            --panel-2: #1d242c;
            --border: #2d3742;
            --text: #edf2f7;
            --muted: #9aa8b6;
            --accent: #2dd4bf;
            --accent-2: #60a5fa;
            --danger: #f87171;
            --warning: #fbbf24;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(45, 212, 191, 0.09), transparent 30rem),
                linear-gradient(135deg, #101418 0%, #14191f 48%, #101418 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 1240px;
            padding: 1.4rem 2rem 5rem;
        }
        [data-testid="stSidebar"] {
            background: #0d1116 !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1.4rem 1.1rem;
        }
        [data-testid="stSidebar"] *, .stApp * {
            letter-spacing: 0 !important;
        }
        h1, h2, h3, p, label, span, div {
            color: var(--text);
        }
        h1 {
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.15rem !important;
        }
        h2, h3 {
            font-weight: 650 !important;
        }
        [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
            font-size: 0.86rem;
        }
        .section-title {
            color: var(--text);
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0 0 0.65rem;
            text-transform: uppercase;
        }
        .surface {
            background: rgba(23, 29, 35, 0.92);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }
        .doc-row {
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        .doc-name {
            color: var(--text);
            font-weight: 650;
            font-size: 0.93rem;
            margin-bottom: 0.35rem;
        }
        .doc-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            color: var(--muted);
            font-size: 0.78rem;
        }
        .doc-meta b {
            color: #dce6ef;
            font-weight: 600;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            background: #121820;
            color: var(--muted);
            font-size: 0.76rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(23, 29, 35, 0.92);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.78rem 0.9rem;
        }
        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-size: 0.76rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-size: 1.05rem !important;
        }
        [data-testid="stFileUploader"] {
            background: #121820;
            border: 1px dashed #46515d;
            border-radius: 8px;
            padding: 0.55rem;
        }
        [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small {
            color: var(--muted) !important;
        }
        [data-baseweb="select"] > div,
        [data-testid="stChatInput"] {
            background: #121820 !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        [data-baseweb="select"] span, [data-baseweb="select"] div {
            color: var(--text) !important;
        }
        .stButton > button {
            background: #202a33 !important;
            border: 1px solid #3a4652 !important;
            border-radius: 8px !important;
            color: var(--text) !important;
            font-weight: 650 !important;
            min-height: 2.35rem;
        }
        .stButton > button:hover {
            border-color: var(--accent) !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] {
            background: #0f766e !important;
            border-color: #14b8a6 !important;
            color: #ffffff !important;
        }
        .stChatMessage {
            background: rgba(23, 29, 35, 0.96) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 0.85rem 1rem !important;
            margin-bottom: 0.65rem;
        }
        .stChatMessage p {
            color: #e8eef5 !important;
            line-height: 1.58;
            font-size: 0.94rem;
        }
        [data-testid="stChatInput"] textarea {
            color: var(--text) !important;
        }
        .stTextArea textarea {
            background: #0f141a !important;
            border: 1px solid var(--border) !important;
            color: #dce6ef !important;
            border-radius: 8px !important;
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace !important;
            font-size: 0.82rem !important;
        }
        [data-testid="stExpander"] {
            background: rgba(18, 24, 32, 0.8) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary p {
            color: #c6d3df !important;
        }
        .stAlert {
            background: #151c24 !important;
            border-color: var(--border) !important;
            border-radius: 8px !important;
        }
        .stAlert p {
            color: #e7edf4 !important;
        }
        code {
            background: #0f141a !important;
            color: var(--accent) !important;
            border-radius: 4px;
            padding: 0.08rem 0.28rem;
        }
        hr {
            border-color: var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(settings: Settings) -> tuple[ProviderName, bool, bool]:
    with st.sidebar:
        st.markdown("### DocChat Engine")
        st.caption("Grounded document Q&A")

        provider = st.selectbox(
            "Provider",
            options=["local", "gemini", "groq"],
            index=["local", "gemini", "groq"].index(st.session_state["selected_provider"])
            if st.session_state["selected_provider"] in ["local", "gemini", "groq"]
            else 0,
            disabled=not settings.routing.allow_ui_provider_override,
        )
        st.session_state["selected_provider"] = provider

        fallback_enabled = st.checkbox(
            "Fallback to secondary provider",
            value=st.session_state["fallback_enabled"],
            help=f"If the primary provider fails, retry with {settings.routing.fallback_provider}.",
        )
        st.session_state["fallback_enabled"] = fallback_enabled

        effective_settings = settings.with_local_model(None)
        local_path = effective_settings.local_llm.model_path
        model_status = "configured"
        if effective_settings.local_llm.backend == "llama_cpp" and local_path and not Path(local_path).exists():
            model_status = "model file missing"
        gemini_status = "key found" if os.getenv(settings.gemini.api_key_env) else "key missing"
        groq_status = "key found" if os.getenv(settings.groq.api_key_env) else "key missing"

        st.divider()
        st.markdown(
            f"""
            <div class="surface">
                <div class="section-title">Runtime</div>
                <div class="doc-meta"><span>Backend <b>{effective_settings.local_llm.backend}</b></span></div>
                <div class="doc-meta"><span>Model <b>{effective_settings.local_llm.model_label}</b></span></div>
                <div class="doc-meta"><span>Status <b>{model_status}</b></span></div>
                <div class="doc-meta"><span>Gemini <b>{settings.gemini.model_name}</b> ({gemini_status})</span></div>
                <div class="doc-meta"><span>Groq <b>{settings.groq.model_name}</b> ({groq_status})</span></div>
                <div class="doc-meta"><span>OCR <b>{settings.ocr.provider}</b></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Images use Tesseract first in auto mode, then Gemini OCR if an API key is available.")

        warm_clicked = st.button("Warm provider", use_container_width=True)

    return provider, fallback_enabled, warm_clicked


def render_document_list() -> None:
    st.markdown('<div class="section-title">Processed Documents</div>', unsafe_allow_html=True)
    if not st.session_state["documents"]:
        st.caption("No documents loaded yet.")
        return

    for doc in st.session_state["documents"]:
        st.markdown(
            f"""
            <div class="doc-row">
                <div class="doc-name">{doc['name']}</div>
                <div class="doc-meta">
                    <span>Type <b>{doc['document_type']}</b></span>
                    <span>Pages <b>{doc['page_count']}</b></span>
                    <span>Tokens <b>{doc['token_count']}</b></span>
                    <span>Chunks <b>{doc['chunk_count']}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for warning in doc.get("warnings", []):
            st.warning(warning)

    with st.expander("Extracted text preview", expanded=False):
        st.text_area(
            "Extracted text preview",
            value=st.session_state["uploaded_preview"],
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )


def render_chat_history() -> None:
    for idx, msg in enumerate(st.session_state["chat_messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                st.caption(
                    f"Provider: `{msg.get('requested_provider', '-')}` | "
                    f"Used: `{msg.get('backend', '-')}` | "
                    f"Fallback: `{msg.get('fallback_used', False)}`"
                )
            for warning in msg.get("warnings", []):
                st.caption(f"Warning: {warning}")
            if msg.get("context_excerpt"):
                with st.expander("Context used", expanded=False):
                    st.text_area(
                        "Context used",
                        value=msg["context_excerpt"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"ctx_history_{idx}",
                    )


def render_status_strip(settings: Settings, selected_provider: ProviderName) -> None:
    document_count = len(st.session_state["documents"])
    session_status = "Ready" if st.session_state["active_session_id"] else "No active session"
    st.markdown(
        f"""
        <div class="surface" style="padding:0.75rem 0.9rem;margin:1rem 0 1.1rem;">
            <div class="doc-meta" style="justify-content:space-between;gap:1rem;">
                <span>Session <b>{session_status}</b></span>
                <span>Documents <b>{document_count}</b></span>
                <span>Provider <b>{selected_provider}</b></span>
                <span>Local model <b>{settings.local_llm.model_label}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_panel(settings: Settings, chat_engine) -> None:
    st.markdown('<div class="section-title">Knowledge Base</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or image files",
        type=[ext.lstrip(".") for ext in settings.app.allowed_extensions],
        accept_multiple_files=True,
        help=f"Up to {settings.app.max_upload_files} files, max {settings.app.max_pages_per_file} pages or frames each.",
        label_visibility="collapsed",
    )

    col_process, col_clear = st.columns([1, 1])
    process_clicked = col_process.button("Process", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear", use_container_width=True)

    if clear_clicked:
        reset_chat_state()
        st.rerun()

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one document first.")
            return
        try:
            with st.spinner("Processing documents..."):
                response = chat_engine.create_session(uploaded_to_incoming(uploaded_files))
        except ChatEngineError as exc:
            st.error(exc.message)
            if exc.details:
                st.json(exc.details)
        except Exception as exc:
            st.exception(exc)
        else:
            st.session_state["active_session_id"] = response.session_id
            st.session_state["uploaded_preview"] = response.combined_preview
            st.session_state["documents"] = [document.model_dump() for document in response.documents]
            st.session_state["chat_messages"] = []
            st.success(f"Processed {len(response.documents)} document(s).")
            for warning in response.warnings:
                st.warning(warning)


def main() -> None:
    st.set_page_config(
        page_title="DocChat Engine",
        page_icon="D",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    base_settings = get_settings()
    sync_session_state(base_settings)

    selected_provider, fallback_enabled, warm_clicked = render_sidebar(base_settings)
    settings = get_settings()
    chat_engine = get_chat_engine()

    if warm_clicked:
        try:
            with st.spinner(f"Warming {selected_provider} provider..."):
                chat_engine.prepare_provider(selected_provider)
        except ChatEngineError as exc:
            st.sidebar.error(exc.message)
            if exc.details:
                st.sidebar.json(exc.details)
        except Exception as exc:
            st.sidebar.exception(exc)
        else:
            st.sidebar.success(f"{selected_provider} is ready.")

    st.title("Document Chat Engine")
    st.caption("A grounded workspace for document upload, review, and Q&A.")
    render_status_strip(settings, selected_provider)

    chat_tab, documents_tab = st.tabs(["Chat", "Documents"])

    with documents_tab:
        upload_col, docs_col = st.columns([0.9, 1.35], gap="large")
        with upload_col:
            render_upload_panel(settings, chat_engine)
        with docs_col:
            render_document_list()

    with chat_tab:
        st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)
        if not st.session_state["active_session_id"]:
            st.info("Process at least one document in the Documents tab to start asking questions.")
            return

        render_chat_history()
        user_query = st.chat_input(f"Ask about the uploaded documents using {selected_provider}")
        if not user_query:
            return

        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        try:
            with st.spinner(f"Generating answer with {selected_provider}..."):
                response = chat_engine.chat(
                    st.session_state["active_session_id"],
                    user_query,
                    provider_name=selected_provider,
                    fallback_enabled=fallback_enabled,
                )
        except ChatEngineError as exc:
            with st.chat_message("assistant"):
                st.error(exc.message)
                if exc.details:
                    st.json(exc.details)
        except Exception as exc:
            with st.chat_message("assistant"):
                st.exception(exc)
        else:
            message = {
                "role": "assistant",
                "content": response.answer,
                "warnings": response.warnings,
                "context_excerpt": response.context_excerpt,
                "backend": response.backend,
                "requested_provider": response.requested_provider,
                "fallback_used": response.fallback_used,
            }
            st.session_state["chat_messages"].append(message)
            with st.chat_message("assistant"):
                st.markdown(response.answer)
                st.caption(
                    f"Provider: `{response.requested_provider}` | "
                    f"Used: `{response.backend}` | "
                    f"Fallback: `{response.fallback_used}`"
                )
                for warning in response.warnings:
                    st.caption(f"Warning: {warning}")
                with st.expander("Context used", expanded=False):
                    st.text_area(
                        "Context used",
                        value=response.context_excerpt,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )


if __name__ == "__main__":
    main()
