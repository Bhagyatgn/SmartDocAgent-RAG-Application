from pathlib import Path
import json
import os

import streamlit as st
from dotenv import load_dotenv
from simple_rag_core import build_agent_from_docs, parse_uploaded_pdfs

APP_NAME = "SmartDocAgent"
CHAT_HISTORY_FILE = Path("chat_history.json")

load_dotenv()

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(1200px 500px at 50% -15%, #1f2a44 0%, #0b1220 45%, #080d18 100%);
                color: #e2e8f0;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                max-width: 1200px;
            }
            .left-logo {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                font-weight: 700;
                font-size: 1.02rem;
                margin-bottom: 1rem;
                color: #f8fafc !important;
                line-height: 1.2;
                word-break: break-word;
            }
            .logo-badge {
                width: 28px;
                height: 28px;
                border-radius: 8px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
                color: white;
                font-weight: 700;
            }
            .section-card {
                background: rgba(16, 24, 39, 0.84);
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.8rem;
                color: #e2e8f0;
                box-shadow: 0 10px 24px rgba(2, 6, 23, 0.42);
            }
            .sidebar-card {
                background: rgba(15, 23, 42, 0.86);
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 0.7rem 0.75rem;
                margin-bottom: 0.7rem;
                box-shadow: 0 6px 16px rgba(2, 6, 23, 0.35);
            }
            .msg-user {
                background: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
                color: #fff;
                border-radius: 14px 14px 4px 14px;
                padding: 0.7rem 0.9rem;
                margin: 0.45rem 0 0.45rem auto;
                width: fit-content;
                max-width: 72%;
                font-size: 0.95rem;
                box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
            }
            .msg-bot {
                background: rgba(2, 6, 23, 0.86);
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 14px 14px 14px 4px;
                padding: 0.75rem 0.95rem;
                margin: 0.45rem auto 0.45rem 0;
                width: fit-content;
                max-width: 80%;
                font-size: 0.95rem;
                box-shadow: 0 6px 16px rgba(2, 6, 23, 0.45);
            }
            .file-card {
                background: rgba(2, 6, 23, 0.95);
                border: 1px solid #93c5fd;
                border-radius: 12px;
                padding: 0.55rem 0.7rem;
                margin: 0.35rem 0;
                color: #f8fafc !important;
            }
            .sticky-input {
                position: sticky;
                bottom: 0.4rem;
                background: rgba(15, 23, 42, 0.96);
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 0.45rem;
                margin-top: 0.8rem;
            }
            .tiny-note {
                color: #94a3b8;
                font-size: 0.82rem;
            }
            .sidebar-meta {
                color: #f8fafc !important;
                font-size: 1rem;
                font-weight: 600;
                margin: 0.15rem 0;
            }
            .sidebar-subtle {
                color: #e2e8f0 !important;
                font-size: 1rem;
                margin-top: 0.35rem;
            }
            div[data-testid="stSidebar"] {
                background: #020617 !important;
                border-right: 1px solid #93c5fd !important;
            }
            section[data-testid="stSidebar"] {
                background: #020617 !important;
                border-right: 1px solid #93c5fd !important;
            }
            div[data-testid="stSidebar"] * {
                color: #f8fafc !important;
                opacity: 1 !important;
            }
            h2, h3, h4 {
                color: #f8fafc !important;
            }
            label, .stCaption {
                color: #cbd5e1 !important;
            }
            div[data-testid="stSidebar"] p,
            div[data-testid="stSidebar"] span,
            div[data-testid="stSidebar"] li,
            div[data-testid="stSidebar"] label,
            div[data-testid="stSidebar"] .stMarkdown,
            div[data-testid="stSidebar"] .stCaption {
                color: #e2e8f0 !important;
                opacity: 1 !important;
            }
            .stButton > button, .stFormSubmitButton > button {
                border-radius: 12px !important;
                border: 1px solid #93c5fd !important;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
                color: #f8fafc !important;
            }
            div[data-testid="stSidebar"] .stButton > button {
                width: 100% !important;
                background: #0b1220 !important;
                border: 2px solid #93c5fd !important;
                border-radius: 14px !important;
                padding: 0.6rem 0.7rem !important;
                font-weight: 600 !important;
                box-shadow: 0 8px 18px rgba(2, 6, 23, 0.45);
                margin-bottom: 0.35rem !important;
                color: #f8fafc !important;
            }
            div[data-testid="stSidebar"] .stMarkdown h3 {
                color: #f8fafc !important;
                margin-bottom: 0.4rem !important;
                font-weight: 700 !important;
            }
            div[data-testid="stSidebar"] .stCaption {
                color: #e2e8f0 !important;
            }
            div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
            div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
                color: #f8fafc !important;
                opacity: 1 !important;
            }
            .stTextInput > div > div > input {
                background: #0f172a !important;
                color: #e2e8f0 !important;
                border: 1px solid #334155 !important;
                border-radius: 10px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Study Chat"
    if "docs_by_file" not in st.session_state:
        st.session_state.docs_by_file = {}
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = load_chat_history()
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = None
    if "rag_status" not in st.session_state:
        st.session_state.rag_status = "No PDFs loaded"
    if "parse_errors" not in st.session_state:
        st.session_state.parse_errors = []
    if "bootstrap_done" not in st.session_state:
        st.session_state.bootstrap_done = False
    if "pending_user_query" not in st.session_state:
        st.session_state.pending_user_query = None


def bootstrap_existing_pdfs() -> None:
    if st.session_state.bootstrap_done:
        return

    if st.session_state.docs_by_file:
        st.session_state.bootstrap_done = True
        return

    upload_dir = Path("uploaded_pdfs")
    if not upload_dir.exists():
        st.session_state.bootstrap_done = True
        return

    existing_pdf_paths = sorted(upload_dir.glob("*.pdf"))
    if not existing_pdf_paths:
        st.session_state.bootstrap_done = True
        return

    uploaded_files_data: list[tuple[str, bytes]] = []
    for pdf_path in existing_pdf_paths:
        try:
            uploaded_files_data.append((pdf_path.name, pdf_path.read_bytes()))
        except OSError:
            continue

    if not uploaded_files_data:
        st.session_state.bootstrap_done = True
        return

    docs_by_file, _, _, parse_errors = parse_uploaded_pdfs(
        uploaded_files_data,
        upload_dir="uploaded_pdfs",
        persist_files=False,
    )
    st.session_state.docs_by_file = docs_by_file
    st.session_state.parse_errors = parse_errors
    refresh_rag_agent()
    st.session_state.bootstrap_done = True


def default_intro_message() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": f"Hi 👋 I am {APP_NAME}. Upload PDFs, then ask anything. I answer each document separately with page references.",
        }
    ]


def load_chat_history() -> list[dict[str, str]]:
    if not CHAT_HISTORY_FILE.exists():
        return default_intro_message()

    try:
        loaded = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            return default_intro_message()

        validated = []
        for item in loaded:
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
                validated.append({"role": item["role"], "content": item["content"]})

        return validated if validated else default_intro_message()
    except (json.JSONDecodeError, OSError):
        return default_intro_message()


def save_chat_history() -> None:
    try:
        CHAT_HISTORY_FILE.write_text(
            json.dumps(st.session_state.chat_messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def save_and_process_uploads(uploaded_files: list) -> tuple[int, int]:
    uploaded_files_data = [(uploaded_file.name, uploaded_file.getbuffer().tobytes()) for uploaded_file in uploaded_files]
    docs_by_file, total_files, total_pages, parse_errors = parse_uploaded_pdfs(
        uploaded_files_data,
        upload_dir="uploaded_pdfs",
    )
    st.session_state.docs_by_file = docs_by_file
    st.session_state.parse_errors = parse_errors

    refresh_rag_agent()

    return total_files, total_pages


def reload_docs_from_disk() -> tuple[int, int]:
    upload_dir = Path("uploaded_pdfs")
    if not upload_dir.exists():
        st.session_state.docs_by_file = {}
        st.session_state.parse_errors = []
        refresh_rag_agent()
        return 0, 0

    existing_pdf_paths = sorted(upload_dir.glob("*.pdf"))
    uploaded_files_data: list[tuple[str, bytes]] = []
    for pdf_path in existing_pdf_paths:
        try:
            uploaded_files_data.append((pdf_path.name, pdf_path.read_bytes()))
        except OSError:
            continue

    docs_by_file, total_files, total_pages, parse_errors = parse_uploaded_pdfs(
        uploaded_files_data,
        upload_dir="uploaded_pdfs",
        persist_files=False,
    )
    st.session_state.docs_by_file = docs_by_file
    st.session_state.parse_errors = parse_errors
    refresh_rag_agent()
    return total_files, total_pages


def remove_deleted_pdf_memory(file_name: str) -> None:
    lowered_name = file_name.lower()
    filtered_messages: list[dict[str, str]] = []

    for message in st.session_state.chat_messages:
        content = message.get("content", "")
        lowered_content = content.lower()

        if message.get("role") == "assistant" and (
            lowered_name in lowered_content
            or f"document: {lowered_name}" in lowered_content
            or f"{file_name} (page".lower() in lowered_content
        ):
            continue

        filtered_messages.append(message)

    st.session_state.chat_messages = filtered_messages if filtered_messages else default_intro_message()
    st.session_state.pending_user_query = None
    save_chat_history()


def refresh_rag_agent() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    agent, status = build_agent_from_docs(st.session_state.docs_by_file, groq_api_key=api_key)
    st.session_state.rag_agent = agent
    st.session_state.rag_status = status


def answer_from_docs(query: str) -> str:
    if not st.session_state.docs_by_file:
        return "Please upload and process PDF files first from **Upload & Process**."

    if st.session_state.rag_agent is None:
        refresh_rag_agent()

    agent = st.session_state.rag_agent
    if agent is None:
        return (
            "RAG system is not ready. "
            f"Status: {st.session_state.rag_status}. "
            "Ensure dependencies are installed and GROQ_API_KEY is set."
        )

    try:
        return agent.run(query)
    except Exception as error:
        error_text = str(error)
        st.session_state.rag_status = f"Runtime error: {error_text}"
        if "invalid_api_key" in error_text or "Invalid API Key" in error_text or "AuthenticationError" in error_text:
            st.session_state.rag_status = "Invalid GROQ_API_KEY (401)"
            return "RAG system is not ready. Status: Invalid GROQ_API_KEY (401). Please set a valid key and retry."
        return f"RAG runtime error: {error_text}"


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f'<div class="left-logo"><span class="logo-badge">S</span>{APP_NAME}</div>', unsafe_allow_html=True)

        if st.button("Upload & Process", use_container_width=True):
            st.session_state.active_page = "Upload & Process"

        if st.button("Study Chat", use_container_width=True):
            st.session_state.active_page = "Study Chat"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📚 Workspace")
        st.markdown(f"<div class='sidebar-meta'>Files loaded: {len(st.session_state.docs_by_file)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-meta'>RAG status: {st.session_state.rag_status}</div>", unsafe_allow_html=True)
        if st.session_state.docs_by_file:
            for name, pages in st.session_state.docs_by_file.items():
                st.markdown(
                    f"<div class='file-card'><b>📄 {name}</b><br><span class='tiny-note'>{len(pages)} pages indexed</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='sidebar-subtle'>No PDFs loaded yet</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💾 Chat Memory")
        st.markdown(f"<div class='sidebar-meta'>Messages remembered: {len(st.session_state.chat_messages)}</div>", unsafe_allow_html=True)
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = default_intro_message()
            st.session_state.pending_user_query = None
            save_chat_history()
            st.rerun()


def render_upload_page() -> None:
    st.markdown("## Upload & Process")
    st.markdown('<div class="section-card">Upload one or more PDFs and index them for chat.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Select PDF file(s)", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        process = st.button("Process", use_container_width=True)
    with col2:
        st.markdown("<div class='tiny-note'>Tip: After processing, switch to Study Chat and ask your question.</div>", unsafe_allow_html=True)

    if process:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
            return

        with st.spinner("Processing PDFs..."):
            files_count, pages_count = save_and_process_uploads(uploaded_files)

        st.success(f"Processed {files_count} file(s), total {pages_count} page(s).")
        if st.session_state.get("parse_errors"):
            st.warning("Some PDFs could not be read and were skipped:")
            for parse_error in st.session_state["parse_errors"]:
                st.caption(f"- {parse_error}")

    if st.session_state.docs_by_file:
        st.markdown("### Processed Files")
        for name, docs in st.session_state.docs_by_file.items():
            col_file, col_delete = st.columns([4, 1])
            with col_file:
                st.markdown(
                    f"<div class='file-card'><b>{name}</b><br><span class='tiny-note'>{len(docs)} pages indexed</span></div>",
                    unsafe_allow_html=True,
                )
            with col_delete:
                if st.button("🗑️", key=f"delete_pdf_{name}", help=f"Delete {name}", use_container_width=True):
                    target_path = Path("uploaded_pdfs") / name
                    try:
                        if target_path.exists():
                            target_path.unlink()
                        remove_deleted_pdf_memory(name)
                        files_count, pages_count = reload_docs_from_disk()
                        st.success(f"Deleted {name}. Now indexed {files_count} file(s), {pages_count} page(s).")
                        st.rerun()
                    except OSError as error:
                        st.error(f"Could not delete {name}: {error}")


def render_chat_page() -> None:
    st.markdown("## Study Chat")

    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f"<div class='msg-user'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-bot'>{message['content']}</div>", unsafe_allow_html=True)

    st.markdown('<div class="sticky-input">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask about your documents", placeholder=f"Ask {APP_NAME} anything about your uploaded PDFs...")
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        cleaned_query = user_query.strip()
        st.session_state.chat_messages.append({"role": "user", "content": cleaned_query})
        st.session_state.pending_user_query = cleaned_query
        save_chat_history()
        st.rerun()

    if st.session_state.pending_user_query:
        pending_query = st.session_state.pending_user_query
        with st.spinner("Thinking..."):
            answer = answer_from_docs(pending_query)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.pending_user_query = None
        save_chat_history()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


apply_custom_theme()
init_state()
bootstrap_existing_pdfs()
render_sidebar()

if st.session_state.active_page == "Upload & Process":
    render_upload_page()
else:
    render_chat_page()
