# =============================================================================
# HF Deployment - Auto-download KB + Vector Store
# =============================================================================
# This version downloads KB from Google Drive on first run
# Use this for HuggingFace Spaces deployment

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Culinary RAG Assistant",
    page_icon="🍳",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stTitle {color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;}
    .stChatMessage {border-radius: 10px; padding: 12px; margin: 8px 0;}
    .stChatMessage[data-testid="stChatMessage-user"] {background-color: #e3f2fd;}
    .stChatMessage[data-testid="stChatMessage-assistant"] {background-color: #f0f4c3;}
    .sidebar .sidebar-content {background-color: #ffffff; border-radius: 10px; padding: 15px;}
    .source-tag {font-size: 0.8em; color: #666; font-style: italic;}
    .stButton>button {border-radius: 8px; background-color: #3498db; color: white;}
    .stButton>button:hover {background-color: #2980b9;}
</style>
""", unsafe_allow_html=True)


def ensure_kb():
    """Download + unzip KB if not present (for HF deployment)."""
    KB_DIR = PROJECT_ROOT / "KB"
    KB_ZIP = PROJECT_ROOT / "KB.zip"
    DRIVE_URL = "https://drive.google.com/uc?export=download&id=1RskXkZXqQiszdQ8QkEYySKlgToQPBZO4"

    if KB_DIR.exists() and any(KB_DIR.iterdir()):
        return

    if not KB_ZIP.exists():
        st.info("📥 Downloading knowledge base (272MB, first run only)...")
        import urllib.request
        import zipfile
        import io

        def download_progress(count, block_size, total_size):
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                if pct % 20 == 0:
                    st.progress(pct / 100, text=f"Downloading... {pct}%")

        urllib.request.urlretrieve(DRIVE_URL, KB_ZIP, download_progress)

    st.info("📦 Extracting knowledge base...")
    with zipfile.ZipFile(KB_ZIP, 'r') as zip_ref:
        zip_ref.extractall(PROJECT_ROOT)


def ensure_vectorstore():
    """Rebuild vector store if not present."""
    from src.ingestion import ingest_documents
    from src.vectorstore import create_vectorstore, save_vectorstore

    VS_DIR = PROJECT_ROOT / "data" / "vectorstore"
    if not VS_DIR.exists():
        st.info("🔨 Building vector store (this takes ~10 minutes on first run)...")
        chunks = ingest_documents()
        st.info(f"   Indexed {len(chunks)} document chunks...")
        vectorstore = create_vectorstore(chunks)
        save_vectorstore(vectorstore)
        st.success("✅ Vector store ready!")


def init_session():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "mode" not in st.session_state:
        st.session_state.mode = "LangChain + LangGraph"
    if "history" not in st.session_state:
        st.session_state.history = []


def get_system_for_mode(mode: str):
    if mode == "Standalone (no LangChain)":
        from standalone.pipeline import StandaloneRAGSystem
        return StandaloneRAGSystem()
    from main import get_system
    return get_system()


def process_query(prompt: str):
    system = st.session_state.rag_system
    with st.spinner("Searching culinary knowledge base..."):
        try:
            result = system.query(prompt)
            response = result["answer"]
            st.session_state.history.append((prompt, response))
        except Exception as e:
            st.session_state.history.append((prompt, f"Error: {str(e)}"))


def main():
    from src.config import DOMAIN_DISCLAIMER

    st.title("🍳 Culinary Arts RAG Assistant")
    st.markdown("### AI-Powered Culinary Knowledge Base with Citations")
    st.markdown("---")

    with st.spinner("Initializing..."):
        ensure_kb()
        ensure_vectorstore()

    init_session()

    if st.session_state.rag_system is None:
        with st.spinner("Loading culinary knowledge base..."):
            st.session_state.rag_system = get_system_for_mode(st.session_state.mode)
            st.session_state.rag_system.initialize()

    with st.sidebar:
        mode = st.radio(
            "Select RAG Engine",
            ["LangChain + LangGraph", "Standalone (no LangChain)"],
            index=0,
            key="mode_selector"
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.session_state.rag_system = None
            st.rerun()

        st.markdown("---")
        st.header("ℹ️ System Info")
        st.markdown("""
        **Culinary RAG System**
        - KB: 9 PDF cookbooks (295MB)
        - Embeddings: all-MiniLM-L6-v2
        - Retrieval: Top-5 chunks, 0.65 cosine threshold
        - LLM: Grok (xAI)
        """)

        st.header("⚠️ Disclaimer")
        st.caption(DOMAIN_DISCLAIMER)

        st.markdown("---")
        st.header("📋 Sample Questions")
        questions = [
            "How do I make fresh pasta?",
            "Safe chicken temperature?",
            "What is the Maillard reaction?",
        ]
        for q in questions:
            if st.button(q, key=f"q_{q[:15]}"):
                process_query(q)
                st.rerun()

    st.header("💬 Chat")
    for q, r in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(r)

    prompt = st.chat_input("Ask about culinary techniques, recipes, food safety...")
    if prompt:
        process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()