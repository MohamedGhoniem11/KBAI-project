import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Culinary RAG Assistant",
    page_icon="🍳",
    layout="wide",
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
    KB_DIR = PROJECT_ROOT / "KB"
    KB_ZIP = PROJECT_ROOT / "KB.zip"
    DRIVE_URL = "https://drive.google.com/uc?export=download&id=1RskXkZXqQiszdQ8QkEYySKlgToQPBZO4"

    if KB_DIR.exists() and any(KB_DIR.iterdir()):
        return

    if not KB_ZIP.exists():
        st.info("Downloading knowledge base (272MB, first run only)...")
        import urllib.request
        import zipfile

        def dl(count, block, total):
            if total > 0 and count % 20 == 0:
                st.progress(min(int(count * block * 100 / total), 100))

        urllib.request.urlretrieve(DRIVE_URL, KB_ZIP, dl)

    st.info("Extracting knowledge base...")
    with zipfile.ZipFile(KB_ZIP, "r") as z:
        z.extractall(PROJECT_ROOT)


def ensure_vectorstore():
    from src.ingestion import ingest_documents
    from src.vectorstore import create_vectorstore, save_vectorstore

    VS_DIR = PROJECT_ROOT / "data" / "vectorstore"
    if not VS_DIR.exists():
        st.info("Building vector store (takes ~10 min on first run)...")
        chunks = ingest_documents()
        st.info(f"Indexed {len(chunks)} chunks...")
        create_and_save = create_vectorstore(chunks)
        save_vectorstore(create_and_save)
        st.success("Vector store ready!")


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
            st.session_state.history.append((prompt, result["answer"]))
        except Exception as e:
            st.session_state.history.append((prompt, f"Error: {e}"))


def main():
    from src.config import DOMAIN_DISCLAIMER
    from src.provider import get_provider

    st.title("Culinary Arts RAG Assistant")
    st.markdown("### AI-Powered Culinary Knowledge Base with Citations")
    st.markdown("---")

    with st.spinner("Initializing..."):
        ensure_kb()
        ensure_vectorstore()

    init_session()

    if st.session_state.rag_system is None:
        with st.spinner("Loading..."):
            st.session_state.rag_system = get_system_for_mode(st.session_state.mode)
            st.session_state.rag_system.initialize()

    with st.sidebar:
        mode = st.radio(
            "Select RAG Engine",
            ["LangChain + LangGraph", "Standalone (no LangChain)"],
            index=0,
            key="mode_selector",
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.session_state.rag_system = None
            st.rerun()

        st.markdown("---")
        st.header("System Info")
        active_provider = get_provider().value.upper()
        st.markdown(f"""
        **Culinary RAG System**
        - LLM: {active_provider}
        - KB: 9 PDF cookbooks (295MB)
        - Embeddings: all-MiniLM-L6-v2
        - Retrieval: Top-5, 0.65 threshold
        """)

        st.header("Disclaimer")
        st.caption(DOMAIN_DISCLAIMER)

        st.markdown("---")
        st.header("Sample Questions")
        questions = [
            "How do I make fresh pasta?",
            "Safe chicken temperature?",
            "What is the Maillard reaction?",
        ]
        for q in questions:
            if st.button(q, key=f"q_{q[:15]}"):
                process_query(q)
                st.rerun()

    st.header("Chat")
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