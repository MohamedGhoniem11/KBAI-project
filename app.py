# =============================================================================
# Streamlit Web UI (FR-01: Natural Language Interface)
# =============================================================================
# Provides a chat-based UI for the Culinary RAG system.
# Supports switching between LangChain and Standalone modes via sidebar radio.
# Run with: streamlit run app.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings in Streamlit logs

import streamlit as st  # Web app framework — turns Python scripts into interactive UIs
from pathlib import Path

from src.config import DOMAIN_DISCLAIMER

PROJECT_ROOT = Path(__file__).parent


def ensure_kb():
    """Download + unzip KB if not present (for HF deployment)."""
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
    """Rebuild vector store if not present."""
    from src.ingestion import ingest_documents
    from src.vectorstore import create_vectorstore, save_vectorstore
    VS_DIR = PROJECT_ROOT / "data" / "vectorstore"
    if not VS_DIR.exists():
        st.info("Building vector store (takes ~10 min on first run)...")
        chunks = ingest_documents()
        st.info(f"Indexed {len(chunks)} chunks...")
        sv = create_vectorstore(chunks)
        save_vectorstore(sv)
        st.success("Vector store ready!")

# set_page_config MUST be the first Streamlit command (Streamlit requirement)
st.set_page_config(
    page_title="Culinary RAG Assistant",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS for better design (after set_page_config)
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


def init_session():
    """Initialize Streamlit session state variables (persist across reruns)."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "mode" not in st.session_state:
        st.session_state.mode = "LangChain + LangGraph"
    if "history" not in st.session_state:
        st.session_state.history = []  # List of (question, answer) tuples


def get_system_for_mode(mode: str):
    """Return the appropriate RAG system based on selected mode."""
    if mode == "Standalone (no LangChain)":
        from standalone.pipeline import StandaloneRAGSystem
        return StandaloneRAGSystem()
    from main import get_system
    return get_system()


def process_query(prompt: str):
    """Unified query processing for chat input and sample question buttons."""
    system = st.session_state.rag_system
    with st.spinner("Searching culinary knowledge base..."):
        try:
            result = system.query(prompt)
            response = result["answer"]
            st.session_state.history.append((prompt, response))
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.session_state.history.append((prompt, error_msg))


def main():
    st.title("🍳 Culinary Arts RAG Assistant")
    st.markdown("### AI-Powered Culinary Knowledge Base with Citations")
    st.markdown("---")
    
    init_session()
    
    # Ensure KB and vector store exist (first run setup for HF Spaces)
    with st.spinner("Checking knowledge base..."):
        ensure_kb()
        ensure_vectorstore()
    
    # Lazy-initialize the RAG system (once, then reuse across reruns)
    if st.session_state.rag_system is None:
        with st.spinner("Loading culinary knowledge base..."):
            st.session_state.rag_system = get_system_for_mode(st.session_state.mode)
            st.session_state.rag_system.initialize()
    
    system = st.session_state.rag_system
    
    # Sidebar: mode selection, system info, disclaimer, sample questions
    with st.sidebar:
        mode = st.radio(
            "Select RAG Engine",
            ["LangChain + LangGraph", "Standalone (no LangChain)"],
            index=0 if st.session_state.mode == "LangChain + LangGraph" else 1,
            key="mode_selector"
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.session_state.rag_system = None  # Force re-initialization
            st.rerun()
        
        st.markdown("---")
        st.header("ℹ️ System Info")
        from src.provider import get_provider
        active_provider = get_provider().value.upper()
        pipeline_text = "LangGraph Agentic (retrieve → reflect → generate)" if st.session_state.mode == "LangChain + LangGraph" else "Linear (retrieve → generate)"
        st.markdown(f"""
        **Culinary RAG System**
        - LLM: {active_provider}
        - KB: PDF/DOCX documents (add to KB/ folder)
        - Embeddings: all-MiniLM-L6-v2 (normalized)
        - Retrieval: Top-5 chunks, 0.65 cosine threshold
        - Pipeline: {pipeline_text}
        """)
        
        st.header("⚠️ Disclaimer (FR-07)")
        st.caption(DOMAIN_DISCLAIMER)
        
        st.markdown("---")
        st.header("📋 Sample Questions")
        sample_questions = [
            "How do I make fresh pasta from scratch?",
            "What is the safe internal temperature for roasted chicken?",
            "What is the Maillard reaction in cooking?",
            "How do I sharpen a chef's knife properly?",
            "What are the food safety guidelines for storing leftovers?"
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q[:20]}"):
                process_query(q)
                st.rerun()
    
    # Chat history display
    st.header("💬 Chat")
    for q, r in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(r)
    
    # Chat input at the bottom
    prompt = st.chat_input("Ask about culinary techniques, recipes, food safety...")
    
    if prompt:
        process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
