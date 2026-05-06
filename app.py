import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st

from main import get_system, CulinaryRAGSystem
from src.config import DOMAIN_DISCLAIMER

# set_page_config MUST be the first Streamlit command
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
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    # History stores (question, response) tuples
    if "history" not in st.session_state:
        st.session_state.history = []


def process_query(prompt: str):
    """Unified query processing for chat input and buttons."""
    system = st.session_state.rag_system
    with st.spinner("Searching culinary knowledge base..."):
        try:
            result = system.query(prompt)
            response = result["answer"]  # Updated key from "response" to "answer"
            # Append (question, response) tuple to history
            st.session_state.history.append((prompt, response))
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.session_state.history.append((prompt, error_msg))


def main():
    st.title("🍳 Culinary Arts RAG Assistant")
    st.markdown("### AI-Powered Culinary Knowledge Base with Citations")
    st.markdown("---")
    
    init_session()
    
    # Initialize system once
    if st.session_state.rag_system is None:
        with st.spinner("Loading culinary knowledge base..."):
            st.session_state.rag_system = get_system()
            st.session_state.rag_system.initialize()
    
    system = st.session_state.rag_system
    
    # Sidebar with disclaimer and info
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.markdown("""
        **Culinary RAG System**
        - KB: PDF/DOCX documents (add to KB/ folder)
        - Embeddings: all-MiniLM-L6-v2 (normalized)
        - Retrieval: Top-5 chunks, 0.65 cosine threshold
        - LLM: Grok (xAI) only
        - Pipeline: LangGraph Agentic (retrieve → reflect → generate)
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
    
    # Chat input
    prompt = st.chat_input("Ask about culinary techniques, recipes, food safety...")
    
    if prompt:
        process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
