# =============================================================================
# Main RAG System (All Requirements)
# =============================================================================
# Entry point for CLI mode. Integrates all 3 stages: Ingestion → Retrieval → Generation.
# Orchestrates the LangGraph pipeline + Grok LLM call.
# Usage: python main.py             (LangChain + LangGraph mode)
#        python main.py --standalone (pure Python, no LangChain)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import sys
from pathlib import Path

# Load .env file if available (contains API keys)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.vectorstore import load_vectorstore, save_vectorstore
from src.retrieval import create_retrieval_engine
from src.langgraph_pipeline import run_agentic_pipeline  # LangGraph stateful graph
from src.llm_integration import create_llm_generator


class CulinaryRAGSystem:
    """Main orchestrator: loads vector store → creates retriëval engine → initializes LLM.
    Exposes query() for single-turn Q&A and add_document() for KB updates."""
    
    def __init__(self):
        self.vectorstore = None
        self.retrieval_engine = None
        self.llm = None
        self._initialized = False  # Lazy initialization
    
    def initialize(self):
        """Initialize all 3 stages. Safe to call multiple times (idempotent)."""
        if self._initialized:
            return
        
        print("Initializing Culinary RAG System...")
        
        # Stage 1: Load pre-built FAISS vector store (built by rebuild_and_test.py)
        print("\n[Stage 1] Loading vector store...")
        self.vectorstore = load_vectorstore()
        
        # Stage 2: Create retrieval engine (wraps FAISS with top-k + threshold logic)
        print("[Stage 2] Setting up retrieval engine...")
        self.retrieval_engine = create_retrieval_engine(self.vectorstore)
        
        # Stage 3: Initialize LLM (reads provider config from .env)
        print("[Stage 3] Initializing LLM...")
        self.llm = create_llm_generator()
        
        self._initialized = True
        print("\n✅ System initialized successfully")
    
    def query(self, question: str) -> dict:
        """Process a question through the full RAG pipeline.
        Returns: answer text, top-4 retrieved chunks (for verification), citations."""
        if not self._initialized:
            self.initialize()
        
        print(f"\n{'='*50}")
        print(f"Query: {question}")
        print('='*50)
        
        # Stage 2: Semantic Retrieval via LangGraph agentic pipeline
        # (retrieve → reflect → re-retrieve if needed → generate context)
        result = run_agentic_pipeline(question, self.retrieval_engine)
        
        chunks = result.get("retrieved_chunks", [])
        citations = result.get("citations", [])
        
        # Stage 3: Augmented Generation — Grok answers grounded in retrieved chunks
        response = self.llm.generate(question, chunks, citations)
        
        return {
            "question": question,
            "answer": response["answer"],
            "retrieved_chunks": response["retrieved_chunks"],  # Top 4 chunks for verification
            "citations": response["citations"],
            "num_chunks": len(chunks)
        }
    
    def add_document(self, file_path: Path):
        """Add a new PDF/DOCX to the knowledge base at runtime (FR-06).
        Loads the document, chunks it, appends to existing FAISS index (no full rebuild)."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        print(f"Adding document: {file_path.name}")
        
        # Load document based on extension
        docs = []
        if file_path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif file_path.suffix.lower() == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Tag all chunks with the source filename
        for doc in docs:
            doc.metadata["source"] = file_path.name
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        new_chunks = splitter.split_documents(docs)
        
        # Append to existing FAISS index (no need to rebuild from scratch)
        self.vectorstore.add_documents(new_chunks)
        save_vectorstore(self.vectorstore)
        
        print(f"✅ Added {len(new_chunks)} new chunks from {file_path.name}")
        return len(new_chunks)
    
    def remove_document(self, source_name: str):
        """Remove document from KB (FR-06: KB Update).
        NOTE: FAISS does NOT support individual document deletion — requires full rebuild."""
        print(f"Removing: {source_name}")
        print("Note: FAISS requires full rebuild to remove documents. Delete file from KB/ and run rebuild_and_test.py")
        print("For dynamic deletion, consider switching to Chroma vector store.")


_system = None  # Singleton instance


def get_system() -> CulinaryRAGSystem:
    """Get or create the singleton RAG system (lazy initialization)."""
    global _system
    if _system is None:
        _system = CulinaryRAGSystem()
    return _system


def main():
    """CLI entry point: runs 3 test queries to demonstrate the system."""
    use_standalone = "--standalone" in sys.argv
    
    if use_standalone:
        from standalone.pipeline import StandaloneRAGSystem
        system = StandaloneRAGSystem()
        print("Mode: Standalone (no LangChain)")
    else:
        system = get_system()
        print("Mode: LangChain + LangGraph")
    
    system.initialize()
    
    # Test queries covering different culinary topics
    test_queries = [
        "How do I make fresh pasta?",
        "What is the safe internal temperature for chicken?",
        "What are the basic knife skills for beginners?"
    ]
    
    for query in test_queries:
        result = system.query(query)
        print(f"\nANSWER:\n{result['answer'][:500]}...")
        print(f"\nRETRIEVED {len(result['retrieved_chunks'])} CHUNKS FOR VERIFICATION:")
        for chunk in result["retrieved_chunks"]:
            print(f"  [Source {chunk['source']}, Page {chunk['page']}] Score: {chunk['score']:.3f}")
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
