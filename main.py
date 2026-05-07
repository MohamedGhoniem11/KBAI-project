# =============================================================================
# Main RAG System (All Requirements)
# =============================================================================
# Integrates all stages: Ingestion, Retrieval, Generation
# Uses Grok (xAI) exclusively

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.vectorstore import load_vectorstore, save_vectorstore
from src.retrieval import create_retrieval_engine
from src.langgraph_pipeline import run_agentic_pipeline
from src.llm_integration import create_llm_generator


class CulinaryRAGSystem:
    """Main RAG system using Grok exclusively."""
    
    def __init__(self):
        self.vectorstore = None
        self.retrieval_engine = None
        self.llm = None
        self._initialized = False
    
    def initialize(self):
        """Initialize system (Stage 1 & 2)."""
        if self._initialized:
            return
        
        # Check for Grok API key
        if "XAI_API_KEY" not in os.environ or not os.environ["XAI_API_KEY"] or os.environ["XAI_API_KEY"] == "your_xai_grok_api_key_here":
            print("Error: XAI_API_KEY not set in .env file. Grok API required.")
            sys.exit(1)
        
        print("Initializing Culinary RAG System (Grok-only)...")
        
        # Stage 1: Load vector store
        print("\n[Stage 1] Loading vector store...")
        self.vectorstore = load_vectorstore()
        
        # Stage 2: Create retrieval engine
        print("[Stage 2] Setting up retrieval engine...")
        self.retrieval_engine = create_retrieval_engine(self.vectorstore)
        
        # Initialize Grok LLM
        print("[Stage 3] Initializing Grok LLM...")
        model = os.getenv("LLM_MODEL", "grok-3-latest")
        self.llm = create_llm_generator(model)
        
        self._initialized = True
        print("\n✅ System initialized successfully")
    
    def query(self, question: str) -> dict:
        """Process query through all stages (FR-01). Returns answer + top 4 chunks for verification."""
        if not self._initialized:
            self.initialize()
        
        print(f"\n{'='*50}")
        print(f"Query: {question}")
        print('='*50)
        
        # Stage 2: Semantic Retrieval (agentic pipeline)
        result = run_agentic_pipeline(question, self.retrieval_engine)
        
        chunks = result.get("retrieved_chunks", [])
        citations = result.get("citations", [])
        
        # Stage 3: Augmented Generation (Grok-only)
        response = self.llm.generate(question, chunks, citations)
        
        return {
            "question": question,
            "answer": response["answer"],
            "retrieved_chunks": response["retrieved_chunks"],  # Top 4 chunks for verification
            "citations": response["citations"],
            "num_chunks": len(chunks)
        }
    
    def add_document(self, file_path: Path):
        """Add PDF/DOCX to KB (FR-06: KB Update)."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        print(f"Adding document: {file_path.name}")
        
        # Load document based on type
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
        
        # Add metadata
        for doc in docs:
            doc.metadata["source"] = file_path.name
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        new_chunks = splitter.split_documents(docs)
        
        # Add to vector store (no retraining needed)
        self.vectorstore.add_documents(new_chunks)
        save_vectorstore(self.vectorstore)
        
        print(f"✅ Added {len(new_chunks)} new chunks from {file_path.name}")
        return len(new_chunks)
    
    def remove_document(self, source_name: str):
        """Remove document from KB (FR-06: KB Update)."""
        print(f"Removing: {source_name}")
        print("Note: FAISS requires full rebuild to remove documents. Delete file from KB/ and run rebuild_and_test.py")
        print("For dynamic deletion, consider switching to Chroma vector store.")


_system = None


def get_system() -> CulinaryRAGSystem:
    """Get or create system."""
    global _system
    if _system is None:
        _system = CulinaryRAGSystem()
    return _system


def main():
    """Test the system via CLI (no Streamlit needed)."""
    use_standalone = "--standalone" in sys.argv
    
    if use_standalone:
        from standalone.pipeline import StandaloneRAGSystem
        system = StandaloneRAGSystem()
        print("Mode: Standalone (no LangChain)")
    else:
        system = get_system()
        print("Mode: LangChain + LangGraph")
    
    system.initialize()
    
    # Test queries
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
