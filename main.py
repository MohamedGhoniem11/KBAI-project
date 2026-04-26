# =============================================================================
# Main RAG System (All Requirements)
# =============================================================================
# Integrates all stages: Ingestion, Retrieval, Generation

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import sys
sys.path.insert(0, "C:/study folder/Kbai/assgiment")

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for API key
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("Warning: OPENAI_API_KEY not set. LLM features will use fallback mode.")

from src.vectorstore import load_vectorstore, save_vectorstore
from src.retrieval import create_retrieval_engine
from src.langgraph_pipeline import run_agentic_pipeline
from src.llm_integration import create_llm_generator


class CulinaryRAGSystem:
    """Main RAG system."""
    
    def __init__(self):
        self.vectorstore = None
        self.retrieval_engine = None
        self.llm = None
        self._initialized = False
    
    def initialize(self):
        """Initialize system (Stage 1 & 2)."""
        if self._initialized:
            return
        
        print("Initializing Culinary RAG System...")
        
        # Stage 1: Load vector store
        print("\n[Stage 1] Loading vector store...")
        self.vectorstore = load_vectorstore()
        
        # Stage 2: Create retrieval engine
        print("[Stage 2] Setting up retrieval engine...")
        self.retrieval_engine = create_retrieval_engine(self.vectorstore)
        
        # Initialize LLM
        print("[Stage 3] Initializing LLM...")
        provider = os.getenv("LLM_PROVIDER", "groq")  # Default to Groq (free API)
        model = os.getenv("LLM_MODEL", "llama3-8b-8192")  # Groq free tier model
        self.llm = create_llm_generator(provider, model)
        
        self._initialized = True
        print("\nSystem initialized successfully")
    
    def query(self, question: str) -> dict:
        """Process query through all stages (FR-01)."""
        if not self._initialized:
            self.initialize()
        
        print(f"\n{'='*50}")
        print(f"Query: {question}")
        print('='*50)
        
        # Stage 2: Semantic Retrieval
        result = run_agentic_pipeline(question, self.retrieval_engine)
        
        chunks = result.get("retrieved_chunks", [])
        citations = result.get("citations", [])
        
        # Stage 3: Augmented Generation
        response = self.llm.generate(question, chunks, citations)
        
        return {
            "question": question,
            "response": response,
            "citations": citations,
            "num_chunks": len(chunks)
        }
    
    def add_document(self, pdf_path: Path):
        """Add document to KB (FR-06: KB Update)."""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP
        
        print(f"Adding document: {pdf_path.name}")
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")
        
        # Load new document
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        new_chunks = splitter.split_documents(docs)
        
        # Add to vector store
        self.vectorstore.add_documents(new_chunks)
        save_vectorstore(self.vectorstore)
        
        print(f"Added {len(new_chunks)} new chunks")
        return len(new_chunks)
    
    def remove_document(self, source_name: str):
        """Remove document from KB (FR-06: KB Update)."""
        print(f"Removing: {source_name}")
        print("Note: Rebuild required to remove documents")
        # FAISS doesn't support direct deletion
        # Would need to rebuild vector store


_system = None


def get_system() -> CulinaryRAGSystem:
    """Get or create system."""
    global _system
    if _system is None:
        _system = CulinaryRAGSystem()
    return _system


def main():
    """Test the system."""
    system = get_system()
    system.initialize()
    
    # Test queries
    test_queries = [
        "How do I make pasta?",
        "What is the safe temperature for chicken?"
    ]
    
    for query in test_queries:
        result = system.query(query)
        print(f"\nRESPONSE:\n{result['response'][:500]}")
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
