# Standalone pipeline — orchestrates embeddings + vector store + retrieval + LLM.
# No LangChain, no LangGraph. Linear pipeline (no agentic reflection loop).
import os
from pathlib import Path
from .config import VECTORSTORE_DIR, EMBEDDING_MODEL
from .embeddings import Embeddings
from .vectorstore import VectorStore
from .retrieval import RetrievalEngine
from .llm_integration import LLMGenerator


class StandaloneRAGSystem:
    """Pure-Python RAG system. Same functionality as CulinaryRAGSystem but:
    - No LangChain dependencies
    - Linear retrieve→generate (no LangGraph reflection loop)
    - JSON metadata instead of pickle"""

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retrieval_engine = None
        self.llm = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        print("Initializing Standalone RAG System (no LangChain)...")

        print("\n[Stage 1] Loading vector store...")
        self.embeddings = Embeddings(EMBEDDING_MODEL)
        self.vectorstore = VectorStore()
        if VECTORSTORE_DIR.exists():
            self.vectorstore.load(VECTORSTORE_DIR)
            print(f"  Loaded {self.vectorstore.size} vectors")
        else:
            print("  No existing vector store found. Run rebuild first.")

        print("[Stage 2] Setting up retrieval engine...")
        self.retrieval_engine = RetrievalEngine(self.vectorstore, self.embeddings)

        print("[Stage 3] Initializing Grok LLM...")
        model = os.getenv("LLM_MODEL", "grok-3-latest")
        self.llm = LLMGenerator(model)

        self._initialized = True
        print("\n✅ Standalone system initialized successfully")

    def query(self, question: str) -> dict:
        """Linear pipeline: retrieve → generate (no LangGraph reflection loop)."""
        if not self._initialized:
            self.initialize()

        print(f"\n{'='*50}")
        print(f"Query: {question}")
        print("=" * 50)

        # Single-pass retrieval (unlike LangGraph which can loop)
        chunks = self.retrieval_engine.retrieve(question)
        citations = [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in chunks
        ]

        response = self.llm.generate(question, chunks, citations)

        return {
            "question": question,
            "answer": response["answer"],
            "retrieved_chunks": response["retrieved_chunks"],
            "citations": response["citations"],
            "num_chunks": len(chunks),
        }

    def add_document(self, file_path: Path):
        """Add a PDF/DOCX to the standalone vector store."""
        from .ingestion import load_documents, chunk_documents
        from .config import CHUNK_SIZE, CHUNK_OVERLAP

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        print(f"Adding document: {file_path.name}")
        docs = load_documents([file_path])
        new_chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)

        texts = [c["content"] for c in new_chunks]
        embeddings = self.embeddings.embed_documents(texts)
        self.vectorstore.add(embeddings, new_chunks)
        self.vectorstore.save(VECTORSTORE_DIR)

        print(f"✅ Added {len(new_chunks)} new chunks from {file_path.name}")
        return len(new_chunks)

    def remove_document(self, source_name: str):
        print(f"Removing: {source_name}")
        print(
            "Note: FAISS requires full rebuild to remove documents. "
            "Delete file from KB/ and run rebuild_and_test.py with --standalone"
        )
