# =============================================================================
# Vector Store Module
# =============================================================================
# Uses FAISS for ANN search

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from .config import VECTORSTORE_DIR
from .embeddings import create_embeddings


def create_vectorstore(chunks: List[Document]) -> FAISS:
    """Create FAISS vector store from document chunks."""
    embeddings = create_embeddings()
    
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Optional[Path] = None):
    """Save vector store to disk."""
    path = path or VECTORSTORE_DIR
    path.mkdir(parents=True, exist_ok=True)
    
    vectorstore.save_local(str(path))
    print(f"Vector store saved to {path}")


def load_vectorstore(path: Optional[Path] = None) -> FAISS:
    """Load vector store from disk."""
    path = path or VECTORSTORE_DIR
    
    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at {path}")
    
    embeddings = create_embeddings()
    vectorstore = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"Vector store loaded from {path}")
    return vectorstore


def build_or_load_vectorstore(chunks: List[Document] = None) -> FAISS:
    """Build new or load existing vector store."""
    if VECTORSTORE_DIR.exists() and (VECTORSTORE_DIR / "index.faiss").exists():
        return load_vectorstore()
    
    if chunks is None:
        raise ValueError("No chunks provided and no existing vector store")
    
    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore)
    return vectorstore