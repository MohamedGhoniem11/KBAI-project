# =============================================================================
# Vector Store Module
# =============================================================================
# Manages FAISS (Facebook AI Similarity Search) index — our vector database.
# Provides create/save/load helpers so the rest of the code doesn't touch FAISS directly.
# Uses Approximate Nearest Neighbor (ANN) for fast semantic search.

from pathlib import Path

from langchain_community.vectorstores import FAISS  # LangChain wrapper for FAISS
from langchain_community.vectorstores.utils import DistanceStrategy  # Enum: COSINE, IP, etc.
from langchain_core.documents import Document

from .config import VECTORSTORE_DIR
from .embeddings import create_embeddings


def create_vectorstore(chunks: list[Document]) -> FAISS:
    """Embed all chunks and build a new FAISS index from scratch."""
    embeddings = create_embeddings()

    # MAX_INNER_PRODUCT = cosine similarity when embeddings are L2-normalized
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Path | None = None):
    """Persist FAISS index + metadata to disk (creates index.faiss + index.pkl)."""
    path = path or VECTORSTORE_DIR
    path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(path))
    print(f"Vector store saved to {path}")


def load_vectorstore(path: Path | None = None) -> FAISS:
    """Load previously saved FAISS index + metadata from disk."""
    path = path or VECTORSTORE_DIR

    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at {path}")

    embeddings = create_embeddings()
    # allow_dangerous_deserialization=True — required by LangChain for pickle-based metadata
    vectorstore = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"Vector store loaded from {path}")
    return vectorstore


def build_or_load_vectorstore(chunks: list[Document] = None) -> FAISS:
    """Load existing index if available, otherwise build new one from chunks."""
    if VECTORSTORE_DIR.exists() and (VECTORSTORE_DIR / "index.faiss").exists():
        return load_vectorstore()

    if chunks is None:
        raise ValueError("No chunks provided and no existing vector store")

    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore)
    return vectorstore
