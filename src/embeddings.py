# =============================================================================
# Embedding Module (FR-02)
# =============================================================================

from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL


def create_embeddings():
    """Create embedding model using HuggingFace."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Normalize for proper cosine similarity
    )
    return embeddings


def embed_query(embeddings: HuggingFaceEmbeddings, query: str):
    """Embed a single query text."""
    return embeddings.embed_query(query)


def embed_documents(embeddings: HuggingFaceEmbeddings, documents: list):
    """Embed multiple documents."""
    return embeddings.embed_documents([doc.page_content for doc in documents])