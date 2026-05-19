# =============================================================================
# Embedding Module (FR-02)
# =============================================================================
# Wraps HuggingFace's sentence-transformers so the rest of the code doesn't
# need to know about model loading details.

from langchain_huggingface import HuggingFaceEmbeddings  # LangChain wrapper for SentenceTransformer

from .config import EMBEDDING_MODEL  # Model name: "all-MiniLM-L6-v2"


def create_embeddings():
    """Create embedding model using HuggingFace. Called once at startup."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # CPU is fine for 384-dim inference
        encode_kwargs={'normalize_embeddings': True}  # L2 normalize so inner product = cosine
    )
    return embeddings


def embed_query(embeddings: HuggingFaceEmbeddings, query: str):
    """Embed a single query text (user's question → 384-dim vector)."""
    return embeddings.embed_query(query)


def embed_documents(embeddings: HuggingFaceEmbeddings, documents: list):
    """Embed multiple documents (list of LangChain Document → array of vectors)."""
    return embeddings.embed_documents([doc.page_content for doc in documents])
