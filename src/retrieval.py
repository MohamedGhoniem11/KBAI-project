# =============================================================================
# Retrieval Engine (FR-03, FR-05)
# =============================================================================
# Takes a user query → embeds it → FAISS similarity search → filter by threshold.
# Returns RetrievedChunk objects with source, page number, and relevance score.


from langchain_community.vectorstores import FAISS
from pydantic import BaseModel  # Data validation: auto type-checking + .model_dump()

from .config import SIMILARITY_THRESHOLD, TOP_K


class RetrievedChunk(BaseModel):
    """Structured output for one retrieved chunk (FR-05: includes source + page)."""
    content: str      # The actual chunk text
    source: str       # PDF filename (e.g., "The_Science_of_Good_Cooking.pdf")
    page: int = 0     # Page number (1-indexed)
    chunk_id: int = 0 # Unique chunk identifier
    score: float = 0.0  # Cosine similarity score (0-1, higher = more relevant)


class RetrievalEngine:
    """Semantic retrieval engine (FR-03). Wraps FAISS with top-k + threshold logic."""

    def __init__(self, vectorstore: FAISS, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD):
        self.vectorstore = vectorstore
        self.top_k = top_k          # How many candidates to pull from FAISS
        self.threshold = threshold  # Minimum similarity score to keep a chunk

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed query → FAISS search → filter by score ≥ threshold."""
        # similarity_search_with_score returns (Document, float) pairs
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.top_k
        )

        print(f"[Retrieval] Query: '{query}' - Found {len(docs_with_scores)} candidates")
        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            print(f"  [{i+1}] Score: {score:.4f} | Source: {source} | Page: {page}")

        # Filter: only keep chunks above similarity threshold
        retrieved = []
        for doc, score in docs_with_scores:
            print(f"[Retrieval] Chunk score: {score:.4f} (threshold: {self.threshold})")
            if score >= self.threshold:
                retrieved.append(RetrievedChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", 0),
                    chunk_id=doc.metadata.get("chunk_id", 0),
                    score=score
                ))

        print(f"[Retrieval] Returning {len(retrieved)} chunks after threshold filter")
        return retrieved

    def as_retriever(self):
        """Return as LangChain retriever object (for compatibility with LCEL chains)."""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )


def create_retrieval_engine(vectorstore: FAISS) -> RetrievalEngine:
    """Factory function — create a RetrievalEngine with default config values."""
    return RetrievalEngine(vectorstore)
