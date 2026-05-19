# Standalone retrieval — no LangChain, manual embedding + FAISS search + threshold filter.
from .config import SIMILARITY_THRESHOLD, TOP_K
from .embeddings import Embeddings
from .vectorstore import VectorStore


class RetrievalEngine:
    """Embed query → FAISS search → filter by score ≥ threshold. Pure Python version."""

    def __init__(
        self,
        vectorstore: VectorStore,
        embeddings: Embeddings,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> list[dict]:
        """Embed the query, search the FAISS index, filter results below threshold."""
        query_vec = self.embeddings.embed_query(query)  # String → 384-dim numpy array
        results = self.vectorstore.search(query_vec, self.top_k)  # FAISS nearest neighbors

        print(f"[Retrieval] Query: '{query}' - Found {len(results)} candidates")
        for r in results:
            print(f"  Score: {r['score']:.4f} | Source: {r['source']} | Page: {r['page']}")

        filtered = [r for r in results if r["score"] >= self.threshold]
        print(f"[Retrieval] Returning {len(filtered)} chunks after threshold filter")
        return filtered
