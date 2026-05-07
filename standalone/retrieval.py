from typing import List, Dict, Optional
from .embeddings import Embeddings
from .vectorstore import VectorStore
from .config import TOP_K, SIMILARITY_THRESHOLD


class RetrievalEngine:
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

    def retrieve(self, query: str) -> List[Dict]:
        query_vec = self.embeddings.embed_query(query)
        results = self.vectorstore.search(query_vec, self.top_k)

        print(f"[Retrieval] Query: '{query}' - Found {len(results)} candidates")
        for r in results:
            print(f"  Score: {r['score']:.4f} | Source: {r['source']} | Page: {r['page']}")

        filtered = [r for r in results if r["score"] >= self.threshold]
        print(f"[Retrieval] Returning {len(filtered)} chunks after threshold filter")
        return filtered
