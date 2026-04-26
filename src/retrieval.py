# =============================================================================
# Retrieval Engine (FR-03, FR-05)
# =============================================================================
# Retrieves top-k=5 chunks with cosine similarity >= 0.65, includes page number

from typing import List
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from .config import TOP_K, SIMILARITY_THRESHOLD


class RetrievedChunk(BaseModel):
    """Retrieved chunk with source and page number (FR-05)."""
    content: str
    source: str
    page: int = 0
    chunk_id: int = 0
    score: float = 0.0


class RetrievalEngine:
    """Semantic retrieval engine (FR-03)."""
    
    def __init__(self, vectorstore: FAISS, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.threshold = threshold
    
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Retrieve top-k chunks with similarity >= threshold (FR-03)."""
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=self.top_k
        )
        
        print(f"[Retrieval] Query: '{query}' - Found {len(docs_with_scores)} candidates")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"  [{i+1}] Score: {score:.4f} | Source: {doc.metadata.get('source', '?')} | Page: {doc.metadata.get('page', '?')}")
        
        retrieved = []
        for doc, score in docs_with_scores:
            # FAISS returns cosine similarity (when normalized)
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
        """Return as LangChain retriever."""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )


def create_retrieval_engine(vectorstore: FAISS) -> RetrievalEngine:
    """Factory function."""
    return RetrievalEngine(vectorstore)