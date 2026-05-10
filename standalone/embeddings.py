# Standalone embedding module — no LangChain wrapper, uses SentenceTransformer directly.
from sentence_transformers import SentenceTransformer  # Direct model loading (no wrapper)
import numpy as np


class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)  # Load once, reuse for all embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Single text → 384-dim normalized vector."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_documents(self, texts: list) -> np.ndarray:
        """Batch of texts → array of 384-dim normalized vectors."""
        return self.model.encode(texts, normalize_embeddings=True)
