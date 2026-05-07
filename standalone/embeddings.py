from sentence_transformers import SentenceTransformer
import numpy as np


class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_documents(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
