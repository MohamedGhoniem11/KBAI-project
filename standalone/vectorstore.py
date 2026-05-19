# Standalone vector store — no LangChain FAISS wrapper, uses raw FAISS API + manual JSON metadata.
import json
from pathlib import Path

import faiss  # Facebook AI Similarity Search — raw C++ bindings (no LangChain wrapper)
import numpy as np


class VectorStore:
    """FAISS vector store with manual metadata management (JSON instead of pickle).
    Stores 384-dim normalized vectors with Inner Product (cosine) similarity."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine for normalized vectors
        self.metadata: list[dict] = []  # Parallel list: index position ↔ metadata dict

    @property
    def size(self) -> int:
        return self.index.ntotal  # Number of vectors currently in the index

    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        """Add new vectors + their metadata to the index."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)  # Single vector → 2D array
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """Find k nearest neighbors by cosine similarity. Returns metadata dicts with score."""
        if self.size == 0:
            return []
        k = min(k, self.size)
        scores, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        return results

    def save(self, path: Path):
        """Persist FAISS index (binary) + metadata (JSON) to disk.
        JSON is human-readable unlike LangChain's pickle format."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, path: Path):
        """Load previously saved FAISS index + metadata from disk."""
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
