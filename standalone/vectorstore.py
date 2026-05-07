import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
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
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self, path: Path):
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
