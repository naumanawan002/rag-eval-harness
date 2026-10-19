import numpy as np
from typing import List, Tuple, Dict
from app.core.chunking import Chunk

class InMemoryVectorStore:
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}

    def upsert(self, chunk: Chunk, vector: List[float]):
        self.chunks[chunk.chunk_id] = chunk
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self.vectors[chunk.chunk_id] = v

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if not self.vectors:
            return []

        qv = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        if q_norm > 0:
            qv = qv / q_norm

        scores = []
        for chunk_id, vec in self.vectors.items():
            # Cosine similarity between normalized vectors is dot product
            sim = float(np.dot(qv, vec))
            scores.append((self.chunks[chunk_id], sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def clear(self):
        self.chunks.clear()
        self.vectors.clear()
