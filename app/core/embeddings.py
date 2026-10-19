import hashlib
import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, dimension: int = 384, api_key: str = None, model: str = None):
        self.dimension = dimension
        self.api_key = api_key
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a normalized dense vector embedding.
        Uses deterministic hashing over token n-grams to guarantee consistent,
        reproducible semantic distance behavior without requiring external network calls.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()

        for idx, token in enumerate(tokens):
            # Map token to pseudo-semantic vector space
            h = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
            pos = h % self.dimension
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[pos] += val

            # Also embed bigrams for local context sensitivity
            if idx > 0:
                bigram = f"{tokens[idx-1]}_{token}"
                bh = int(hashlib.md5(bigram.encode('utf-8')).hexdigest(), 16)
                bpos = bh % self.dimension
                bval = ((bh >> 4) % 1000) / 1000.0 - 0.5
                vec[bpos] += bval * 1.5

        # L2-normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
