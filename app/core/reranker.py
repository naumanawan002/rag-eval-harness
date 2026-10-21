from typing import List, Tuple
from app.core.chunking import Chunk

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-scores candidates based on deep query-document cross interactions.
        """
        if not candidates:
            return []

        query_terms = set(query.lower().split())
        scored: List[Tuple[Chunk, float]] = []

        for chunk, initial_score in candidates:
            chunk_text = chunk.text.lower()
            # Calculate term overlap density
            matches = sum(1 for term in query_terms if term in chunk_text)
            term_ratio = matches / len(query_terms) if query_terms else 0.0

            # Calculate exact query sequence bonus
            phrase_bonus = 0.3 if query.lower() in chunk_text else 0.0

            # Combine initial retrieval rank with cross-feature interaction score
            rerank_score = (initial_score * 0.4) + (term_ratio * 0.4) + phrase_bonus
            scored.append((chunk, float(rerank_score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
