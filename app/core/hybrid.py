from typing import List, Tuple, Dict
from app.core.chunking import Chunk

class HybridSearcher:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Merges dense vector and sparse keyword rankings via Reciprocal Rank Fusion (RRF).
        RRF_Score(doc) = sum(1 / (k + rank_i))
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # 1. Process dense rankings
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Process sparse rankings
        for rank, (chunk, _) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by aggregated RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(chunk_map[cid], score) for cid, score in ranked]
