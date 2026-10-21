import time
from typing import List, Dict, Any
from app.core.chunking import Chunk
from app.core.bm25 import BM25Index
from app.core.embeddings import EmbeddingEngine
from app.core.hybrid import HybridSearcher
from app.core.reranker import CrossEncoderReranker
from app.storage.vector_store import InMemoryVectorStore
from app.evaluation.metrics import RAGEvaluationMetrics

class EvaluationHarness:
    def __init__(
        self,
        chunks: List[Chunk],
        embedding_engine: EmbeddingEngine,
        vector_store: InMemoryVectorStore,
        bm25_index: BM25Index,
        hybrid_searcher: HybridSearcher,
        reranker: CrossEncoderReranker
    ):
        self.chunks = chunks
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker

    def run_benchmark(self, eval_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs comparative benchmark across 4 distinct retrieval strategies.
        """
        strategies = ["Dense_Only", "Sparse_BM25", "Hybrid_RRF", "Hybrid_Plus_Rerank"]
        results = {
            strat: {
                "hit_rate_at_3": 0.0,
                "hit_rate_at_5": 0.0,
                "mrr": 0.0,
                "precision_at_5": 0.0,
                "avg_latency_ms": 0.0,
            }
            for strat in strategies
        }

        num_queries = len(eval_dataset)
        if num_queries == 0:
            return results

        for item in eval_dataset:
            query = item["query"]
            ground_truth_chunks = set(item["expected_chunk_ids"])

            # 1. Dense Only
            t0 = time.perf_counter()
            q_vec = self.embedding_engine.embed_text(query)
            dense_res = self.vector_store.search(q_vec, top_k=10)
            dense_time = (time.perf_counter() - t0) * 1000
            dense_ids = [c.chunk_id for c, _ in dense_res]
            self._accumulate_metrics(results["Dense_Only"], dense_ids, ground_truth_chunks, dense_time)

            # 2. Sparse Only (BM25)
            t0 = time.perf_counter()
            sparse_res = self.bm25_index.search(query, top_k=10)
            sparse_time = (time.perf_counter() - t0) * 1000
            sparse_ids = [c.chunk_id for c, _ in sparse_res]
            self._accumulate_metrics(results["Sparse_BM25"], sparse_ids, ground_truth_chunks, sparse_time)

            # 3. Hybrid Search (RRF)
            t0 = time.perf_counter()
            hybrid_res = self.hybrid_searcher.fuse(dense_res, sparse_res, top_k=10)
            hybrid_time = dense_time + sparse_time + (time.perf_counter() - t0) * 1000
            hybrid_ids = [c.chunk_id for c, _ in hybrid_res]
            self._accumulate_metrics(results["Hybrid_RRF"], hybrid_ids, ground_truth_chunks, hybrid_time)

            # 4. Hybrid + Reranker
            t0 = time.perf_counter()
            rerank_res = self.reranker.rerank(query, hybrid_res, top_k=5)
            rerank_time = hybrid_time + (time.perf_counter() - t0) * 1000
            rerank_ids = [c.chunk_id for c, _ in rerank_res]
            self._accumulate_metrics(results["Hybrid_Plus_Rerank"], rerank_ids, ground_truth_chunks, rerank_time)

        # Average results
        for strat in strategies:
            for metric in ["hit_rate_at_3", "hit_rate_at_5", "mrr", "precision_at_5", "avg_latency_ms"]:
                results[strat][metric] = round(results[strat][metric] / num_queries, 4)

        return results

    def _accumulate_metrics(self, strat_dict: Dict[str, float], retrieved_ids: List[str], ground_truth: set, latency_ms: float):
        strat_dict["hit_rate_at_3"] += RAGEvaluationMetrics.hit_rate_at_k(retrieved_ids, ground_truth, k=3)
        strat_dict["hit_rate_at_5"] += RAGEvaluationMetrics.hit_rate_at_k(retrieved_ids, ground_truth, k=5)
        strat_dict["mrr"] += RAGEvaluationMetrics.reciprocal_rank(retrieved_ids, ground_truth)
        strat_dict["precision_at_5"] += RAGEvaluationMetrics.context_precision(retrieved_ids, ground_truth, k=5)
        strat_dict["avg_latency_ms"] += latency_ms

    def format_markdown_table(self, benchmark_results: Dict[str, Any]) -> str:
        lines = [
            "| Retrieval Strategy | Hit Rate @ 3 | Hit Rate @ 5 | MRR | Precision @ 5 | Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ]
        for strat, m in benchmark_results.items():
            lines.append(
                f"| **{strat.replace('_', ' ')}** | {m['hit_rate_at_3']:.2%} | {m['hit_rate_at_5']:.2%} | {m['mrr']:.4f} | {m['precision_at_5']:.4f} | {m['avg_latency_ms']:.2f} ms |"
            )
        return "\n".join(lines)
