from typing import List, Set

class RAGEvaluationMetrics:
    @staticmethod
    def hit_rate_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str], k: int) -> float:
        """
        Returns 1.0 if any ground-truth chunk appears in the top-k retrieved chunks, else 0.0.
        """
        top_k = retrieved_chunk_ids[:k]
        for cid in top_k:
            if cid in ground_truth_chunk_ids:
                return 1.0
        return 0.0

    @staticmethod
    def reciprocal_rank(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str]) -> float:
        """
        Returns 1 / rank of the first relevant chunk retrieved.
        """
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in ground_truth_chunk_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def context_precision(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str], k: int) -> float:
        """
        Fraction of retrieved top-k chunks that are relevant.
        """
        top_k = retrieved_chunk_ids[:k]
        if not top_k:
            return 0.0
        relevant_count = sum(1 for cid in top_k if cid in ground_truth_chunk_ids)
        return relevant_count / len(top_k)

    @staticmethod
    def context_recall(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str]) -> float:
        """
        Fraction of ground-truth chunks successfully retrieved.
        """
        if not ground_truth_chunk_ids:
            return 1.0
        found = sum(1 for cid in ground_truth_chunk_ids if cid in retrieved_chunk_ids)
        return found / len(ground_truth_chunk_ids)

    @staticmethod
    def faithfulness_score(answer: str, context: str) -> float:
        """
        Calculates lexical faithfulness (hallucination defense)
        verifying that claim statements in the answer are supported by retrieved context.
        """
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return 1.0

        supported = answer_words.intersection(context_words)
        return round(len(supported) / len(answer_words), 4)
