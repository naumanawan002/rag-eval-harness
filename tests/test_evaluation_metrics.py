from app.evaluation.metrics import RAGEvaluationMetrics

def test_hit_rate_at_k():
    retrieved = ["c1", "c2", "c3", "c4"]
    ground_truth = {"c3"}

    assert RAGEvaluationMetrics.hit_rate_at_k(retrieved, ground_truth, k=3) == 1.0
    assert RAGEvaluationMetrics.hit_rate_at_k(retrieved, ground_truth, k=2) == 0.0

def test_reciprocal_rank():
    retrieved = ["c1", "c2", "c3"]
    # First relevant chunk is at rank 2
    ground_truth = {"c2", "c3"}
    assert RAGEvaluationMetrics.reciprocal_rank(retrieved, ground_truth) == 0.5

def test_context_precision():
    retrieved = ["c1", "c2", "c3", "c4"]
    ground_truth = {"c1", "c3"}
    # 2 out of 4 retrieved chunks are relevant
    assert RAGEvaluationMetrics.context_precision(retrieved, ground_truth, k=4) == 0.5

def test_faithfulness_score():
    context = "Raft uses an elected leader to manage replicated logs across quorum nodes."
    answer = "Raft uses an elected leader."
    score = RAGEvaluationMetrics.faithfulness_score(answer, context)
    assert score == 1.0  # All words in the answer exist in the context
