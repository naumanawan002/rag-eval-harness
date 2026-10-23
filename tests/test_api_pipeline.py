from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_full_pipeline():
    # 1. Health check
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # 2. Ingest document
    doc_payload = {
        "doc_id": "test-doc-auth",
        "text": "OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts.",
        "metadata": {"category": "security"}
    }
    ingest_res = client.post("/ingest", json=doc_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"
    assert ingest_res.json()["chunks_created"] > 0

    # 3. Query RAG
    query_payload = {
        "query": "What is OAuth 2.0 used for?",
        "strategy": "hybrid_rerank",
        "top_k": 3
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert "OAuth" in data["answer"] or "authorization" in data["answer"]
    assert len(data["citations"]) > 0
    assert len(data["retrieved_chunks"]) > 0

    # 4. Evaluate harness
    eval_payload = [
        {
            "query": "OAuth 2.0 authorization framework",
            "expected_chunk_ids": ["test-doc-auth-chunk-0"]
        }
    ]
    eval_res = client.post("/evaluate", json=eval_payload)
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert eval_data["queries_evaluated"] == 1
    assert "metrics" in eval_data
    assert "markdown_table" in eval_data
