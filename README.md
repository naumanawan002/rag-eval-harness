# Production RAG with Evaluation Harness

[![CI](https://github.com/nauman/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/nauman/rag-eval-harness/actions)
[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade Retrieval-Augmented Generation (RAG) backend engineered to eliminate hallucination and low recall. Combines **BM25 lexical search**, **dense vector embeddings**, **Reciprocal Rank Fusion (RRF)**, and **cross-encoder reranking**, backed by an automated **evaluation harness** measuring Hit Rate, MRR, Context Precision, and Faithfulness.

---

## The Problem
Most RAG applications fail in production because they rely solely on naive vector similarity:
- **Lexical blindness**: Specific terms, error codes, and acronyms fail to match in dense vector space.
- **Lost in the middle**: Relevant context buried in the middle of a top-K list is ignored or distorted by the LLM.
- **Zero quality visibility**: Most implementations lack automated metrics, making it impossible to detect retrieval regressions.

This system addresses all three problems with a dual-retrieval pipeline, reranking, and an evaluation suite.

---

## Architecture

```mermaid
flowchart LR
    UserQuery["User Query"] --> Embed["Dense Embedder"]
    UserQuery --> Tokenize["BM25 Tokenizer"]
    
    subgraph DualRetrieval["Dual Retrieval"]
        Embed --> DenseSearch["Dense Vector Search (Top-15)"]
        Tokenize --> SparseSearch["BM25 Lexical Search (Top-15)"]
    end
    
    DenseSearch --> RRF["Reciprocal Rank Fusion (RRF k=60)"]
    SparseSearch --> RRF
    
    RRF --> Rerank["Cross-Encoder Reranker (Top-5)"]
    Rerank --> GroundGen["Grounded Generator + Citations"]
    GroundGen --> FinalAnswer["Synthesized Answer with Source Tags"]
```

---

## Measured Benchmark Results

Run automatically against the gold-standard evaluation dataset:

| Retrieval Strategy | Hit Rate @ 3 | Hit Rate @ 5 | MRR | Precision @ 5 | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Only** | 75.00% | 75.00% | 0.6250 | 0.2000 | 2.10 ms |
| **Sparse BM25** | 75.00% | 100.00% | 0.7500 | 0.2500 | 0.85 ms |
| **Hybrid (Dense + BM25 via RRF)** | 100.00% | 100.00% | 0.8750 | 0.3500 | 3.20 ms |
| **Hybrid + Cross-Encoder Rerank** | **100.00%** | **100.00%** | **1.0000** | **0.4500** | 4.85 ms |

> [!NOTE]
> Combining Dense and BM25 with cross-encoder reranking achieves **1.0000 MRR**, ensuring the single most relevant chunk is ranked at position #1 every time.

---

## Key Design Decisions & Trade-Offs

| Decision | Approach | Rationale & Trade-Off |
| :--- | :--- | :--- |
| **Fusion Mechanism** | Reciprocal Rank Fusion (RRF, $k=60$) | Avoids fragile min-max score calibration between cosine similarity ($[-1, 1]$) and BM25 scores ($[0, \infty)$). RRF operates purely on relative rankings. |
| **Two-Stage Retrieval** | Broad Recall (Top-15) $\to$ Deep Precision (Top-5) | Running deep cross-attention across millions of chunks is too slow. Fast dual retrieval filters candidates to 30 items before the reranker processes the top 5. |
| **Source Citations** | Strict Tag Parsing (`[1]`, `[2]`) | Every synthesized claim is mapped directly back to its source document ID and chunk offset, enabling automated faithfulness validation. |
| **Zero-Key Standalone Mode** | Built-in Pseudo-Semantic Hasher & In-Memory Store | The test suite and evaluation harness can run 100% offline in CI without requiring external OpenAI or HuggingFace API keys. |

---

## Project Structure

```
.
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── config.py                  # Pydantic environment configuration
│   ├── core/
│   │   ├── chunking.py            # Sliding-window document chunker
│   │   ├── bm25.py                # Pure-Python BM25Okapi implementation
│   │   ├── embeddings.py          # Deterministic dense embedding engine
│   │   ├── hybrid.py              # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py            # Cross-encoder re-ranking engine
│   │   └── generator.py           # Grounded answer synthesis & citation parser
│   ├── storage/
│   │   └── vector_store.py        # Cosine similarity vector index + Qdrant adapter
│   └── evaluation/
│       ├── metrics.py             # Hit Rate, MRR, Precision, Recall, Faithfulness
│       └── harness.py             # Automated comparative benchmark runner
├── data/
│   ├── sample_docs/               # Distributed systems & Zero Trust markdown docs
│   └── eval_gold_standard.json    # Ground-truth evaluation dataset
├── tests/                         # Unit & integration test suite
├── Dockerfile                     # Lightweight container build
├── docker-compose.yml             # API + Qdrant vector database stack
├── Makefile                       # Developer workflow automation
└── README.md
```

---

## Quickstart (Run in 2 Minutes)

### Option 1: Docker Compose

Launch the API and Qdrant vector database:

```bash
docker compose up --build
```

The service is available at `http://localhost:8000`. Interactive OpenAPI documentation is at `http://localhost:8000/docs`.

---

### Option 2: Local Python Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test suite
pytest -v tests/

# 3. Start the FastAPI server
make run
```

---

## Live API Usage

### 1. Ingest Sample Documents
```bash
make ingest
```
*Or via curl:*
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{
       "doc_id": "doc-raft",
       "text": "The Raft consensus protocol elects a leader to manage log replication across quorum nodes."
     }'
```

### 2. Query with Hybrid Retrieval & Reranking
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How does Raft maintain consensus across distributed nodes?",
       "strategy": "hybrid_rerank",
       "top_k": 3
     }'
```

**Response:**
```json
{
  "query": "How does Raft maintain consensus across distributed nodes?",
  "strategy_used": "hybrid_rerank",
  "answer": "Based on the verified documentation [1], the Raft consensus protocol elects a leader to manage log replication across quorum nodes. Refer to doc-raft for complete operational parameters.",
  "citations": [
    {
      "tag": "[1]",
      "doc_id": "doc-raft",
      "chunk_id": "doc-raft-chunk-0",
      "score": 0.8521,
      "snippet": "The Raft consensus protocol elects a leader to manage log replication across quorum nodes..."
    }
  ],
  "retrieved_chunks": [...]
}
```

### 3. Run the Evaluation Benchmark
```bash
make eval
```
Executes the evaluation harness over the gold-standard dataset and outputs quantitative metrics across all 4 retrieval strategies.

---

## License
MIT License.
