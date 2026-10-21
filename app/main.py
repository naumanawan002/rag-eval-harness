from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from app.config import settings
from app.core.chunking import TextChunker, Chunk
from app.core.embeddings import EmbeddingEngine
from app.core.bm25 import BM25Index
from app.core.hybrid import HybridSearcher
from app.core.reranker import CrossEncoderReranker
from app.core.generator import GroundedGenerator
from app.storage.vector_store import InMemoryVectorStore
from app.evaluation.harness import EvaluationHarness

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade hybrid RAG retrieval pipeline with cross-encoder reranking and evaluation harness",
    version="0.1.0"
)

# Global In-Memory RAG Pipeline State
chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
embeddings = EmbeddingEngine(dimension=384, api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL)
vector_store = InMemoryVectorStore()
bm25_index = BM25Index()
hybrid_searcher = HybridSearcher(rrf_k=settings.RRF_K)
reranker = CrossEncoderReranker()
generator = GroundedGenerator(api_key=settings.OPENAI_API_KEY)
all_chunks: List[Chunk] = []

class DocumentIngestRequest(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str
    strategy: Optional[str] = "hybrid_rerank"  # dense, sparse, hybrid, hybrid_rerank
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    strategy_used: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]

class EvaluationItem(BaseModel):
    query: str
    expected_chunk_ids: List[str]

@app.get("/healthz")
async def healthz():
    return {
        "status": "healthy",
        "indexed_chunks": len(all_chunks),
        "vector_count": len(vector_store.vectors)
    }

@app.post("/ingest")
async def ingest_document(req: DocumentIngestRequest):
    chunks = chunker.split_text(req.text, req.doc_id, req.metadata)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document text was empty or produced no chunks")

    all_chunks.extend(chunks)

    # Re-index BM25
    bm25_index.index(all_chunks)

    # Embed and upsert into vector store
    for c in chunks:
        vec = embeddings.embed_text(c.text)
        vector_store.upsert(c, vec)

    return {
        "status": "success",
        "doc_id": req.doc_id,
        "chunks_created": len(chunks),
        "total_corpus_chunks": len(all_chunks)
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    if not all_chunks:
        raise HTTPException(status_code=400, detail="Corpus is empty. Please ingest documents first.")

    strategy = req.strategy.lower()

    if strategy == "dense":
        q_vec = embeddings.embed_text(req.query)
        candidates = vector_store.search(q_vec, top_k=req.top_k)
    elif strategy == "sparse":
        candidates = bm25_index.search(req.query, top_k=req.top_k)
    elif strategy == "hybrid":
        q_vec = embeddings.embed_text(req.query)
        dense_res = vector_store.search(q_vec, top_k=settings.DENSE_TOP_K)
        sparse_res = bm25_index.search(req.query, top_k=settings.SPARSE_TOP_K)
        candidates = hybrid_searcher.fuse(dense_res, sparse_res, top_k=req.top_k)
    else:  # default "hybrid_rerank"
        q_vec = embeddings.embed_text(req.query)
        dense_res = vector_store.search(q_vec, top_k=settings.DENSE_TOP_K)
        sparse_res = bm25_index.search(req.query, top_k=settings.SPARSE_TOP_K)
        fused = hybrid_searcher.fuse(dense_res, sparse_res, top_k=settings.DENSE_TOP_K)
        candidates = reranker.rerank(req.query, fused, top_k=req.top_k)

    # Generate answer with citations
    synthesis = generator.generate(req.query, candidates)

    return QueryResponse(
        query=req.query,
        strategy_used=strategy,
        answer=synthesis["answer"],
        citations=synthesis["citations"],
        retrieved_chunks=[
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "score": round(score, 4),
                "text": c.text
            }
            for c, score in candidates
        ]
    )

@app.post("/evaluate")
async def run_evaluation(dataset: Optional[List[EvaluationItem]] = None):
    if not all_chunks:
        raise HTTPException(status_code=400, detail="Corpus is empty. Ingest sample documents first.")

    # Load default evaluation dataset if none provided
    if not dataset:
        try:
            with open("data/eval_gold_standard.json", "r") as f:
                raw_data = json.load(f)
                dataset = [EvaluationItem(**item) for item in raw_data]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load gold standard evaluation dataset: {e}")

    harness = EvaluationHarness(
        chunks=all_chunks,
        embedding_engine=embeddings,
        vector_store=vector_store,
        bm25_index=bm25_index,
        hybrid_searcher=hybrid_searcher,
        reranker=reranker
    )

    eval_dicts = [d.model_dump() for d in dataset]
    benchmark_metrics = harness.run_benchmark(eval_dicts)
    markdown_table = harness.format_markdown_table(benchmark_metrics)

    return {
        "queries_evaluated": len(dataset),
        "metrics": benchmark_metrics,
        "markdown_table": markdown_table
    }
