from app.core.chunking import Chunk
from app.core.bm25 import BM25Index
from app.core.hybrid import HybridSearcher

def test_bm25_search():
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="Python is a dynamic programming language used in machine learning.", index=0)
    c2 = Chunk(chunk_id="c2", doc_id="d2", text="Go is a compiled systems programming language with lightweight goroutines.", index=0)
    c3 = Chunk(chunk_id="c3", doc_id="d3", text="PostgreSQL is an open-source relational database supporting JSON and vectors.", index=0)

    index = BM25Index()
    index.index([c1, c2, c3])

    results = index.search("goroutines and compiled systems", top_k=2)
    assert len(results) > 0
    # c2 must rank highest for goroutines
    assert results[0][0].chunk_id == "c2"

def test_hybrid_rrf_fusion():
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="Chunk 1", index=0)
    c2 = Chunk(chunk_id="c2", doc_id="d2", text="Chunk 2", index=0)
    c3 = Chunk(chunk_id="c3", doc_id="d3", text="Chunk 3", index=0)

    # Dense ranked: c1, c2, c3
    dense_res = [(c1, 0.95), (c2, 0.85), (c3, 0.75)]
    # Sparse ranked: c2, c1, c3
    sparse_res = [(c2, 12.5), (c1, 8.2), (c3, 2.1)]

    searcher = HybridSearcher(rrf_k=60)
    fused = searcher.fuse(dense_res, sparse_res, top_k=3)

    assert len(fused) == 3
    # Both c1 and c2 have ranks 1 and 2, so their RRF scores are equal and highest
    top_ids = {fused[0][0].chunk_id, fused[1][0].chunk_id}
    assert "c1" in top_ids
    assert "c2" in top_ids
