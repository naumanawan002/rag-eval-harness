from app.core.chunking import TextChunker

def test_text_chunker_basic():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    sample_text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12 Word13"
    chunks = chunker.split_text(sample_text, doc_id="test-doc")

    assert len(chunks) >= 2
    assert chunks[0].doc_id == "test-doc"
    assert chunks[0].index == 0
    assert chunks[0].chunk_id == "test-doc-chunk-0"
    # Verify overlap exists between chunk 0 and chunk 1
    chunk0_words = set(chunks[0].text.split())
    chunk1_words = set(chunks[1].text.split())
    assert len(chunk0_words.intersection(chunk1_words)) > 0

def test_text_chunker_empty():
    chunker = TextChunker()
    chunks = chunker.split_text("", doc_id="empty-doc")
    assert chunks == []
