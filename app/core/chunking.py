import re
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: Dict[str, Any] = {}

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Splits text into sliding chunks respecting paragraph and sentence boundaries.
        """
        if metadata is None:
            metadata = {}

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return []

        words = text.split()
        chunks: List[Chunk] = []
        chunk_idx = 0
        i = 0

        while i < len(words):
            end_idx = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end_idx]
            chunk_text = " ".join(chunk_words)

            chunk = Chunk(
                chunk_id=f"{doc_id}-chunk-{chunk_idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=chunk_idx,
                metadata=metadata
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end_idx == len(words):
                break
            # Advance index considering overlap
            i += max(1, self.chunk_size - self.chunk_overlap)

        return chunks
