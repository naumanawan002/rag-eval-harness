import math
import re
from typing import List, Dict, Tuple
from app.core.chunking import Chunk

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # chunk_id -> {term -> count}
        self.chunks_map: Dict[str, Chunk] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric words
        return re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())

    def index(self, chunks: List[Chunk]):
        self.corpus_size = len(chunks)
        total_len = 0

        for chunk in chunks:
            self.chunks_map[chunk.chunk_id] = chunk
            tokens = self._tokenize(chunk.text)
            doc_len = len(tokens)
            self.doc_lengths[chunk.chunk_id] = doc_len
            total_len += doc_len

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs[chunk.chunk_id] = tf

            # Unique terms in this chunk for DF calculation
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = {}

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # Robertson-Spärck Jones IDF formula
            df = self.doc_freqs[token]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for chunk_id, tf_map in self.term_freqs.items():
                f = tf_map.get(token, 0)
                if f == 0:
                    continue

                doc_len = self.doc_lengths[chunk_id]
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (f * (self.k1 + 1.0)) / denom
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks_map[cid], score) for cid, score in ranked]
