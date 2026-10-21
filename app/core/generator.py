from typing import List, Tuple, Dict, Any
from app.core.chunking import Chunk

class GroundedGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def generate(self, query: str, context_chunks: List[Tuple[Chunk, float]]) -> Dict[str, Any]:
        """
        Synthesizes a grounded answer citing source documents.
        """
        if not context_chunks:
            return {
                "answer": "I do not have sufficient information in the provided documentation to answer this question.",
                "citations": [],
                "grounded": False
            }

        citations = []
        context_snippets = []
        for idx, (chunk, score) in enumerate(context_chunks, start=1):
            citation_tag = f"[{idx}]"
            citations.append({
                "tag": citation_tag,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
                "snippet": chunk.text[:120] + "..."
            })
            context_snippets.append(f"{citation_tag} (Source: {chunk.doc_id}): {chunk.text}")

        # Deterministic grounded synthesis format
        joined_context = "\n\n".join(context_snippets)
        top_chunk = context_chunks[0][0]

        answer = (
            f"Based on the verified documentation [1], {top_chunk.text.split('.')[0]}. "
            f"Refer to {top_chunk.doc_id} for complete operational parameters."
        )

        return {
            "answer": answer,
            "citations": citations,
            "grounded": True,
            "context_used": joined_context
        }
