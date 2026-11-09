
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.rag.retriever import Retriever


class HybridRetriever:
    """Combine vector search and BM25 using Reciprocal Rank Fusion."""

    def __init__(
        self,
        top_k: int = 3,
        chunks_file: str | None = None,
        persist_directory: str | None = None,
        collection_name: str = "jumia_documents",
    ):
        if top_k <= 0:
            raise ValueError("top_k doit être positif.")

        root = Path(__file__).resolve().parents[2]

        self.top_k = top_k
        self.candidate_k = max(top_k * 3, 10)
        self.rrf_k = 60

        # Vector retrieval
        self.vector_retriever = Retriever(
            top_k=self.candidate_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        # Load chunks used by BM25
        chunks_path = (
            Path(chunks_file)
            if chunks_file
            else root / "data" / "processed" / "chunks.jsonl"
        )

        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"Fichier chunks introuvable : {chunks_path}"
            )

        self.chunks = []

        with chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.chunks.append(json.loads(line))

        if not self.chunks:
            raise ValueError("Aucun chunk disponible pour BM25.")

        self.chunk_by_id = {
            chunk["chunk_id"]: chunk for chunk in self.chunks
        }

        if len(self.chunk_by_id) != len(self.chunks):
            raise ValueError("Des identifiants de chunks sont dupliqués.")

        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, question: str) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        vector_results = self.vector_retriever.retrieve(question)

        query_tokens = self._tokenize(question)
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_indices = [
            index
            for index, score in enumerate(bm25_scores)
            if score > 0
        ]

        bm25_indices.sort(
            key=lambda index: bm25_scores[index],
            reverse=True,
        )

        bm25_indices = bm25_indices[:self.candidate_k]

        rrf_scores = {}
        retrieved = {}

        # Vector ranking
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1 / (self.rrf_k + rank)
            )

            retrieved[chunk_id] = result

        # BM25 ranking
        for rank, index in enumerate(bm25_indices, start=1):
            chunk = self.chunks[index]
            chunk_id = chunk["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1 / (self.rrf_k + rank)
            )

            if chunk_id not in retrieved:
                metadata = {
                    key: value
                    for key, value in chunk.items()
                    if key not in ("chunk_id", "text")
                }

                retrieved[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": metadata,
                    "distance": None,
                }

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        results = []

        for chunk_id in ranked_ids[:self.top_k]:
            result = retrieved[chunk_id].copy()
            result["score"] = rrf_scores[chunk_id]
            results.append(result)

        return results
