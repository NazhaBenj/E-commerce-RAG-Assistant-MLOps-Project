
from pathlib import Path

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


class Retriever:
    """Retrieve relevant Jumia passages from a configurable ChromaDB index."""

    def __init__(
        self,
        top_k: int = 3,
        persist_directory: str | None = None,
        collection_name: str = "jumia_documents",
    ):
        root = Path(__file__).resolve().parents[2]

        self.top_k = top_k
        self.embedding_model = EmbeddingModel()

        index_dir = (
            Path(persist_directory)
            if persist_directory
            else root / "indexes" / "chroma_ecommerce"
        )

        self.vector_store = VectorStore(
            persist_directory=str(index_dir),
            collection_name=collection_name,
        )

    def retrieve(self, question: str) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        query_embedding = self.embedding_model.embed_query(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        retrieved_chunks = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            })

        return retrieved_chunks
