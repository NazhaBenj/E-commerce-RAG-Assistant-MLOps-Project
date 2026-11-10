
import chromadb
from pathlib import Path


class VectorStore:
    """Manage the persistent ChromaDB vector collection."""

    def __init__(
        self,
        persist_directory: str = "indexes/chroma_ecommerce",
        collection_name: str = "jumia_documents",
    ):
        self.persist_directory = str(Path(persist_directory))
        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        """Delete and recreate the collection to avoid obsolete vectors."""

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Le nombre de chunks doit correspondre au nombre d'embeddings."
            )

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    key: value
                    for key, value in chunk.items()
                    if key not in ["chunk_id", "text"]
                }
                for chunk in chunks
            ],
    )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> dict:
        count = self.collection.count()

        if count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )

    def count(self) -> int:
        return self.collection.count()
