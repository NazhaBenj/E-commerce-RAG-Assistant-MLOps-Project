
import argparse
import json
from pathlib import Path

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_INDEX = ROOT / "indexes" / "chroma_ecommerce"

COLLECTION_NAME = "jumia_documents"


def load_chunks(chunks_file: Path):
    if not chunks_file.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {chunks_file}"
        )

    with chunks_file.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("Aucun chunk trouvé.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Des identifiants de chunks sont dupliqués.")

    return chunks


def build_index(chunks_file: Path, index_dir: Path):
    print("\nChargement des chunks...")
    chunks = load_chunks(chunks_file)
    print(f"{len(chunks)} chunks chargés.")

    print("\nChargement du modèle d'embeddings...")
    embedding_model = EmbeddingModel()

    print("\nGénération des embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Le nombre d'embeddings ne correspond pas "
            "au nombre de chunks."
        )

    print(f"{len(embeddings)} embeddings générés.")

    print("\nConnexion à ChromaDB...")
    vector_store = VectorStore(
        persist_directory=str(index_dir),
        collection_name=COLLECTION_NAME,
    )

    print("\nRéinitialisation de la collection expérimentale...")
    vector_store.reset_collection()

    print("\nEnregistrement des vecteurs...")
    vector_store.add_documents(chunks, embeddings)

    indexed_count = vector_store.count()

    print("\n" + "=" * 45)
    print("RÉSULTAT DE L'INDEXATION")
    print("=" * 45)
    print(f"Chunks chargés      : {len(chunks)}")
    print(f"Embeddings générés  : {len(embeddings)}")
    print(f"Vecteurs indexés    : {indexed_count}")
    print(f"Fichier source      : {chunks_file}")
    print(f"Index               : {index_dir}")

    if indexed_count != len(chunks):
        raise RuntimeError(
            "Le nombre de vecteurs indexés est incorrect."
        )

    print("\nIndexation terminée avec succès !")


def main():
    parser = argparse.ArgumentParser(
        description="Construction d'un index ChromaDB."
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
        help="Fichier JSONL contenant les chunks.",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Répertoire de persistance ChromaDB.",
    )

    args = parser.parse_args()

    chunks_file = (
        args.chunks
        if args.chunks.is_absolute()
        else ROOT / args.chunks
    )

    index_dir = (
        args.index
        if args.index.is_absolute()
        else ROOT / args.index
    )

    build_index(chunks_file, index_dir)


if __name__ == "__main__":
    main()
