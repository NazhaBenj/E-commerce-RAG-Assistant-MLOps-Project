
import argparse
import csv
import json
from pathlib import Path

from src.rag.chunking import create_document_chunks

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data" / "source_manifest.csv"
TEXT_DIR = ROOT / "data" / "processed" / "source_text"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

REQUIRED_COLUMNS = {
    "document_id",
    "category",
    "topic",
    "source_url",
    "language",
    "collection_date",
}


def main(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, output=DEFAULT_OUTPUT):

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifeste introuvable : {MANIFEST}")

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans le manifeste : {missing_columns}"
            )

        documents = list(reader)

    if not documents:
        raise ValueError("Le manifeste ne contient aucun document.")

    all_chunks = []
    processed_documents = set()

    for doc in documents:
        document_id = doc["document_id"].strip()

        if not document_id:
            print("[SKIP] Document sans identifiant.")
            continue

        if document_id in processed_documents:
            raise ValueError(f"Document dupliqué dans le manifeste : {document_id}")

        processed_documents.add(document_id)

        text_path = TEXT_DIR / f"{document_id}.txt"

        if not text_path.is_file():
            print(f"[SKIP] Texte introuvable : {text_path}")
            continue

        text = text_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"[SKIP] Texte vide : {document_id}")
            continue

        chunks = create_document_chunks(
            text=text,
            document_id=document_id,
            category=doc["category"].strip(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in chunks:
            chunk.update({
                "topic": doc["topic"].strip(),
                "source_url": doc["source_url"].strip(),
                "language": doc["language"].strip(),
                "collection_date": doc["collection_date"].strip(),
            })

        all_chunks.extend(chunks)

        print(f"[OK] {document_id} : {len(chunks)} chunks générés")

    if not all_chunks:
        raise ValueError("Aucun chunk généré. Vérifie les documents sources.")

    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Des identifiants de chunks sont dupliqués.")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("GÉNÉRATION DES CHUNKS TERMINÉE")
    print("=" * 50)
    print(f"Taille des chunks : {chunk_size} caractères")
    print(f"Overlap : {chunk_overlap} caractères")
    print(f"Documents référencés : {len(processed_documents)}")
    print(f"Total des chunks : {len(all_chunks)}")
    print(f"Taille maximale : {max(len(c['text']) for c in all_chunks)} caractères")
    print(f"Fichier créé : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération des chunks du corpus.")

    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    main(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output=args.output,
    )
