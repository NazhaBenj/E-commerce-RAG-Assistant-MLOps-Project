import csv
from pathlib import Path

from src.ingestion.html_cleaner import clean_html

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "source_manifest.csv"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed" / "source_text"


def process_corpus():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        documents = list(csv.DictReader(file))

    for doc in documents:
        source_path = RAW_DIR / doc["filename"]

        if not source_path.is_file():
            print(f"[ERREUR] Fichier introuvable : {source_path}")
            continue

        text = clean_html(source_path)

        output_path = OUTPUT_DIR / f"{doc['document_id']}.txt"
        output_path.write_text(text, encoding="utf-8")

        print(
            f"[OK] {doc['document_id']} | "
            f"{len(text)} caractères | {output_path.name}"
        )

    print(f"\nTraitement terminé : {OUTPUT_DIR}")


if __name__ == "__main__":
    process_corpus()
