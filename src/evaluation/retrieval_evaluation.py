import json
import csv
from pathlib import Path

import yaml

from src.rag.hybrid_retriever import HybridRetriever


# ============================================================
# CONFIGURATION GENERALE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = ROOT / "data" / "evaluation" / "questions.yaml"
REPORT_DIR = ROOT / "reports" / "evaluation"

TOP_K = 5


# ============================================================
# CONFIGURATIONS A/B
# ============================================================

CONFIGURATIONS = {
    "chunk_300_50": {
        "chunk_size": 300,
        "chunk_overlap": 50,

        "chunks_file": (
            ROOT / "data" / "processed" /
            "experiments" / "chunks_300_50.jsonl"
        ),

        "index_dir": (
            ROOT / "indexes" / "experiments" /
            "chroma_300_50"
        ),

        # Annotations de la configuration A
        "relevant_chunks": {
            "Q001": [
                "JUMIA-008-CHUNK-0020",
                "JUMIA-008-CHUNK-0021"
            ],

            "Q002": [
                "JUMIA-004-CHUNK-0003"
            ],

            "Q003": [
                "JUMIA-005-CHUNK-0028"
            ]
        }
    },

    "chunk_500_100": {
        "chunk_size": 500,
        "chunk_overlap": 100,

        "chunks_file": (
            ROOT / "data" / "processed" /
            "experiments" / "chunks_500_100.jsonl"
        ),

        "index_dir": (
            ROOT / "indexes" / "experiments" /
            "chroma_500_100"
        ),

        # Annotations de la configuration B
        "relevant_chunks": {
            "Q001": [
                "JUMIA-008-CHUNK-0013"
            ],

            "Q002": [
                "JUMIA-004-CHUNK-0002"
            ],

            "Q003": [
                "JUMIA-005-CHUNK-0019"
            ]
        }
    }
}


# ============================================================
# CHARGEMENT DES QUESTIONS
# ============================================================

def load_questions():

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        questions = yaml.safe_load(file)

    return questions


# ============================================================
# CALCUL DES METRIQUES
# ============================================================

def calculate_metrics(retrieved_ids, relevant_ids, k=TOP_K):

    retrieved_top_k = retrieved_ids[:k]

    relevant_set = set(relevant_ids)

    relevant_found = [
        chunk_id
        for chunk_id in retrieved_top_k
        if chunk_id in relevant_set
    ]

    # Recall@K
    recall = (
        len(set(relevant_found)) / len(relevant_set)
        if relevant_set else 0.0
    )

    # Precision@K
    precision = len(relevant_found) / k

    # Mean Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_top_k, start=1):

        if chunk_id in relevant_set:
            reciprocal_rank = 1 / rank
            break

    return {
        "recall_at_5": recall,
        "precision_at_5": precision,
        "mrr": reciprocal_rank
    }


# ============================================================
# NOMBRE DE CHUNKS
# ============================================================

def count_chunks(chunks_file):

    with chunks_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        return sum(
            1 for line in file
            if line.strip()
        )


# ============================================================
# EVALUATION D'UNE CONFIGURATION
# ============================================================

def evaluate_configuration(config_name, config, questions):

    print("\n" + "=" * 55)
    print(f"ÉVALUATION : {config_name}")
    print("=" * 55)

    # Initialisation du retriever correspondant
    retriever = HybridRetriever(
        top_k=TOP_K,
        chunks_file=str(config["chunks_file"]),
        persist_directory=str(config["index_dir"])
    )

    details = []

    recalls = []
    precisions = []
    reciprocal_ranks = []

    # Parcours des questions
    for question_data in questions:

        question_id = question_data["id"]
        question_text = question_data["question"]

        expected_status = question_data["expected_status"]

        print(f"\nQuestion : {question_text}")

        # Retrieval
        retrieved_chunks = retriever.retrieve(question_text)

        # Extraction des identifiants
        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        ]

        print(f"Chunks récupérés : {retrieved_ids}")

        # Question hors périmètre
        if expected_status == "out_of_scope":

            print(
                "Métriques non applicables "
                "(aucun chunk pertinent annoté)."
            )

            details.append({
                "id": question_id,
                "question": question_text,
                "expected_status": expected_status,
                "relevant_chunks": [],
                "retrieved_chunks": retrieved_ids,
                "recall_at_5": None,
                "precision_at_5": None,
                "mrr": None
            })

            continue

        # Annotations propres à la configuration
        relevant_ids = config["relevant_chunks"].get(
            question_id,
            []
        )

        if not relevant_ids:
            raise ValueError(
                f"Aucune annotation pour {question_id} "
                f"dans {config_name}"
            )

        # Calcul des métriques
        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
            TOP_K
        )

        recall = metrics["recall_at_5"]
        precision = metrics["precision_at_5"]
        mrr = metrics["mrr"]

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(mrr)

        print(f"Recall@5 : {recall:.3f}")
        print(f"Precision@5 : {precision:.3f}")
        print(f"MRR : {mrr:.3f}")

        details.append({
            "id": question_id,
            "question": question_text,
            "expected_status": expected_status,
            "relevant_chunks": relevant_ids,
            "retrieved_chunks": retrieved_ids,
            "recall_at_5": recall,
            "precision_at_5": precision,
            "mrr": mrr
        })

    # ========================================================
    # MOYENNES
    # ========================================================

    evaluated_questions = len(recalls)

    summary = {
        "configuration": config_name,
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "total_chunks": count_chunks(config["chunks_file"]),
        "total_questions": len(questions),
        "evaluated_questions": evaluated_questions,

        "recall_at_5": (
            sum(recalls) / evaluated_questions
            if evaluated_questions else 0.0
        ),

        "precision_at_5": (
            sum(precisions) / evaluated_questions
            if evaluated_questions else 0.0
        ),

        "mrr": (
            sum(reciprocal_ranks) / evaluated_questions
            if evaluated_questions else 0.0
        )
    }

    print("\n" + "-" * 55)
    print(f"RÉSULTATS GLOBAUX : {config_name}")
    print("-" * 55)

    print(f"Chunks : {summary['total_chunks']}")
    print(f"Recall@5 : {summary['recall_at_5']:.3f}")
    print(f"Precision@5 : {summary['precision_at_5']:.3f}")
    print(f"MRR : {summary['mrr']:.3f}")

    return {
        "summary": summary,
        "details": details
    }


# ============================================================
# SAUVEGARDE DES RAPPORTS JSON ET CSV
# ============================================================

def save_reports(config_name, results):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Rapport JSON
    json_path = REPORT_DIR / f"{config_name}_results.json"

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Rapport CSV
    csv_path = REPORT_DIR / f"{config_name}_results.csv"

    fieldnames = [
        "id",
        "question",
        "expected_status",
        "relevant_chunks",
        "retrieved_chunks",
        "recall_at_5",
        "precision_at_5",
        "mrr"
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for detail in results["details"]:

            row = detail.copy()

            row["relevant_chunks"] = ", ".join(
                row["relevant_chunks"]
            )

            row["retrieved_chunks"] = ", ".join(
                row["retrieved_chunks"]
            )

            writer.writerow(row)

    print(f"\nRapport JSON : {json_path}")
    print(f"Rapport CSV : {csv_path}")


# ============================================================
# COMPARAISON FINALE A/B
# ============================================================

def print_comparison(all_results):

    print("\n")
    print("=" * 65)
    print("COMPARAISON FINALE DES CONFIGURATIONS")
    print("=" * 65)

    for config_name, results in all_results.items():

        summary = results["summary"]

        print(f"\n{config_name}")

        print(f"Chunks : {summary['total_chunks']}")

        print(
            f"Recall@5 : "
            f"{summary['recall_at_5']:.3f}"
        )

        print(
            f"Precision@5 : "
            f"{summary['precision_at_5']:.3f}"
        )

        print(
            f"MRR : "
            f"{summary['mrr']:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Évaluation comparative du retrieval RAG"
    )

    parser.add_argument(
        "--config",
        choices=[
            "all",
            "chunk_300_50",
            "chunk_500_100"
        ],
        default="all"
    )

    args = parser.parse_args()

    questions = load_questions()

    selected_configs = (
        CONFIGURATIONS
        if args.config == "all"
        else {
            args.config: CONFIGURATIONS[args.config]
        }
    )

    all_results = {}

    for config_name, config in selected_configs.items():

        results = evaluate_configuration(
            config_name,
            config,
            questions
        )

        save_reports(
            config_name,
            results
        )

        all_results[config_name] = results

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()