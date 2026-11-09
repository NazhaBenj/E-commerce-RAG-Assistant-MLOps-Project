
import json
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports" / "evaluation"

TRACKING_URI = "http://127.0.0.1:5000"

RETRIEVAL_EXPERIMENT = "Ecommerce-RAG-Chunking-Experiments"
GENERATION_EXPERIMENT = "Ecommerce-RAG-Generation-Evaluation"


# ============================================================
# CONFIGURATIONS DE CHUNKING
# ============================================================

CHUNKING_CONFIGS = {
    "chunk_300_50": {
        "chunk_size": 300,
        "chunk_overlap": 50,
        "report_file": "chunk_300_50_results.json",
    },
    "chunk_500_100": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "report_file": "chunk_500_100_results.json",
    },
}


# ============================================================
# UTILITAIRES
# ============================================================

def load_report(filename):
    report_path = REPORT_DIR / filename

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Rapport introuvable : {report_path}. "
            "Lance d'abord l'évaluation correspondante."
        )

    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def log_report_artifacts(filename):
    report_path = REPORT_DIR / filename

    mlflow.log_artifact(str(report_path))

    csv_path = report_path.with_suffix(".csv")

    if csv_path.is_file():
        mlflow.log_artifact(str(csv_path))


def extract_metrics(summary, metric_names):
    """Extract available numeric metrics from an evaluation summary."""

    metrics = {}

    for name in metric_names:
        value = summary.get(name)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[name] = float(value)

    return metrics


# ============================================================
# 1. TRACKING DES EXPÉRIENCES DE RETRIEVAL
# ============================================================

def log_retrieval_configuration(name, config):

    report = load_report(config["report_file"])
    summary = report["summary"]

    with mlflow.start_run(run_name=name):

        # Parameters
        mlflow.log_params({
            "experiment_type": "chunking_comparison",
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "chunking_strategy": "sentence_aware",
            "embedding_model": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "retriever": "HybridRetriever",
            "retrieval_method": "Vector Search + BM25 + RRF",
            "top_k": 5,
            "number_of_documents": 8,
            "number_of_questions": summary["total_questions"],
            "evaluated_questions": summary["evaluated_questions"],
        })

        # Metrics
        metrics = extract_metrics(
            summary,
            [
                "total_chunks",
                "recall_at_5",
                "precision_at_5",
                "mrr",
            ],
        )

        if not metrics:
            raise ValueError(
                f"Aucune métrique de retrieval disponible pour {name}."
            )

        mlflow.log_metrics(metrics)

        # Artifacts
        log_report_artifacts(config["report_file"])

    print(f"[OK] Configuration retrieval enregistrée : {name}")


def track_retrieval_experiments():

    mlflow.set_experiment(RETRIEVAL_EXPERIMENT)

    print("\n========== RETRIEVAL TRACKING ==========")
    print(f"Expérience : {RETRIEVAL_EXPERIMENT}")

    for name, config in CHUNKING_CONFIGS.items():
        log_retrieval_configuration(name, config)


# ============================================================
# 2. TRACKING DE L'ÉVALUATION DE GÉNÉRATION
# ============================================================

def track_generation_evaluation():

    filename = "generation_results.json"

    report = load_report(filename)
    summary = report["summary"]

    with mlflow.start_run(
        run_name="generation_evaluation"
    ):

        # Parameters
        mlflow.log_params({
            "experiment_type": "generation_evaluation",
            "llm": "llama3.2:latest",
            "llm_provider": "Ollama",
            "embedding_model": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "retriever": "HybridRetriever",
            "top_k": 5,
            "temperature": 0.0,
            "number_of_documents": 8,
            "number_of_questions": summary["total_questions"],
            "answered_questions": summary["answered_questions"],
            "prompt_version": "current_system_prompt",
        })

        # Metrics
        metrics = extract_metrics(
            summary,
            [
                "status_accuracy",
                "faithfulness_proxy",
                "answer_relevance_proxy",
                "citation_rate",
            ],
        )

        if not metrics:
            raise ValueError(
                "Aucune métrique de génération disponible."
            )

        mlflow.log_metrics(metrics)

        # Artifacts
        log_report_artifacts(filename)

    print("[OK] Évaluation de génération enregistrée.")


# ============================================================
# MAIN
# ============================================================

def main():

    mlflow.set_tracking_uri(TRACKING_URI)

    print("========================================")
    print("       MLFLOW - E-COMMERCE RAG")
    print("========================================")

    # Retrieval experiments
    track_retrieval_experiments()

    # Generation evaluation
    mlflow.set_experiment(GENERATION_EXPERIMENT)

    print("\n========== GENERATION TRACKING ==========")
    print(f"Expérience : {GENERATION_EXPERIMENT}")

    track_generation_evaluation()

    print("\n========================================")
    print("Toutes les expériences ont été enregistrées.")
    print("Consulte MLflow pour visualiser les résultats.")
    print("========================================")


if __name__ == "__main__":
    main()
