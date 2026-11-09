
import json
import csv
import re
import unicodedata
from pathlib import Path

import yaml

from src.rag.rag_chain import RAGChain


ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = ROOT / "data" / "evaluation" / "questions.yaml"
REPORT_DIR = ROOT / "reports" / "evaluation"

CHUNK_ID_PATTERN = re.compile(r"JUMIA-\d{3}-CHUNK-\d{4}")

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de",
    "et", "ou", "en", "au", "aux", "a", "est", "sont",
    "pour", "par", "sur", "dans", "ce", "cette", "ces",
    "qui", "que", "quoi", "ne", "pas", "se", "son", "sa",
    "ses", "leur", "leurs", "avec", "plus", "peut",
    "peuvent", "il", "elle", "ils", "elles", "comme",
    "selon", "cela", "donc", "ainsi", "tout", "tous",
    "chaque", "information", "informations"
}


def load_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())

    return "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )


def tokenize(text):
    normalized = normalize(text)

    words = re.findall(r"\b[a-z0-9]+\b", normalized)

    return {
        word for word in words
        if len(word) > 2 and word not in STOPWORDS
    }


def calculate_f1(predicted, reference):
    if not predicted or not reference:
        return 0.0

    common = predicted & reference

    precision = len(common) / len(predicted)
    recall = len(common) / len(reference)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def extract_source_ids(response):
    return {
        source["chunk_id"]
        for source in response.get("sources", [])
        if isinstance(source, dict) and source.get("chunk_id")
    }


def calculate_faithfulness(answer, sources):
    """
    Lexical faithfulness proxy:
    proportion of significant answer terms found
    in the retrieved source texts.

    This is an approximation, not a semantic entailment metric.
    """

    answer_tokens = tokenize(answer)

    context_text = " ".join(
        source.get("text", "")
        for source in sources
        if isinstance(source, dict)
    )

    context_tokens = tokenize(context_text)

    if not answer_tokens:
        return 0.0

    return len(answer_tokens & context_tokens) / len(answer_tokens)


def evaluate_generation(answer, expected_answer, source_ids):
    answer_tokens = tokenize(answer)
    reference_tokens = tokenize(expected_answer)

    answer_relevance = calculate_f1(
        answer_tokens,
        reference_tokens
    )

    cited_ids = set(CHUNK_ID_PATTERN.findall(answer))

    if cited_ids:
        citation_rate = len(cited_ids & source_ids) / len(cited_ids)
    else:
        citation_rate = 0.0

    return {
        "answer_relevance": answer_relevance,
        "citation_rate": citation_rate,
        "cited_ids": sorted(cited_ids),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()

    if not questions:
        raise ValueError("Le jeu d'évaluation est vide.")

    print("Initialisation du RAG...")
    rag = RAGChain(top_k=5)

    results = []

    print("\n========== ÉVALUATION DE LA GÉNÉRATION ==========\n")

    for item in questions:
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_status = item.get("expected_status")

        print(f"Question : {question}")

        try:
            response = rag.invoke(question)

            answer = response.get("answer", "")
            status = response.get("status", "unknown")

            sources = response.get("sources", [])
            source_ids = extract_source_ids(response)

            faithfulness = calculate_faithfulness(
                answer,
                sources
            )

            metrics = evaluate_generation(
                answer,
                expected_answer,
                source_ids
            )

            result = {
                "id": item["id"],
                "question": question,
                "status": status,
                "expected_status": expected_status,
                "status_match": status == expected_status,
                "answer": answer,
                "retrieved_chunks": response.get("retrieved_chunks", 0),
                "used_chunks": response.get("used_chunks", 0),
                "faithfulness": faithfulness,
                "answer_relevance": metrics["answer_relevance"],
                "citation_rate": metrics["citation_rate"],
                "retrieved_source_ids": sorted(source_ids),
                "cited_ids": metrics["cited_ids"],
            }

            results.append(result)

            print(f"Statut : {status}")
            print(f"Réponse : {answer}")
            print(f"Chunks utilisés : {response.get('used_chunks', 0)}")
            print(f"Faithfulness (proxy) : {faithfulness:.3f}")
            print(
                f"Answer Relevance (proxy) : "
                f"{metrics['answer_relevance']:.3f}"
            )
            print(f"Citation Rate : {metrics['citation_rate']:.3f}")
            print("-" * 60)

        except Exception as error:
            print(f"Erreur pour {item['id']} : {error}")

    answered_results = [
        result for result in results
        if result["expected_status"] == "answered"
    ]

    summary = {
        "total_questions": len(questions),
        "successful_evaluations": len(results),
        "answered_questions": len(answered_results),
        "status_accuracy": (
            sum(r["status_match"] for r in results) / len(results)
            if results else 0.0
        ),
        "faithfulness_proxy": (
            sum(r["faithfulness"] for r in answered_results)
            / len(answered_results)
            if answered_results else None
        ),
        "answer_relevance_proxy": (
            sum(r["answer_relevance"] for r in answered_results)
            / len(answered_results)
            if answered_results else None
        ),
        "citation_rate": (
            sum(r["citation_rate"] for r in answered_results)
            / len(answered_results)
            if answered_results else None
        ),
    }

    report = {
        "summary": summary,
        "details": results,
    }

    json_path = REPORT_DIR / "generation_results.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    csv_path = REPORT_DIR / "generation_results.csv"

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "id",
            "question",
            "status",
            "expected_status",
            "status_match",
            "retrieved_chunks",
            "used_chunks",
            "faithfulness",
            "answer_relevance",
            "citation_rate",
            "answer",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== RÉSULTATS GLOBAUX ==========")

    for metric, value in summary.items():
        print(f"{metric} : {value}")

    print(f"\nRapport JSON : {json_path}")
    print(f"Rapport CSV : {csv_path}")


if __name__ == "__main__":
    main()
