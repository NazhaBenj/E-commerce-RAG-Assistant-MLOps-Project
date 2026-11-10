
import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="reports/evaluation/chunk_300_50_results.json"
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--output",
        default="reports/evaluation/quality_gate_results.json"
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    summary = report["summary"]
    details = report["details"]
    gates = params["quality_gate"]

    checks = [
        ("Recall@5", summary["recall_at_5"],
         gates["min_recall_at_5"]),
        ("Precision@5", summary["precision_at_5"],
         gates["min_precision_at_5"]),
        ("MRR", summary["mrr"], gates["min_mrr"]),
        ("Minimum evaluated questions",
         summary["evaluated_questions"],
         gates["min_evaluated_questions"]),
    ]

    results = []

    for name, actual, threshold in checks:
        passed = actual is not None and actual >= threshold
        results.append({
            "gate": name,
            "actual": actual,
            "threshold": threshold,
            "status": "PASSED" if passed else "FAILED"
        })

    has_ood = any(
        q.get("expected_status") == "out_of_scope"
        for q in details
    )

    if gates.get("require_out_of_scope_case", False):
        results.append({
            "gate": "Out-of-scope test case present",
            "actual": has_ood,
            "threshold": True,
            "status": "PASSED" if has_ood else "FAILED"
        })

    accepted = all(r["status"] == "PASSED" for r in results)

    output = {
        "configuration": summary["configuration"],
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "gates": results
    }

    print("\n========== RAG QUALITY GATES ==========")

    for result in results:
        print(
            f'{result["gate"]}: '
            f'{result["actual"]} -> {result["status"]}'
        )

    print("=======================================")
    print(f'DECISION: {output["decision"]}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Rapport enregistré : {output_path}")

    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
