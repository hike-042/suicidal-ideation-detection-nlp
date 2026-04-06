"""
Benchmark runner for the active website risk pipeline.

This evaluates the current four-class system on a labeled CSV benchmark using
either the local fallback classifier or the full orchestrator path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.agents.orchestrator import AgentOrchestrator, MLFallbackClassifier


LABELS = [
    "LOW_RISK",
    "MODERATE_RISK",
    "HIGH_RISK_SELF_HARM",
    "HIGH_RISK_HARM_TO_OTHERS",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the active website risk pipeline.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/benchmarks/system_benchmark.csv",
        help="CSV benchmark file with columns: text, expected_label, category, notes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fallback", "orchestrator"],
        default="fallback",
        help="Use the local fallback classifier or the full orchestrator path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Directory for benchmark JSON/CSV reports.",
    )
    return parser.parse_args()


def get_system(mode: str):
    if mode == "orchestrator":
        return AgentOrchestrator()
    return MLFallbackClassifier()


def load_benchmark(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "expected_label", "category", "notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark file is missing required columns: {sorted(missing)}")
    bad_labels = sorted(set(df["expected_label"]) - set(LABELS))
    if bad_labels:
        raise ValueError(f"Benchmark file contains unsupported labels: {bad_labels}")
    return df


def run_benchmark(df: pd.DataFrame, mode: str) -> tuple[dict, pd.DataFrame]:
    system = get_system(mode)
    rows = []

    for _, record in df.iterrows():
        text = str(record["text"])
        expected = str(record["expected_label"])
        result = system.analyze(text)
        predicted = result.get("classification", {}).get("risk_level", "LOW_RISK")
        score = int(result.get("classification", {}).get("risk_score", 0) or 0)
        confidence = float(result.get("classification", {}).get("confidence", 0.0) or 0.0)
        rows.append(
            {
                "text": text,
                "expected_label": expected,
                "predicted_label": predicted,
                "correct": expected == predicted,
                "risk_score": score,
                "confidence": confidence,
                "tier_used": result.get("tier_used", ""),
                "category": str(record.get("category", "")),
                "notes": str(record.get("notes", "")),
            }
        )

    results_df = pd.DataFrame(rows)
    y_true = results_df["expected_label"]
    y_pred = results_df["predicted_label"]

    category_metrics = []
    for category, group in results_df.groupby("category"):
        category_metrics.append(
            {
                "category": category,
                "samples": int(len(group)),
                "accuracy": float(accuracy_score(group["expected_label"], group["predicted_label"])),
                "correct": int(group["correct"].sum()),
                "mismatches": int((~group["correct"]).sum()),
            }
        )

    label_metrics = []
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    for label in LABELS:
        r = report.get(label, {})
        label_metrics.append(
            {
                "label": label,
                "precision": float(r.get("precision", 0.0)),
                "recall": float(r.get("recall", 0.0)),
                "f1": float(r.get("f1-score", 0.0)),
                "support": int(r.get("support", 0)),
            }
        )

    mismatches_df = results_df.loc[~results_df["correct"]].copy()
    mismatch_categories = []
    if not mismatches_df.empty:
        grouped = mismatches_df.groupby(["category", "expected_label", "predicted_label"]).size().reset_index(name="count")
        mismatch_categories = grouped.sort_values(["count", "category"], ascending=[False, True]).to_dict(orient="records")

    metrics = {
        "mode": mode,
        "samples": int(len(results_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": report,
        "label_metrics": label_metrics,
        "category_metrics": category_metrics,
        "category_failure_summary": mismatch_categories,
        "mismatches": mismatches_df[["text", "expected_label", "predicted_label", "category", "notes"]].to_dict(orient="records"),
    }
    return metrics, results_df


def save_outputs(metrics: dict, results_df: pd.DataFrame, output_dir: str, mode: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"benchmark_{mode}.json"
    csv_path = out_dir / f"benchmark_predictions_{mode}.csv"

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    results_df.to_csv(csv_path, index=False)

    return json_path, csv_path


def print_summary(metrics: dict):
    print("\n" + "=" * 72)
    print(f"  SYSTEM BENCHMARK - {metrics['mode'].upper()}")
    print("=" * 72)
    print(f"Samples             : {metrics['samples']}")
    print(f"Accuracy            : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)   : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)      : {metrics['recall_macro']:.4f}")
    print(f"F1 (macro)          : {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted)   : {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted)       : {metrics['f1_weighted']:.4f}")

    print("\nPer-class recall:")
    for item in metrics["label_metrics"]:
        print(f"  {item['label']:<24s} {item['recall']:.4f}  (support={item['support']})")

    print("\nCategory accuracy:")
    for item in sorted(metrics["category_metrics"], key=lambda x: (x['accuracy'], x['category']))[:12]:
        print(f"  {item['category']:<24s} {item['accuracy']:.4f}  ({item['correct']}/{item['samples']})")

    mismatches = metrics.get("mismatches", [])
    print(f"\nMismatches          : {len(mismatches)}")
    for item in mismatches[:12]:
        print(f"- category={item['category']} expected={item['expected_label']} predicted={item['predicted_label']} text={item['text'][:80]}")

    if metrics.get("category_failure_summary"):
        print("\nMost common failure slices:")
        for item in metrics["category_failure_summary"][:10]:
            print(
                f"  {item['category']}: expected {item['expected_label']} -> predicted {item['predicted_label']} x{item['count']}"
            )
    print("=" * 72 + "\n")


def main():
    args = parse_args()
    df = load_benchmark(args.data)
    metrics, results_df = run_benchmark(df, args.mode)
    json_path, csv_path = save_outputs(metrics, results_df, args.output_dir, args.mode)
    print_summary(metrics)
    print(f"Saved JSON report : {json_path}")
    print(f"Saved CSV report  : {csv_path}")


if __name__ == "__main__":
    main()
