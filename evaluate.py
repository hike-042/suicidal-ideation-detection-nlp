"""
evaluate.py - Evaluation script for saved 3-class models.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved suicidal ideation detection models")
    parser.add_argument("--model-dir", type=str, default="outputs/models/")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=["classical", "lstm", "transformer"], required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-gpu", action="store_true")
    return parser.parse_args()


def section(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")


def get_device(no_gpu: bool) -> str:
    if no_gpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def print_metrics_report(name: str, metrics: dict):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall   : {metrics.get('recall', 0):.4f}")
    print(f"F1       : {metrics.get('f1', 0):.4f}")
    print(f"AUC-ROC  : {metrics.get('auc_roc', 0):.4f}")
    if metrics.get("classification_report"):
        print("\nClassification Report:\n")
        print(metrics["classification_report"])


def evaluate_classical(model_dir: str, data_path: str, model_name_filter, output_dir: str):
    import joblib
    import scipy.sparse as sp

    from src.data.preprocess import (
        load_dataset,
        preprocess_dataframe,
        extract_sentiment_features,
    )
    from src.evaluation.metrics import evaluate_model
    from src.visualization.plots import plot_confusion_matrix, plot_feature_importance

    model_dir_path = Path(model_dir)
    plots_dir = Path(output_dir, "plots", "evaluation")
    plots_dir.mkdir(parents=True, exist_ok=True)

    vec_path = model_dir_path / "tfidf_vectorizer.joblib"
    if not vec_path.exists():
        print(f"ERROR: Missing vectorizer at {vec_path}")
        sys.exit(1)
    tfidf_vectorizer = joblib.load(str(vec_path))

    model_files = {
        "logistic_regression": model_dir_path / "logistic_regression.joblib",
        "linear_svm": model_dir_path / "linear_svm.joblib",
        "random_forest": model_dir_path / "random_forest.joblib",
        "gradient_boosting": model_dir_path / "gradient_boosting.joblib",
    }
    if model_name_filter:
        normalized = model_name_filter.lower().replace(" ", "_")
        model_files = {normalized: model_files.get(normalized)}

    df = preprocess_dataframe(load_dataset(data_path))
    X_text = df["cleaned_text"].values
    y_test = df["label"].values
    X_tfidf = tfidf_vectorizer.transform(X_text)
    X_sent = extract_sentiment_features(X_text)
    X_test = sp.hstack([X_tfidf, X_sent])

    all_metrics = {}
    for model_name, model_path in model_files.items():
        if not model_path or not model_path.exists():
            continue
        estimator = joblib.load(str(model_path))
        metrics = evaluate_model(estimator, X_test, y_test, model_name=model_name)
        all_metrics[model_name] = metrics
        print_metrics_report(model_name, metrics)
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"Confusion Matrix - {model_name}",
            save_path=str(plots_dir / f"eval_cm_{model_name}.png"),
        )
        if model_name == "logistic_regression":
            plot_feature_importance(
                estimator,
                tfidf_vectorizer,
                save_path=str(plots_dir / "eval_feature_importance_lr.png"),
                title="Top Predictive Features (Evaluation Set)",
            )
    return all_metrics


def evaluate_lstm(model_dir: str, data_path: str, model_name_filter, output_dir: str, batch_size: int, device: str):
    import joblib
    import torch

    from src.data.preprocess import load_dataset, preprocess_dataframe
    from src.data.dataset import build_dataloaders
    from src.models.lstm import LSTMClassifier
    from src.models.text_cnn import TextCNNClassifier
    from src.evaluation.metrics import evaluate_model_torch
    from src.visualization.plots import plot_confusion_matrix

    model_dir_path = Path(model_dir)
    plots_dir = Path(output_dir, "plots", "evaluation")
    plots_dir.mkdir(parents=True, exist_ok=True)

    vocab = joblib.load(str(model_dir_path / "vocabulary.joblib"))
    df = preprocess_dataframe(load_dataset(data_path))
    X_text = df["cleaned_text"].values
    y_test = df["label"].values
    _, _, test_loader = build_dataloaders(X_text[:1], y_test[:1], X_text[:1], y_test[:1], X_text, y_test, vocab=vocab, batch_size=batch_size, max_len=128)

    configs = {
        "lstm": (LSTMClassifier, {"vocab_size": len(vocab), "embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3, "num_classes": 3}),
        "textcnn": (TextCNNClassifier, {"vocab_size": len(vocab), "embed_dim": 128, "num_filters": 128, "filter_sizes": [2, 3, 4], "dropout": 0.3, "num_classes": 3}),
    }

    all_metrics = {}
    for key, (ModelClass, kwargs) in configs.items():
        if model_name_filter and model_name_filter.lower() not in {key, key.upper()}:
            continue
        weights_path = model_dir_path / f"{key}_model.pt"
        if not weights_path.exists():
            continue
        model = ModelClass(**kwargs).to(device)
        model.load_state_dict(torch.load(str(weights_path), map_location=device))
        metrics = evaluate_model_torch(model, test_loader, device=device, model_name=key.upper())
        all_metrics[key.upper()] = metrics
        print_metrics_report(key.upper(), metrics)
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"Confusion Matrix - {key.upper()}",
            save_path=str(plots_dir / f"eval_cm_{key}.png"),
        )
    return all_metrics


def evaluate_transformer_model(model_dir: str, data_path: str, output_dir: str, batch_size: int, device: str):
    from src.data.preprocess import load_dataset, preprocess_dataframe
    from src.models.transformer import TransformerClassifier
    from src.evaluation.metrics import evaluate_transformer
    from src.visualization.plots import plot_confusion_matrix

    model_path = Path(model_dir) / "distilbert_classifier"
    if not model_path.exists():
        print(f"ERROR: Missing transformer model at {model_path}")
        sys.exit(1)

    classifier = TransformerClassifier(model_name=str(model_path), num_labels=3, max_length=128, device=device)
    classifier.load(str(model_path))

    df = preprocess_dataframe(load_dataset(data_path), for_transformer=True)
    metrics = evaluate_transformer(classifier, df["text"].values, df["label"].values, batch_size=batch_size)
    print_metrics_report("DistilBERT", metrics)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title="Confusion Matrix - DistilBERT",
        save_path=str(Path(output_dir, "plots", "evaluation", "eval_cm_distilbert.png")),
    )
    return {"DistilBERT": metrics}


def save_evaluation_report(all_metrics: dict, output_dir: str, model_type: str):
    out_path = Path(output_dir, "results", f"evaluation_{model_type}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {name: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()} for name, metrics in all_metrics.items()},
            fh,
            indent=2,
        )
    print(f"\nEvaluation report saved to {out_path}")


def main():
    args = parse_args()
    device = get_device(args.no_gpu)

    section(f"EVALUATION - {args.model_type.upper()}")
    print(f"Model directory : {args.model_dir}")
    print(f"Test data       : {args.data}")
    print(f"Device          : {device}")

    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    all_metrics = {}
    if args.model_type == "classical":
        all_metrics = evaluate_classical(args.model_dir, args.data, args.model_name, args.output_dir)
    elif args.model_type == "lstm":
        all_metrics = evaluate_lstm(args.model_dir, args.data, args.model_name, args.output_dir, args.batch_size, device)
    elif args.model_type == "transformer":
        all_metrics = evaluate_transformer_model(args.model_dir, args.data, args.output_dir, args.batch_size, device)

    if all_metrics:
        save_evaluation_report(all_metrics, args.output_dir, args.model_type)
    section("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
