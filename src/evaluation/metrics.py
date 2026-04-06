"""
Comprehensive evaluation metrics for suicidal ideation detection models.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true,
    y_pred,
    y_proba=None,
    model_name: str = "model",
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels.
    y_pred : array-like of int
        Predicted hard labels.
    y_proba : array-like of float, optional
        Predicted probabilities for the positive class (column 1).
        Required for ROC-AUC and average precision.
    model_name : str
        Identifier stored in the returned dict.

    Returns
    -------
    dict
        Keys: model_name, accuracy, precision_weighted, recall_weighted,
        f1_weighted, precision_macro, recall_macro, f1_macro, specificity,
        roc_auc (if y_proba given), average_precision (if y_proba given).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)

    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Specificity = TN / (TN + FP)  — binary only
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        specificity = float("nan")

    metrics: dict = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "specificity": specificity,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["average_precision"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["average_precision"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Full formatted report
# ---------------------------------------------------------------------------


def print_full_report(
    y_true,
    y_pred,
    y_proba=None,
    model_name: str = "model",
) -> None:
    """
    Print a nicely formatted classification report, confusion matrix, and
    all scalar metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
    y_proba        : array-like of float, optional — positive-class probabilities
    model_name     : str
    """
    metrics = compute_metrics(y_true, y_pred, y_proba, model_name)

    border = "=" * 65
    print(f"\n{border}")
    print(f"  Evaluation Report — {model_name}")
    print(border)

    print("\n[Classification Report]")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Non-Suicide", "Suicide"],
            zero_division=0,
        )
    )

    print("[Confusion Matrix]")
    cm = confusion_matrix(y_true, y_pred)
    header = f"{'':20s}  {'Pred Non-Suicide':>16s}  {'Pred Suicide':>12s}"
    print(header)
    labels = ["Non-Suicide", "Suicide"]
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>14d}" for v in row)
        print(f"  True {labels[i]:<14s}  {row_str}")

    print("\n[Scalar Metrics]")
    skip = {"model_name"}
    for key, val in metrics.items():
        if key in skip:
            continue
        if isinstance(val, float):
            print(f"  {key:<25s}: {val:.4f}")
        else:
            print(f"  {key:<25s}: {val}")
    print(border + "\n")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(results_dict: Dict[str, dict]) -> pd.DataFrame:
    """
    Print a formatted comparison table for multiple models and return a
    DataFrame sorted descending by F1 (weighted).

    Parameters
    ----------
    results_dict : dict
        {model_name: metrics_dict}  — metrics_dict as returned by
        ``compute_metrics``.

    Returns
    -------
    pd.DataFrame
    """
    records = []
    for name, metrics in results_dict.items():
        row = {"model": name}
        row.update({k: v for k, v in metrics.items() if k != "model_name"})
        records.append(row)

    df = pd.DataFrame(records)
    sort_col = "f1_weighted" if "f1_weighted" in df.columns else df.columns[1]
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Pretty-print
    float_cols = df.select_dtypes(include=[float]).columns
    print("\n" + "=" * 90)
    print(f"  Model Comparison  (sorted by {sort_col})")
    print("=" * 90)
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))
    print("=" * 90 + "\n")

    return df


# ---------------------------------------------------------------------------
# ROC curves for multiple models
# ---------------------------------------------------------------------------


def plot_roc_curves(
    models_results: Dict[str, dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves for multiple models on a single figure.

    Parameters
    ----------
    models_results : dict
        {model_name: {"y_true": array, "y_proba": array}}
        where y_proba contains positive-class probabilities.
    save_path : str, optional
        File path to save the figure.  If None the figure is displayed.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.50)")

    for name, data in models_results.items():
        y_true = np.asarray(data["y_true"])
        y_proba = np.asarray(data["y_proba"])
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
        except ValueError as exc:
            print(f"  Skipping ROC for '{name}': {exc}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"ROC curves saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------


def cross_validate_model(
    model,
    X,
    y,
    cv: int = 5,
    scoring: List[str] = None,
) -> dict:
    """
    Run sklearn ``cross_validate`` and return mean ± std for each metric.

    Parameters
    ----------
    model : sklearn estimator
    X : array-like — feature matrix
    y : array-like — labels
    cv : int — number of folds
    scoring : list of str — sklearn scorer names
               default: ["accuracy", "f1_weighted", "roc_auc"]

    Returns
    -------
    dict
        {metric_name: {"mean": float, "std": float}}
    """
    if scoring is None:
        scoring = ["accuracy", "f1_weighted", "roc_auc"]

    print(f"Running {cv}-fold cross-validation …")
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        if key in cv_results:
            scores = cv_results[key]
            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores))
            summary[metric] = {"mean": mean_val, "std": std_val}
            print(f"  {metric:<20s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            summary[metric] = {"mean": float("nan"), "std": float("nan")}

    return summary


# ---------------------------------------------------------------------------
# Compatibility evaluation helpers used by train.py / evaluate.py
# ---------------------------------------------------------------------------


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            target_names=["Low Risk", "Moderate Risk", "High Risk"],
        ),
    }
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")
    return metrics


def evaluate_model_torch(model, dataloader, device: str = "cpu", model_name: str = "torch_model") -> dict:
    import torch

    model.eval()
    y_true = []
    y_pred = []
    y_proba = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                xb, lengths, yb = batch
                logits = model(xb.to(device), lengths.to(device))
            else:
                xb, yb = batch
                logits = model(xb.to(device))

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_proba.extend(probs.tolist())

    return evaluate_model(
        model=_StaticPredictionModel(np.array(y_pred), np.array(y_proba)),
        X_test=np.empty((len(y_pred), 0)),
        y_test=np.array(y_true),
        model_name=model_name,
    )


def evaluate_transformer(classifier, texts, labels, batch_size: int = 32) -> dict:
    preds, probs = classifier.predict(list(texts), batch_size=batch_size)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(labels, probs, multi_class="ovr", average="weighted"),
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": classification_report(
            labels, preds, zero_division=0, target_names=["Low Risk", "Moderate Risk", "High Risk"]
        ),
    }


class _StaticPredictionModel:
    """
    Tiny compatibility shim so evaluate_model can be reused for precomputed outputs.
    """

    def __init__(self, preds: np.ndarray, proba: np.ndarray):
        self._preds = preds
        self._proba = proba

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        return self._proba
