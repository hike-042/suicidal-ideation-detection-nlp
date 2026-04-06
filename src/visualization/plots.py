"""
Visualization utilities for suicidal ideation detection project.
Uses matplotlib and seaborn.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Consistent colour palette across all plots
_PALETTE = ["#4C72B0", "#DD8452"]
_CLASS_LABELS = ["Non-Suicide", "Suicide"]
_RISK_CLASS_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]

sns.set_theme(style="whitegrid", palette=_PALETTE)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save the figure to *save_path* or display it interactively."""
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved → {save_path}")
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true,
    y_pred=None,
    labels: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot a confusion matrix heatmap showing both raw counts and row-normalised
    percentages.

    Parameters
    ----------
    y_true, y_pred : array-like — ground-truth and predicted labels
    labels         : list of str — class names (default: ["Non-Suicide", "Suicide"])
    save_path      : file path to save; if None, the plot is shown
    """
    if y_pred is None:
        cm = np.asarray(y_true)
        if labels is None:
            if cm.shape[0] == 3:
                labels = ["Low Risk", "Moderate Risk", "High Risk"]
            else:
                labels = _CLASS_LABELS
    else:
        if labels is None:
            unique = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
            labels = ["Low Risk", "Moderate Risk", "High Risk"] if len(unique) == 3 else _CLASS_LABELS
        cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Build annotation strings: "count\n(pct%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def plot_class_distribution(
    y,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart showing class counts and percentages.

    Parameters
    ----------
    y         : array-like of int — label array
    save_path : file path to save; if None, the plot is shown
    """
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    default_labels = _RISK_CLASS_LABELS if len(unique) == 3 else _CLASS_LABELS
    label_names = [default_labels[int(u)] if int(u) < len(default_labels) else str(u)
                   for u in unique]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(label_names, counts, color=_PALETTE[: len(unique)], edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({count / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Text length distribution
# ---------------------------------------------------------------------------


def plot_text_length_distribution(
    texts: List[str],
    labels,
    save_path: Optional[str] = None,
) -> None:
    """
    Histogram of text lengths (word count) overlaid per class.

    Parameters
    ----------
    texts     : list of str
    labels    : array-like of int — parallel label array
    save_path : file path to save; if None, the plot is shown
    """
    labels = np.asarray(labels)
    lengths = np.array([len(t.split()) for t in texts])

    fig, ax = plt.subplots(figsize=(8, 5))

    class_names = _RISK_CLASS_LABELS if len(np.unique(labels)) == 3 else _CLASS_LABELS
    palette = ["#22c55e", "#f59e0b", "#ef4444"] if len(class_names) == 3 else _PALETTE
    for idx, cls_name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        ax.hist(
            lengths[mask],
            bins=40,
            alpha=0.6,
            label=f"{cls_name} (n={mask.sum():,})",
            color=palette[idx % len(palette)],
            edgecolor="white",
        )

    ax.set_xlabel("Text Length (words)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Text Length Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Word cloud
# ---------------------------------------------------------------------------


def plot_wordcloud(
    texts: List[str],
    title: str = "Word Cloud",
    save_path: Optional[str] = None,
) -> None:
    """
    WordCloud visualisation of the combined text corpus.

    Parameters
    ----------
    texts     : list of str
    title     : plot title
    save_path : file path to save; if None, the plot is shown
    """
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError(
            "The 'wordcloud' package is required for plot_wordcloud. "
            "Install it with: pip install wordcloud"
        ) from exc

    combined = " ".join(texts)
    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=False,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold")

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    feature_names,
    importances=None,
    top_n: int = 20,
    save_path: Optional[str] = None,
    title: str = "Top Feature Importances",
) -> None:
    """
    Horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    feature_names : list of str
    importances   : array-like of float — importance scores parallel to feature_names
    top_n         : how many top features to display
    save_path     : file path to save; if None, the plot is shown
    """
    if hasattr(feature_names, "coef_") or hasattr(feature_names, "feature_importances_"):
        model = feature_names
        vectorizer = importances
        feature_names = list(vectorizer.get_feature_names_out())
        if hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_)
            importances = np.mean(np.abs(coefs), axis=0)
        else:
            importances = np.asarray(model.feature_importances_)
    else:
        feature_names = list(feature_names)
        importances = np.asarray(importances)

    indices = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    colors = sns.color_palette("Blues_d", n_colors=len(top_vals))
    ax.barh(top_names, top_vals, color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Dual-subplot figure showing loss and accuracy curves over training epochs.

    Parameters
    ----------
    train_losses, val_losses : list of float — per-epoch loss
    train_accs, val_accs     : list of float — per-epoch accuracy
    save_path                : file path to save; if None, the plot is shown
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    ax1.plot(epochs, train_losses, "o-", color=_PALETTE[0], label="Train Loss")
    ax1.plot(epochs, val_losses, "s--", color=_PALETTE[1], label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Accuracy subplot
    ax2.plot(epochs, train_accs, "o-", color=_PALETTE[0], label="Train Accuracy")
    ax2.plot(epochs, val_accs, "s--", color=_PALETTE[1], label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.set_ylim([0.0, 1.05])

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------


def plot_model_comparison(
    results_df,
    metric: str = "f1",
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart comparing multiple models on a chosen metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain a "model" column and a column named *metric*.
    metric     : str — column name for the metric to visualise (default: "f1")
    save_path  : file path to save; if None, the plot is shown
    """
    if isinstance(results_df, dict):
        rows = []
        for model_name, metrics in results_df.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)
        results_df = pd.DataFrame(rows)

    # Support both "f1" and "f1_weighted" column names
    col = metric if metric in results_df.columns else f"{metric}_weighted"
    if col not in results_df.columns:
        raise KeyError(
            f"Column '{metric}' (or '{metric}_weighted') not found in results_df. "
            f"Available columns: {list(results_df.columns)}"
        )

    df_sorted = results_df.sort_values(col, ascending=True).reset_index(drop=True)
    n = len(df_sorted)
    colors = sns.color_palette("Blues_d", n_colors=n)

    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.55)))
    bars = ax.barh(
        df_sorted["model"],
        df_sorted[col],
        color=colors,
        edgecolor="white",
    )

    for bar, val in zip(bars, df_sorted[col]):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Model Comparison — {col.replace('_', ' ').title()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim([0.0, min(1.0, df_sorted[col].max() * 1.12)])

    _save_or_show(fig, save_path)


def plot_training_curves(history: dict, title: str = "Training Curves", save_path: Optional[str] = None) -> None:
    """
    Compatibility wrapper around plot_training_history.
    """
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    train_accs = history.get("train_acc", [])
    val_accs = history.get("val_acc", [])

    if not train_losses and not val_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No epoch history available", ha="center", va="center", color=_PALETTE[0])
        ax.set_title(title)
        ax.axis("off")
        _save_or_show(fig, save_path)
        return

    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=save_path)


def plot_risk_wordclouds(
    texts: List[str],
    labels,
    label_names: Optional[dict] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Generate separate word clouds for each risk class.
    """
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError("wordcloud is required for risk-specific word clouds.") from exc

    labels = np.asarray(labels)
    label_names = label_names or {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    color_map = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

    for label in sorted(set(labels.tolist())):
        class_texts = [t for t, y in zip(texts, labels) if y == label]
        if not class_texts:
            continue
        cloud = WordCloud(
            width=1200,
            height=700,
            background_color="white",
            colormap="viridis",
            max_words=150,
            collocations=False,
        ).generate(" ".join(class_texts))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label_names.get(int(label), str(label)), color=color_map.get(int(label), "#4C72B0"), fontsize=16)
        out_path = None
        if save_dir:
            safe_name = label_names.get(int(label), str(label)).lower().replace(" ", "_")
            out_path = os.path.join(save_dir, f"wordcloud_{safe_name}.png")
        _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Single ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true,
    y_proba,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a single ROC curve with AUC score displayed in the legend.

    Parameters
    ----------
    y_true    : array-like of int — ground-truth binary labels
    y_proba   : array-like of float — positive-class probabilities
    model_name: str — label used in the legend
    save_path : file path to save; if None, the plot is shown
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        color=_PALETTE[0],
        lw=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.10, color=_PALETTE[0])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    _save_or_show(fig, save_path)
