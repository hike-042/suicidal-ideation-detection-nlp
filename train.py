"""
train.py - Main training script for Suicidal Ideation Detection in Social Media
Usage:
    python train.py --synthetic --models classical
    python train.py --data data/raw/social_media_suicide_detection.csv --models all
    python train.py --models transformer --epochs 5
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train suicidal ideation detection models (3-class: low/moderate/high risk)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/social_media_suicide_detection.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic 3-class data instead of real CSV",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help='Comma-separated list: "classical", "lstm", "transformer", or "all"',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Root output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs for deep learning models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for deep learning models",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum TF-IDF features",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU even if available",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def section(title: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def ensure_dirs(output_dir: str):
    for sub in ("models", "plots", "results"):
        Path(output_dir, sub).mkdir(parents=True, exist_ok=True)


def models_requested(models_arg: str):
    if models_arg.strip().lower() == "all":
        return {"classical", "lstm", "transformer"}
    return {m.strip().lower() for m in models_arg.split(",")}


def _safe_float(val):
    """Convert numpy float / plain float to Python float for JSON serialisation."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def make_logistic_regression(**kwargs):
    """Compatibility wrapper for older scikit-learn versions."""
    from sklearn.linear_model import LogisticRegression

    try:
        return LogisticRegression(multi_class="multinomial", **kwargs)
    except TypeError:
        return LogisticRegression(**kwargs)


# ---------------------------------------------------------------------------
# 3-class label helpers
# ---------------------------------------------------------------------------

RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}


def per_class_f1(y_true, y_pred):
    """Return a dict of per-class F1 scores."""
    from sklearn.metrics import f1_score
    scores = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        RISK_LABELS.get(i, str(i)): _safe_float(s)
        for i, s in enumerate(scores)
    }


def compute_3class_metrics(y_true, y_pred, y_proba=None, model_name="model"):
    """Compute accuracy, weighted F1, per-class F1, and optionally AUC-ROC."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix, classification_report,
    )
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy":   _safe_float(accuracy_score(y_true, y_pred)),
        "f1_weighted": _safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": _safe_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": _safe_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=[RISK_LABELS[i] for i in sorted(RISK_LABELS)],
            zero_division=0,
        ),
    }

    # AUC-ROC (one-vs-rest, requires probabilities)
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            metrics["auc_roc"] = _safe_float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        except Exception:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args):
    """Load a dataframe with columns 'text' and 'risk_level' (0/1/2)."""
    import pandas as pd
    import numpy as np

    if args.synthetic:
        print("Generating synthetic 3-class dataset...")
        return _generate_synthetic_3class(n_samples=3000, random_state=42)

    if not Path(args.data).exists():
        print(f"ERROR: Dataset not found at '{args.data}'.")
        print("  Run with --synthetic to use synthetic data, or provide a valid --data path.")
        sys.exit(1)

    print(f"Loading dataset from: {args.data}")
    df = pd.read_csv(args.data)

    # Detect label column
    label_col = None
    for candidate in ("risk_level", "label", "class"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        print(f"ERROR: Expected label column ('risk_level', 'label', or 'class') not found.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    text_col = "text" if "text" in df.columns else df.columns[0]

    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "risk_level"})

    # Normalise labels to 0/1/2 integers if they are strings
    if df["risk_level"].dtype == object:
        mapping = {
            "low_risk": 0, "low": 0, "non-suicide": 0, "non_suicide": 0,
            "moderate_risk": 1, "moderate": 1, "depression": 1,
            "high_risk": 2, "high": 2, "suicide": 2,
        }
        df["risk_level"] = df["risk_level"].str.strip().str.lower().map(mapping)
        df = df.dropna(subset=["risk_level"])
        df["risk_level"] = df["risk_level"].astype(int)

    return df


def _generate_synthetic_3class(n_samples: int = 3000, random_state: int = 42):
    """
    Build a synthetic 3-class dataset.
    Classes: 0=low_risk, 1=moderate_risk, 2=high_risk
    """
    import random
    import pandas as pd

    rng = random.Random(random_state)
    n_per_class = n_samples // 3

    # --- Low risk (normal social media) ---
    low_risk_templates = [
        "Today I {verb}. {feeling}",
        "{weather} {activity}",
        "{hobby} has been {adj} lately.",
        "Just {verb2} with {person}. {positive}",
        "Working on {project}. Things are {adj2}.",
    ]
    verbs = ["went for a walk", "tried a new recipe", "finished reading a book",
             "spent time with family", "had a great workout", "visited the farmers market",
             "watched a documentary", "cleaned the house", "played board games"]
    feelings = ["Feeling good!", "Really enjoyed it.", "Highly recommend it.",
                 "Such a nice day.", "Life is good.", "Grateful for today."]
    weathers = ["Sunny day today.", "Beautiful weather outside.", "It rained all morning.",
                 "Perfect autumn afternoon.", "Snow is falling gently."]
    activities = ["Went for a run.", "Had coffee with a friend.", "Did some gardening.",
                   "Took the dog to the park.", "Baked cookies."]
    hobbies = ["Photography", "Cooking", "Running", "Reading", "Cycling", "Knitting",
                "Gaming", "Drawing", "Hiking", "Yoga"]
    adjs = ["really fun", "surprisingly relaxing", "challenging but rewarding",
             "great", "enjoyable", "a game-changer"]
    verbs2 = ["had lunch", "caught up", "went hiking", "watched a movie", "played tennis"]
    persons = ["my friend", "my family", "a colleague", "my partner", "my sister"]
    positives = ["Best day in a while!", "So much fun.", "Really needed that.",
                  "Feeling recharged.", "Great times."]
    projects = ["a new side project", "a challenging assignment", "a creative project",
                  "a home improvement task", "a work presentation"]
    adjs2 = ["going well", "coming together nicely", "really promising", "on track"]

    low_samples = []
    for _ in range(n_per_class):
        t = rng.choice(low_risk_templates)
        text = t.format(
            verb=rng.choice(verbs),
            feeling=rng.choice(feelings),
            weather=rng.choice(weathers),
            activity=rng.choice(activities),
            hobby=rng.choice(hobbies),
            adj=rng.choice(adjs),
            verb2=rng.choice(verbs2),
            person=rng.choice(persons),
            positive=rng.choice(positives),
            project=rng.choice(projects),
            adj2=rng.choice(adjs2),
        )
        low_samples.append(text.strip())

    # --- Moderate risk (depression / indirect distress) ---
    moderate_templates = [
        "I've been feeling really {neg_feel} lately. {context}",
        "Everything just feels so {adj_neg}. {continuation}",
        "I don't know how much longer I can {struggle}. {support}",
        "Struggling with {issue} lately. {reflection}",
        "{context2} It's been really hard to {task}.",
        "Some days I wonder {wonder}. {closing}",
    ]
    neg_feels = ["down", "tired", "empty", "overwhelmed", "lost", "disconnected",
                  "anxious", "hopeless", "numb", "exhausted"]
    contexts = ["I can't seem to shake it.", "Not sure what's wrong with me.",
                  "Everything feels heavy.", "I just want to feel normal again.",
                  "I've been isolating myself.", "Hard to get out of bed most days."]
    adj_negs = ["heavy", "pointless", "exhausting", "grey", "difficult", "overwhelming"]
    continuations = ["I can't explain it.", "It's getting harder every day.",
                      "I feel so alone in this.", "Nobody understands.",
                      "I barely leave the house anymore.", "Sleep is the only escape."]
    struggles = ["cope", "keep going", "pretend everything is fine", "stay motivated",
                  "feel something", "connect with people"]
    supports = ["Therapy hasn't helped much.", "I've tried talking to people.",
                  "Not sure where to turn.", "I feel like a burden.",
                  "Even my closest friends don't know."]
    issues = ["depression", "anxiety", "loneliness", "grief", "burnout",
               "social isolation", "self-doubt", "relationship problems"]
    reflections = ["It's affecting my work.", "I've been cancelling plans.",
                    "My appetite is gone.", "I cry without knowing why.",
                    "I have no energy for anything I used to love."]
    contexts2 = ["Since my breakup,", "After losing my job,", "Since my diagnosis,",
                  "After my friend moved away,", "Since the accident,"]
    tasks = ["function normally", "get out of bed", "eat regularly",
              "concentrate on anything", "feel motivated"]
    wonders = ["if things will ever get better", "if I'll always feel this way",
                "what the point of it all is", "if anyone actually cares"]
    closings = ["I try to stay positive.", "I keep going through the motions.",
                 "I don't know what else to do.", "Hoping it passes."]

    moderate_samples = []
    for _ in range(n_per_class):
        t = rng.choice(moderate_templates)
        text = t.format(
            neg_feel=rng.choice(neg_feels),
            context=rng.choice(contexts),
            adj_neg=rng.choice(adj_negs),
            continuation=rng.choice(continuations),
            struggle=rng.choice(struggles),
            support=rng.choice(supports),
            issue=rng.choice(issues),
            reflection=rng.choice(reflections),
            context2=rng.choice(contexts2),
            task=rng.choice(tasks),
            wonder=rng.choice(wonders),
            closing=rng.choice(closings),
        )
        moderate_samples.append(text.strip())

    # --- High risk (explicit suicidal ideation) ---
    high_templates = [
        "{context} I {feeling}. {continuation}",
        "I've been thinking about {ideation}. {detail}",
        "{context} The pain is {pain_adj}. {closing}",
        "I don't want to be alive anymore. {detail2}",
        "{context} I {feeling2} and I {action}. {closing2}",
        "I have a plan. {plan_detail} I'm not afraid anymore.",
    ]
    hi_contexts = [
        "After everything that happened,", "I can't take it anymore.",
        "I've tried everything.", "Nobody cares anyway.",
        "Living with this pain every day,", "I've given up fighting.",
        "I've said my goodbyes.", "There's no point continuing.",
    ]
    hi_feelings = [
        "feel completely hopeless", "see no reason to keep going",
        "feel like a burden to everyone", "don't see any way out",
        "feel worthless and empty", "feel like the world would be better without me",
        "have lost all will to live", "am done with everything",
    ]
    hi_continuations = [
        "I've made up my mind.", "I don't know how much longer I can hold on.",
        "I've been thinking about this for months.", "The thoughts won't go away.",
        "I have nothing left to live for.", "I just want it to end.",
        "I'm not going to fight it anymore.",
    ]
    ideations = [
        "suicide every day", "ending my life", "ways to kill myself",
        "how to end the pain permanently", "not waking up tomorrow",
        "disappearing forever", "self-harm", "taking all my pills",
    ]
    details = [
        "I've been researching methods.", "I wrote a note already.",
        "I've told no one.", "I've chosen a date.",
        "I've given away my belongings.", "I've said goodbye to the people I love.",
    ]
    pain_adjs = ["unbearable", "too much", "constant", "suffocating", "endless"]
    hi_closings = [
        "I can't go on like this.", "I'm exhausted of pretending.",
        "I just want peace.", "Nothing will get better.",
        "I'm done.", "This is my only way out.",
    ]
    details2 = [
        "I've made a plan and I'm ready.", "I've been saying goodbye.",
        "I have the means.", "I've written letters to my family.",
        "I've researched everything I need to know.",
    ]
    feelings2 = [
        "feel completely trapped", "see no escape", "feel utterly alone",
        "feel like I am already dead inside", "have no reason left to stay",
    ]
    actions = [
        "am going to end it tonight", "have a plan I intend to follow through with",
        "have already decided this is the end", "am not going to wake up tomorrow if I can help it",
    ]
    closings2 = [
        "Please don't try to stop me.", "I've never been more certain.",
        "I'm at peace with this decision.", "There is no changing my mind.",
    ]
    plan_details = [
        "I know exactly how I will do it.", "I have everything I need.",
        "I have chosen a time and place.", "I have written my goodbye letters.",
    ]

    high_samples = []
    for _ in range(n_per_class):
        t = rng.choice(high_templates)
        text = t.format(
            context=rng.choice(hi_contexts),
            feeling=rng.choice(hi_feelings),
            continuation=rng.choice(hi_continuations),
            ideation=rng.choice(ideations),
            detail=rng.choice(details),
            pain_adj=rng.choice(pain_adjs),
            closing=rng.choice(hi_closings),
            detail2=rng.choice(details2),
            feeling2=rng.choice(feelings2),
            action=rng.choice(actions),
            closing2=rng.choice(closings2),
            plan_detail=rng.choice(plan_details),
        )
        high_samples.append(text.strip())

    records = (
        [{"text": t, "risk_level": 0} for t in low_samples]
        + [{"text": t, "risk_level": 1} for t in moderate_samples]
        + [{"text": t, "risk_level": 2} for t in high_samples]
    )

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Synthetic dataset: {len(df):,} samples")
    print(df["risk_level"].value_counts().sort_index().to_string())
    return df


# ---------------------------------------------------------------------------
# Text preprocessing wrapper (compatible with existing src/data/preprocess.py)
# ---------------------------------------------------------------------------

def preprocess_texts(texts):
    """Apply TextPreprocessor from src or fall back to simple cleaning."""
    try:
        from src.data.preprocess import TextPreprocessor
        pp = TextPreprocessor(remove_stopwords=True, lemmatize=True, min_length=3)
        return pp.fit_transform(list(texts))
    except Exception:
        import re
        def _clean(t):
            t = str(t).lower()
            t = re.sub(r"http\S+", " ", t)
            t = re.sub(r"@\w+", " ", t)
            t = re.sub(r"#\w+", " ", t)
            t = re.sub(r"[^a-z\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t
        return [_clean(t) for t in texts]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(X_train, X_val, X_test, max_features):
    """TF-IDF + sentiment features stacked into sparse matrices."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse as sp

    print("  Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf  = tfidf.transform(X_test)

    try:
        from src.features.sentiment_features import extract_sentiment_features
        import numpy as np

        print("  Extracting sentiment features (train)...")
        sent_train = extract_sentiment_features(list(X_train)).values
        print("  Extracting sentiment features (val)...")
        sent_val   = extract_sentiment_features(list(X_val)).values
        print("  Extracting sentiment features (test)...")
        sent_test  = extract_sentiment_features(list(X_test)).values

        X_train_feat = sp.hstack([X_train_tfidf, sp.csr_matrix(sent_train)])
        X_val_feat   = sp.hstack([X_val_tfidf,   sp.csr_matrix(sent_val)])
        X_test_feat  = sp.hstack([X_test_tfidf,  sp.csr_matrix(sent_test)])
    except Exception as e:
        print(f"  WARNING: Sentiment features unavailable ({e}). Using TF-IDF only.")
        X_train_feat = X_train_tfidf
        X_val_feat   = X_val_tfidf
        X_test_feat  = X_test_tfidf

    return X_train_feat, X_val_feat, X_test_feat, tfidf


# ---------------------------------------------------------------------------
# Classical ML pipeline
# ---------------------------------------------------------------------------

def run_classical(df, output_dir, max_features, random_state=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    import numpy as np
    import joblib

    section("CLASSICAL ML PIPELINE")

    print("[1/5] Preprocessing text...")
    X_raw = df["text"].values
    y     = df["risk_level"].values
    X_clean = preprocess_texts(X_raw)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_clean, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state, stratify=y_tmp
    )
    print(f"    Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    print("[2/5] Extracting features...")
    X_tr_feat, X_val_feat, X_test_feat, tfidf = extract_features(
        X_tr, X_val, X_test, max_features
    )

    print("[3/5] Training classical models...")
    base_models = {
        "Logistic Regression": make_logistic_regression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=random_state), cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    models_dir = Path(output_dir, "models")
    plots_dir  = Path(output_dir, "plots")

    trained_models = {}
    for name, model in base_models.items():
        print(f"    Fitting {name}...")
        t0 = time.time()
        model.fit(X_tr_feat, y_tr)
        print(f"    {name} — {time.time() - t0:.1f}s")
        trained_models[name] = model

    print("[4/5] Evaluating on test set...")
    all_results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_feat)
        try:
            y_proba = model.predict_proba(X_test_feat)
        except Exception:
            y_proba = None

        metrics = compute_3class_metrics(y_test, y_pred, y_proba, model_name=name)
        all_results[name] = metrics

        pc_f1 = metrics["per_class_f1"]
        print(
            f"    {name:<30s}  "
            f"Acc={metrics['accuracy']:.4f}  "
            f"WtdF1={metrics['f1_weighted']:.4f}  "
            f"LR={pc_f1.get('Low Risk', 0):.4f}  "
            f"MR={pc_f1.get('Moderate Risk', 0):.4f}  "
            f"HR={pc_f1.get('High Risk', 0):.4f}"
        )

        # Confusion matrix plot
        try:
            from src.visualization.plots import plot_confusion_matrix
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                title=f"Confusion Matrix – {name}",
                save_path=str(plots_dir / f"cm_{name.replace(' ', '_')}.png"),
            )
        except Exception:
            pass

    print("[5/5] Saving models and vectorizer...")
    joblib.dump(tfidf, str(models_dir / "tfidf_vectorizer.joblib"))
    for name, model in trained_models.items():
        safe_name = name.replace(" ", "_").lower()
        joblib.dump(model, str(models_dir / f"{safe_name}.joblib"))

    return all_results


# ---------------------------------------------------------------------------
# LSTM / TextCNN pipeline
# ---------------------------------------------------------------------------

def run_deep_learning(df, output_dir, epochs, batch_size, device):
    from src.data.preprocess import TextPreprocessor
    from sklearn.model_selection import train_test_split

    section("LSTM / TextCNN PIPELINE")

    print("[1/4] Preprocessing text...")
    pp = TextPreprocessor(remove_stopwords=True, lemmatize=True, min_length=3)
    X_clean = pp.fit_transform(df["text"].tolist())
    y = df["risk_level"].values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_clean, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    print("[2/4] Building vocabulary and data loaders...")
    try:
        from src.data.dataset import Vocabulary, build_dataloaders
        vocab = Vocabulary(min_freq=2)
        vocab.build(X_tr)
        print(f"    Vocabulary size: {len(vocab):,}")
        train_loader, val_loader, test_loader = build_dataloaders(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            vocab=vocab, batch_size=batch_size, max_len=128
        )
    except ImportError as e:
        print(f"    WARNING: Could not import dataset module ({e}). Skipping deep learning.")
        return {}

    models_dir = Path(output_dir, "models")
    plots_dir  = Path(output_dir, "plots")

    all_results = {}

    model_configs = []
    try:
        from src.models.lstm import LSTMClassifier
        model_configs.append((
            LSTMClassifier, "LSTM",
            {"vocab_size": len(vocab), "embed_dim": 128, "hidden_dim": 256,
             "num_layers": 2, "dropout": 0.3, "num_classes": 3}
        ))
    except ImportError:
        print("    WARNING: LSTMClassifier not available.")

    try:
        from src.models.text_cnn import TextCNNClassifier
        model_configs.append((
            TextCNNClassifier, "TextCNN",
            {"vocab_size": len(vocab), "embed_dim": 128, "num_filters": 128,
             "filter_sizes": [2, 3, 4], "dropout": 0.3, "num_classes": 3}
        ))
    except ImportError:
        print("    WARNING: TextCNNClassifier not available.")

    for ModelClass, name, kwargs in model_configs:
        print(f"\n[3/4] Training {name}...")
        try:
            import torch
            from src.training.trainer import Trainer
            from src.evaluation.metrics import evaluate_model_torch
            from src.visualization.plots import plot_training_curves

            model = ModelClass(**kwargs).to(device)
            trainer = Trainer(model, device=device, epochs=epochs, patience=3)
            history = trainer.train(train_loader, val_loader)

            try:
                plot_training_curves(
                    history, title=name,
                    save_path=str(plots_dir / f"training_{name.lower()}.png")
                )
            except Exception:
                pass

            print(f"[4/4] Evaluating {name}...")
            metrics_raw = evaluate_model_torch(model, test_loader, device=device, model_name=name)

            # Re-compute 3-class metrics with our helper
            y_pred_all = []
            y_true_all = []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    xb, yb = batch[0].to(device), batch[1]
                    logits = model(xb)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    y_pred_all.extend(preds.tolist())
                    y_true_all.extend(yb.numpy().tolist())

            metrics = compute_3class_metrics(y_true_all, y_pred_all, model_name=name)
            metrics["f1"] = metrics["f1_weighted"]  # alias
            all_results[name] = metrics

            pc = metrics["per_class_f1"]
            print(
                f"    {name:<20s}  "
                f"Acc={metrics['accuracy']:.4f}  "
                f"WtdF1={metrics['f1_weighted']:.4f}  "
                f"HR-F1={pc.get('High Risk', 0):.4f}"
            )

            torch.save(model.state_dict(), str(models_dir / f"{name.lower()}_model.pt"))
        except Exception as exc:
            print(f"    ERROR training {name}: {exc}")
            import traceback; traceback.print_exc()

    try:
        import joblib
        joblib.dump(vocab, str(models_dir / "vocabulary.joblib"))
    except Exception:
        pass

    return all_results


# ---------------------------------------------------------------------------
# Transformer pipeline
# ---------------------------------------------------------------------------

def run_transformer(df, output_dir, epochs, batch_size, device):
    from sklearn.model_selection import train_test_split

    section("TRANSFORMER (DistilBERT) PIPELINE")

    print("[1/4] Preparing text...")
    X = df["text"].values
    y = df["risk_level"].values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    models_dir = Path(output_dir, "models")
    plots_dir  = Path(output_dir, "plots")
    model_save_path = str(models_dir / "distilbert_classifier")

    print("[2/4] Initialising TransformerClassifier...")
    try:
        from src.models.transformer import TransformerClassifier
    except ImportError as e:
        print(f"    ERROR: Cannot import TransformerClassifier: {e}")
        return {}

    classifier = TransformerClassifier(
        model_name="distilbert-base-uncased",
        num_labels=3,
        max_length=128,
        device=device,
    )

    print("[3/4] Fine-tuning...")
    try:
        from src.training.transformer_trainer import TransformerTrainer
        trainer = TransformerTrainer(
            classifier=classifier,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=2e-5,
            warmup_ratio=0.1,
        )
        history = trainer.train(X_tr, y_tr, X_val, y_val)
        classifier.save(model_save_path)
    except Exception as e:
        print(f"    ERROR during transformer training: {e}")
        import traceback; traceback.print_exc()
        return {}

    try:
        from src.visualization.plots import plot_training_curves
        plot_training_curves(
            history, title="DistilBERT Fine-tuning",
            save_path=str(plots_dir / "training_distilbert.png"),
        )
    except Exception:
        pass

    print("[4/4] Evaluating...")
    try:
        import torch
        import numpy as np
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        y_pred_all = []

        for i in range(0, len(X_test), batch_size):
            batch_texts = list(X_test[i:i + batch_size])
            enc = tokenizer(batch_texts, truncation=True, padding=True,
                             max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = classifier.model(**enc).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_all.extend(preds.tolist())

        metrics = compute_3class_metrics(y_test, y_pred_all, model_name="DistilBERT")
        metrics["f1"] = metrics["f1_weighted"]

        pc = metrics["per_class_f1"]
        print(
            f"    DistilBERT  "
            f"Acc={metrics['accuracy']:.4f}  "
            f"WtdF1={metrics['f1_weighted']:.4f}  "
            f"HR-F1={pc.get('High Risk', 0):.4f}"
        )
        return {"DistilBERT": metrics}
    except Exception as e:
        print(f"    WARNING: Could not compute detailed metrics: {e}")
        return {"DistilBERT": {"accuracy": 0.0, "f1_weighted": 0.0, "f1": 0.0,
                                "per_class_f1": {}, "auc_roc": 0.0}}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict):
    section("MODEL COMPARISON SUMMARY")

    # Header
    col_model  = 30
    col_acc    = 10
    col_wtdf1  = 10
    col_lrf1   = 12
    col_mrf1   = 14
    col_hrf1   = 12

    header = (
        f"{'Model':<{col_model}} "
        f"{'Accuracy':>{col_acc}} "
        f"{'WtdF1':>{col_wtdf1}} "
        f"{'LowRisk-F1':>{col_lrf1}} "
        f"{'ModRisk-F1':>{col_mrf1}} "
        f"{'HighRisk-F1':>{col_hrf1}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, m in sorted(all_results.items(),
                           key=lambda x: x[1].get("f1_weighted", 0), reverse=True):
        pc = m.get("per_class_f1", {})
        print(
            f"{name:<{col_model}} "
            f"{m.get('accuracy', 0):>{col_acc}.4f} "
            f"{m.get('f1_weighted', m.get('f1', 0)):>{col_wtdf1}.4f} "
            f"{pc.get('Low Risk', 0):>{col_lrf1}.4f} "
            f"{pc.get('Moderate Risk', 0):>{col_mrf1}.4f} "
            f"{pc.get('High Risk', 0):>{col_hrf1}.4f}"
        )
    print()


def save_results(all_results: dict, output_dir: str):
    results_dir = Path(output_dir, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "training_results.json"

    serialisable = {}
    for model_name, metrics in all_results.items():
        serialisable[model_name] = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in metrics.items()
        }

    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    start_time = time.time()

    section("=== Suicidal Ideation Detection in Social Media ===")
    print(f"  Three-class classification: Low Risk | Moderate Risk | High Risk")
    print(f"  Data source: Twitter / Reddit social media posts")

    ensure_dirs(args.output_dir)
    requested = models_requested(args.models)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    device = "cpu"
    if not args.no_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = "mps"
                print("\nApple MPS detected.")
            else:
                print("\nNo GPU found; using CPU.")
        except ImportError:
            print("\ntorch not installed; using CPU.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    section("DATA LOADING")

    df = load_data(args)

    print(f"\nDataset loaded: {len(df):,} samples")
    print("Class distribution:")
    dist = df["risk_level"].value_counts().sort_index()
    for idx, cnt in dist.items():
        label = RISK_LABELS.get(idx, str(idx))
        print(f"  {idx} ({label:<16}): {cnt:>6,}  ({100*cnt/len(df):.1f}%)")

    try:
        from src.visualization.plots import (
            plot_class_distribution,
            plot_text_length_distribution,
            plot_risk_wordclouds,
        )

        plot_class_distribution(
            df["risk_level"].values,
            save_path=str(Path(args.output_dir, "plots", "class_distribution.png")),
        )
        plot_text_length_distribution(
            df["text"].astype(str).tolist(),
            df["risk_level"].values,
            save_path=str(Path(args.output_dir, "plots", "text_length_distribution.png")),
        )
        plot_risk_wordclouds(
            df["text"].astype(str).tolist(),
            df["risk_level"].values,
            label_names=RISK_LABELS,
            save_dir=str(Path(args.output_dir, "plots")),
        )
    except Exception as e:
        print(f"WARNING: Could not generate exploratory plots: {e}")

    # ------------------------------------------------------------------
    # Run pipelines
    # ------------------------------------------------------------------
    all_results = {}

    if "classical" in requested:
        try:
            results = run_classical(df.copy(), args.output_dir, args.max_features)
            all_results.update(results)
        except Exception as e:
            print(f"ERROR in classical pipeline: {e}")
            import traceback; traceback.print_exc()

    if "lstm" in requested:
        try:
            import torch  # noqa: F401
            results = run_deep_learning(
                df.copy(), args.output_dir, args.epochs, args.batch_size, device
            )
            all_results.update(results)
        except ImportError:
            print("WARNING: PyTorch not installed. Skipping LSTM/TextCNN pipeline.")
        except Exception as e:
            print(f"ERROR in LSTM pipeline: {e}")
            import traceback; traceback.print_exc()

    if "transformer" in requested:
        try:
            import torch        # noqa: F401
            import transformers # noqa: F401
            results = run_transformer(
                df.copy(), args.output_dir, args.epochs, args.batch_size, device
            )
            all_results.update(results)
        except ImportError:
            print("WARNING: torch/transformers not installed. Skipping transformer pipeline.")
        except Exception as e:
            print(f"ERROR in transformer pipeline: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_results:
        print_summary_table(all_results)
        save_results(all_results, args.output_dir)

        try:
            from src.visualization.plots import plot_model_comparison
            plot_model_comparison(
                all_results,
                save_path=str(Path(args.output_dir, "plots", "model_comparison.png")),
            )
        except Exception as e:
            print(f"WARNING: Could not generate comparison plot: {e}")
    else:
        print("No results collected. Check errors above.")

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    section("DONE")


if __name__ == "__main__":
    main()
