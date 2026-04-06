"""
demo.py - Interactive CLI demo for 3-class suicidal ideation detection.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

MODELS_DIR = Path("outputs/models")

DISCLAIMER = (
    "\n" + "!" * 70 + "\n"
    "  DISCLAIMER: This is a research tool only.\n"
    "  It is NOT a clinical diagnostic instrument and must NOT replace\n"
    "  professional mental health assessment.\n\n"
    "  If you or someone you know is in crisis, please reach out:\n"
    "    * 988 Suicide & Crisis Lifeline (US): call or text 988\n"
    "    * Crisis Text Line (US): text HOME to 741741\n"
    "    * International crisis centres: https://www.iasp.info/resources/Crisis_Centres/\n"
    "!" * 70 + "\n"
)

LABELS = {
    0: "LOW_RISK",
    1: "MODERATE_RISK",
    2: "HIGH_RISK",
}


def make_logistic_regression(**kwargs):
    """
    Build a LogisticRegression model compatible with both newer and older
    scikit-learn versions.
    """
    from sklearn.linear_model import LogisticRegression

    try:
        return LogisticRegression(multi_class="multinomial", **kwargs)
    except TypeError:
        return LogisticRegression(**kwargs)


def find_best_classical_model():
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    if not vec_path.exists():
        return None, None, None

    try:
        import joblib

        preference = [
            "logistic_regression.joblib",
            "linear_svm.joblib",
            "random_forest.joblib",
            "gradient_boosting.joblib",
        ]
        for filename in preference:
            path = MODELS_DIR / filename
            if path.exists():
                return joblib.load(path), joblib.load(vec_path), path.stem
        return None, None, None
    except Exception as exc:
        print(f"  Warning: could not load saved model ({exc})")
        return None, None, None


def train_fallback_model():
    from sklearn.model_selection import train_test_split

    from src.data.preprocess import (
        generate_synthetic_data,
        preprocess_dataframe,
        extract_tfidf_features,
    )

    print("  No saved models found. Training a quick fallback model on synthetic data...")
    df = preprocess_dataframe(generate_synthetic_data(n_samples=1800, random_state=42))
    X = df["cleaned_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tfidf, _, X_test_tfidf, vectorizer = extract_tfidf_features(
        X_train, X_test, X_test, max_features=10_000
    )
    model = make_logistic_regression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print(f"  Fallback model accuracy: {model.score(X_test_tfidf, y_test):.4f}")
    return model, vectorizer


def predict(text: str, model, vectorizer, preprocess_fn):
    import numpy as np
    import scipy.sparse as sp

    cleaned = preprocess_fn(text)
    X_tfidf = vectorizer.transform([cleaned])

    try:
        n_extra = model.n_features_in_ - X_tfidf.shape[1]
        if n_extra > 0:
            X = sp.hstack([X_tfidf, sp.csr_matrix((1, n_extra))])
        else:
            X = X_tfidf
    except AttributeError:
        X = X_tfidf

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    classes = list(model.classes_)
    pred_label = int(classes[pred_idx])
    confidence = float(proba[pred_idx]) * 100.0

    feature_names = np.array(vectorizer.get_feature_names_out())
    try:
        coef_matrix = model.coef_
        coef = coef_matrix[pred_label]
        feature_vals = X_tfidf.toarray()[0]
        scores = coef * feature_vals
        top_indices = np.argsort(scores)[::-1][:10]
        top_features = [
            (feature_names[i], float(scores[i]))
            for i in top_indices
            if feature_vals[i] > 0
        ][:5]
    except Exception:
        top_features = []

    return LABELS.get(pred_label, str(pred_label)), confidence, top_features


def run_demo(model, vectorizer, model_name: str):
    from src.data.preprocess import clean_text_single

    print(f"\n  Active model: {model_name}")
    print(DISCLAIMER)
    print("  Enter text to analyse (type 'quit' or 'exit' to stop).\n")

    while True:
        try:
            user_input = input("  >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting demo.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n  Exiting demo. Take care.")
            break

        try:
            label, confidence, top_features = predict(
                user_input, model, vectorizer, clean_text_single
            )
        except Exception as exc:
            print(f"  Prediction error: {exc}")
            continue

        bar = "-" * 50
        print(f"\n  {bar}")
        print(f"  Prediction  : {label}")
        print(f"  Confidence  : {confidence:.1f}%")

        if top_features:
            print("\n  Key contributing terms:")
            for feat, score in top_features:
                print(f"    + '{feat}' (weight: {score:+.4f})")

        if label == "HIGH_RISK":
            print("\n  *** If this reflects real feelings, seek immediate support. ***")
            print("      988 (US) | Text HOME to 741741")
        elif label == "MODERATE_RISK":
            print("\n  *** Support may be helpful if this reflects real distress. ***")

        print(f"  {bar}\n")


def main():
    print("\n" + "=" * 70)
    print("   Suicidal Ideation Detection - Interactive Demo")
    print("=" * 70)

    print("\n[1/2] Looking for saved models in outputs/models/...")
    model, vectorizer, model_name = find_best_classical_model()
    if model is None:
        print("[2/2] Training fallback model...")
        model, vectorizer = train_fallback_model()
        model_name = "logistic_regression_fallback"
    else:
        print(f"  Found model: {model_name}")
        print("[2/2] Model loaded successfully.")

    run_demo(model, vectorizer, model_name)


if __name__ == "__main__":
    main()
