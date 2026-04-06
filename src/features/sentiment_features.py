"""
sentiment_features.py
---------------------
Sentiment analysis and hand-crafted lexical features for the
Suicidal Ideation Detection project.

Functions
---------
extract_sentiment_features   – per-text feature DataFrame (TextBlob + VADER + lexical)
get_all_handcrafted_features – wrapper returning a numpy array
"""

import re
import warnings
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TextBlob
# ---------------------------------------------------------------------------
try:
    from textblob import TextBlob  # type: ignore
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False
    warnings.warn(
        "textblob is not installed. TextBlob features will be set to 0.0. "
        "Install with: pip install textblob",
        ImportWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# VADER (shipped with NLTK)
# ---------------------------------------------------------------------------
try:
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("[INFO] Downloading NLTK vader_lexicon …")
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except Exception as _vader_exc:
    _VADER_AVAILABLE = False
    _VADER_ANALYZER = None  # type: ignore[assignment]
    warnings.warn(
        f"VADER not available: {_vader_exc}. VADER features will be set to 0.0.",
        ImportWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Pre-compiled patterns for lexical features
# ---------------------------------------------------------------------------
_RE_EXCLAMATION = re.compile(r"!")
_RE_QUESTION = re.compile(r"\?")
_RE_UPPERCASE_WORD = re.compile(r"\b[A-Z]{2,}\b")


# ---------------------------------------------------------------------------
# extract_sentiment_features
# ---------------------------------------------------------------------------

def extract_sentiment_features(texts: List[str]) -> pd.DataFrame:
    """
    Extract a rich set of sentiment and lexical features from a list of texts.

    Feature groups
    --------------
    TextBlob (2 features):
        tb_polarity    – polarity score ∈ [-1, 1]
        tb_subjectivity – subjectivity score ∈ [0, 1]

    VADER (4 features):
        vader_compound – overall sentiment score ∈ [-1, 1]
        vader_pos      – proportion of positive tokens ∈ [0, 1]
        vader_neu      – proportion of neutral tokens ∈ [0, 1]
        vader_neg      – proportion of negative tokens ∈ [0, 1]

    Lexical / surface (7 features):
        text_length    – total character count
        word_count     – number of whitespace-separated tokens
        avg_word_length – mean token length (chars)
        exclamation_count – number of '!' characters
        question_count    – number of '?' characters
        caps_ratio     – fraction of characters that are uppercase
        unique_word_ratio – vocabulary richness (unique words / total words)

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    pd.DataFrame with one row per input text and 13 named feature columns.
    """
    records = []
    total = len(texts)
    report_every = max(1, total // 10)

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        record: dict = {}

        # ---- TextBlob features ------------------------------------------
        if _TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                record["tb_polarity"] = blob.sentiment.polarity
                record["tb_subjectivity"] = blob.sentiment.subjectivity
            except Exception:
                record["tb_polarity"] = 0.0
                record["tb_subjectivity"] = 0.0
        else:
            record["tb_polarity"] = 0.0
            record["tb_subjectivity"] = 0.0

        # ---- VADER features ---------------------------------------------
        if _VADER_AVAILABLE and _VADER_ANALYZER is not None:
            try:
                scores = _VADER_ANALYZER.polarity_scores(text)
                record["vader_compound"] = scores["compound"]
                record["vader_pos"] = scores["pos"]
                record["vader_neu"] = scores["neu"]
                record["vader_neg"] = scores["neg"]
            except Exception:
                record["vader_compound"] = 0.0
                record["vader_pos"] = 0.0
                record["vader_neu"] = 0.0
                record["vader_neg"] = 0.0
        else:
            record["vader_compound"] = 0.0
            record["vader_pos"] = 0.0
            record["vader_neu"] = 0.0
            record["vader_neg"] = 0.0

        # ---- Lexical / surface features ---------------------------------
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        avg_word_length = (
            float(np.mean([len(w) for w in words])) if words else 0.0
        )
        exclamation_count = len(_RE_EXCLAMATION.findall(text))
        question_count = len(_RE_QUESTION.findall(text))

        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )

        unique_word_ratio = (
            len(set(w.lower() for w in words)) / word_count
            if word_count > 0
            else 0.0
        )

        record["text_length"] = text_length
        record["word_count"] = word_count
        record["avg_word_length"] = avg_word_length
        record["exclamation_count"] = exclamation_count
        record["question_count"] = question_count
        record["caps_ratio"] = caps_ratio
        record["unique_word_ratio"] = unique_word_ratio

        records.append(record)

        if (i + 1) % report_every == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"  [sentiment_features] {i+1:>6}/{total}  ({pct:.1f}%)", end="\r")

    print()  # newline after progress

    df = pd.DataFrame(records, columns=[
        "tb_polarity",
        "tb_subjectivity",
        "vader_compound",
        "vader_pos",
        "vader_neu",
        "vader_neg",
        "text_length",
        "word_count",
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "unique_word_ratio",
    ])

    return df


# ---------------------------------------------------------------------------
# get_all_handcrafted_features
# ---------------------------------------------------------------------------

def get_all_handcrafted_features(texts: List[str]) -> np.ndarray:
    """
    Extract all hand-crafted sentiment and lexical features and return
    them as a numpy array suitable for feeding directly into sklearn models.

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    np.ndarray of shape (n_samples, 13), dtype float32.
    """
    df = extract_sentiment_features(texts)
    return df.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sample_texts = [
        "I feel completely hopeless and I want to end my life.",
        "Today was a great day! I went hiking and enjoyed the sunshine.",
        "Nobody cares about me and the pain is unbearable.",
        "Just finished a delicious meal with my family. Life is good!",
        "I have been thinking about suicide every single day.",
        "Working on a new project at work and feeling productive.",
        "WHY DOES EVERYTHING HURT SO MUCH??",
        "spring is here and the flowers are blooming beautifully.",
    ]

    print("=" * 60)
    print("Smoke test: extract_sentiment_features")
    print("=" * 60)
    df_features = extract_sentiment_features(_sample_texts)
    print(df_features.to_string())

    print("\n" + "=" * 60)
    print("Smoke test: get_all_handcrafted_features")
    print("=" * 60)
    arr = get_all_handcrafted_features(_sample_texts)
    print(f"Array shape : {arr.shape}")
    print(f"Array dtype : {arr.dtype}")
    print(f"Sample row 0: {arr[0]}")

    print("\n[INFO] All smoke tests passed.")
