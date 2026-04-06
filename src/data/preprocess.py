"""
preprocess.py
3-class social media text preprocessor for the Suicidal Ideation Detection project.

Supported classes
-----------------
    low_risk      (0) – normal everyday posts
    moderate_risk (1) – depressive language, no explicit intent
    high_risk     (2) – explicit suicidal ideation / self-harm intent

Public API
----------
TextPreprocessor          – full NLP cleaning pipeline (social-media aware)
load_and_preprocess_data  – load CSV, clean text, encode 3-class labels
split_data                – stratified train / val / test split
"""

import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# NLTK bootstrap – download required corpora silently if missing
# ---------------------------------------------------------------------------
import nltk
from src.data.generation_lexicon import expand_generation_language

_NLTK_RESOURCES = [
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("corpora/omw-1.4",      "omw-1.4"),
]

for _resource_path, _resource_id in _NLTK_RESOURCES:
    try:
        nltk.data.find(_resource_path)
    except LookupError:
        print(f"[INFO] Downloading NLTK resource: {_resource_id}")
        nltk.download(_resource_id, quiet=True)

from nltk.corpus import stopwords as _nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_ENGLISH_STOPWORDS = set(_nltk_stopwords.words("english"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# ---------------------------------------------------------------------------
# 3-class label configuration
# ---------------------------------------------------------------------------

#: Canonical label ordering used to fix LabelEncoder class indices.
#: low_risk=0, moderate_risk=1, high_risk=2
LABEL_ORDER: List[str] = ["low_risk", "moderate_risk", "high_risk"]

#: String risk level → integer code.
LABEL_TO_INT: Dict[str, int] = {
    "low_risk":      0,
    "moderate_risk": 1,
    "high_risk":     2,
}

#: Integer code → string risk level.
INT_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Full NLP preprocessing pipeline for social-media text (Twitter / Reddit).

    Pipeline (in order)
    -------------------
    1. clean_social_media  – strip RT prefix, Reddit blockquotes, @mentions,
                             resolve #hashtags to bare words, normalise
                             repeated characters (sooooo → so), remove emojis.
    2. clean_text          – lowercase, strip URLs, strip HTML, remove numbers,
                             remove non-alphabetic characters, collapse whitespace.
    3. tokenize            – NLTK word_tokenize.
    4. (optional) remove_stopwords – drop English stopwords.
    5. (optional) lemmatize        – WordNet lemmatization.
    6. filter min_length   – drop tokens shorter than min_length characters.
    7. join                – return space-joined string.

    Parameters
    ----------
    remove_stopwords : bool
        Remove English stop-words (default True).
    lemmatize : bool
        Apply WordNet lemmatizer (default True).
    min_length : int
        Minimum token length to keep (default 3).
    """

    # Compiled patterns (class-level for efficiency)
    _RE_URL           = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+!*(),]|%[0-9a-fA-F]{2})+",
        re.IGNORECASE,
    )
    _RE_HTML          = re.compile(r"<[^>]+>")
    _RE_MENTION       = re.compile(r"@\w+")
    _RE_HASHTAG       = re.compile(r"#(\w+)")          # keep the word, drop '#'
    _RE_RT_PREFIX     = re.compile(r"^\s*RT\s*:?\s*", re.IGNORECASE)
    _RE_REDDIT_QUOTE  = re.compile(r"^>+\s*", re.MULTILINE)
    _RE_ASTERISK_EMPH = re.compile(r"\*{1,2}(\w[\w\s]*?\w)\*{1,2}")
    _RE_REPEATED_CHAR = re.compile(r"(.)\1{2,}")       # sooooo → so
    _RE_EMOJI         = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    _RE_NON_ALPHA     = re.compile(r"[^a-zA-Z\s]")
    _RE_NUMBERS       = re.compile(r"\b\d+\b")
    _RE_EXTRA_WS      = re.compile(r"\s+")

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_length: int = 3,
    ) -> None:
        self.remove_stopwords_enabled = remove_stopwords
        self.lemmatize_enabled        = lemmatize
        self.min_length               = min_length
        self._lemmatizer              = WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Step 1 – Social-media-specific cleaning
    # ------------------------------------------------------------------

    def clean_social_media(self, text: str) -> str:
        """
        Strip social-media markup from *text*.

        Operations (in order):
        1. Remove URLs (http / https).
        2. Strip RT prefix ("RT :" or "RT ").
        3. Strip Reddit blockquote markers (leading '>').
        4. Remove asterisk emphasis (*word* / **word**) — keep the word.
        5. Replace @mentions with TOKEN_MENTION placeholder.
        6. Replace #hashtag with the bare word (strip '#' only).
        7. Remove emojis (keep surrounding text meaning).
        8. Normalise repeated characters (sooooo → so).
        9. Strip extra whitespace.

        Parameters
        ----------
        text : str

        Returns
        -------
        str – text with social-media markup removed.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = expand_generation_language(text)

        # 1. URLs
        text = self._RE_URL.sub(" ", text)
        # 2. RT prefix
        text = self._RE_RT_PREFIX.sub("", text)
        # 3. Reddit blockquotes
        text = self._RE_REDDIT_QUOTE.sub("", text)
        # 4. *emphasis* → bare word
        text = self._RE_ASTERISK_EMPH.sub(r"\1", text)
        # 5. @mentions → placeholder
        text = self._RE_MENTION.sub("TOKEN_MENTION", text)
        # 6. #hashtag → bare word
        text = self._RE_HASHTAG.sub(r"\1", text)
        # 7. Emojis
        text = self._RE_EMOJI.sub(" ", text)
        # 8. Repeated characters  (sooooo → so)
        text = self._RE_REPEATED_CHAR.sub(r"\1\1", text)
        # 9. Whitespace
        text = self._RE_EXTRA_WS.sub(" ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Step 2 – Standard text cleaning
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Apply standard surface cleaning.

        Operations (in order):
        1. Lowercase.
        2. Strip HTML tags.
        3. Remove standalone numbers.
        4. Remove non-alphabetic characters (punctuation etc.).
        5. Collapse extra whitespace.

        Note: call clean_social_media() *before* this method so that URL /
        mention / hashtag tokens have already been handled.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = text.lower()
        text = self._RE_HTML.sub(" ", text)
        text = self._RE_NUMBERS.sub(" ", text)
        text = self._RE_NON_ALPHA.sub(" ", text)
        text = self._RE_EXTRA_WS.sub(" ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Step 3 – Tokenise
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """Tokenise *text* using NLTK word_tokenize."""
        return word_tokenize(text)

    # ------------------------------------------------------------------
    # Step 4 – Stop-word removal
    # ------------------------------------------------------------------

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove English stopwords from *tokens*."""
        return [t for t in tokens if t not in _ENGLISH_STOPWORDS]

    # ------------------------------------------------------------------
    # Step 5 – Lemmatisation
    # ------------------------------------------------------------------

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply WordNet lemmatization to each token in *tokens*."""
        return [self._lemmatizer.lemmatize(t) for t in tokens]

    # ------------------------------------------------------------------
    # Internal length filter
    # ------------------------------------------------------------------

    def _filter_length(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if len(t) >= self.min_length]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """
        Run the complete preprocessing pipeline on *text*.

        Pipeline:
            clean_social_media
            → clean_text (lowercase, strip HTML/numbers/non-alpha)
            → tokenize
            → filter min_length
            → (remove stopwords)
            → (lemmatize)
            → join with spaces

        Parameters
        ----------
        text : str

        Returns
        -------
        str – cleaned, tokenised, re-joined text.
        """
        text   = self.clean_social_media(text)
        text   = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self._filter_length(tokens)
        if self.remove_stopwords_enabled:
            tokens = self.remove_stopwords(tokens)
        if self.lemmatize_enabled:
            tokens = self.lemmatize_tokens(tokens)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Apply preprocess() to every element of *texts* with progress output.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        list of str – preprocessed texts in the same order.
        """
        total        = len(texts)
        results: List[str] = []
        report_every = max(1, total // 10)

        for i, text in enumerate(texts):
            results.append(self.preprocess(text))
            if (i + 1) % report_every == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"  [preprocess] {i + 1:>6}/{total}  ({pct:.1f}%)", end="\r")

        print()   # newline after progress
        return results


# ---------------------------------------------------------------------------
# load_and_preprocess_data
# ---------------------------------------------------------------------------

def load_and_preprocess_data(
    filepath: str,
    text_col: str = "text",
    label_col: str = "risk_level",
    preprocessor: Optional[TextPreprocessor] = None,
) -> Tuple[List[str], np.ndarray, LabelEncoder]:
    """
    Load a CSV, preprocess text, and encode 3-class labels.

    Handles both string labels ("high_risk") and numeric labels (0/1/2).
    Enforces canonical label encoding: low_risk=0, moderate_risk=1, high_risk=2.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    text_col : str
        Name of the text column (default "text").
    label_col : str
        Name of the risk-level column (default "risk_level").
    preprocessor : TextPreprocessor or None
        Custom preprocessor.  If None, a default TextPreprocessor() is used.

    Returns
    -------
    X : list of str
        Preprocessed texts.
    y : np.ndarray of int64
        Encoded labels — low_risk=0, moderate_risk=1, high_risk=2.
    le : LabelEncoder
        Fitted encoder.  Use le.inverse_transform() to decode predictions.
    """
    print(f"[INFO] Loading data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[INFO] Raw shape: {df.shape}")
    print(f"[INFO] Columns  : {list(df.columns)}")

    # -- Validate required columns
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found.  Available: {list(df.columns)}"
            )

    # -- Keep only needed columns to reduce memory
    keep_cols = [text_col, label_col]
    for opt_col in ["source", "subreddit"]:
        if opt_col in df.columns:
            keep_cols.append(opt_col)

    df = df[keep_cols].dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]

    print(f"[INFO] After null/empty drop: {len(df):,} rows")

    # -- Normalise label strings (handle case, whitespace, dash vs underscore)
    label_series = (
        df[label_col].astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    # -- Coerce numeric labels and legacy binary labels
    legacy_map = {
        # numeric strings → canonical
        "0": "low_risk",
        "1": "moderate_risk",
        "2": "high_risk",
        # legacy binary
        "suicide":     "high_risk",
        "non_suicide": "low_risk",
        "non-suicide": "low_risk",
        "depression":  "moderate_risk",
        "normal":      "low_risk",
    }
    label_series = label_series.replace(legacy_map)
    df[label_col] = label_series

    # -- Build LabelEncoder with forced canonical order
    present_labels = df[label_col].unique().tolist()
    ordered_labels = [lbl for lbl in LABEL_ORDER if lbl in present_labels]
    # Append any unexpected labels at the end (so nothing crashes)
    ordered_labels += [lbl for lbl in present_labels if lbl not in ordered_labels]

    le = LabelEncoder()
    le.fit(ordered_labels)
    y = le.transform(df[label_col].values).astype(np.int64)

    print(f"\n[INFO] Label mapping (LabelEncoder):")
    for lbl, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {lbl:>20} → {int(code)}")

    # -- Class distribution
    print(f"\n[INFO] Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for code, cnt in zip(unique, counts):
        lbl = le.inverse_transform([int(code)])[0]
        pct = 100.0 * cnt / len(y)
        print(f"  {lbl:>20} : {cnt:>6,}  ({pct:.1f}%)")

    # -- Source distribution (if column present)
    if "source" in df.columns:
        print(f"\n[INFO] Source distribution:")
        for src, cnt in df["source"].value_counts().items():
            pct = 100.0 * cnt / len(df)
            print(f"  {str(src):>10} : {cnt:>6,}  ({pct:.1f}%)")

    # -- Preprocessing
    if preprocessor is None:
        preprocessor = TextPreprocessor()

    texts = df[text_col].astype(str).tolist()
    print(f"\n[INFO] Preprocessing {len(texts):,} texts ...")
    X = preprocessor.fit_transform(texts)

    avg_tokens = np.mean([len(t.split()) for t in X]) if X else 0.0
    print(f"\n[INFO] Dataset ready: {len(X):,} samples")
    print(f"[INFO] Avg tokens per sample (after preprocessing): {avg_tokens:.1f}")

    return X, y, le


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

def split_data(
    X: List[str],
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[
    List[str], List[str], List[str],
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Stratified train / validation / test split preserving all 3 class ratios.

    Parameters
    ----------
    X : list of str
    y : np.ndarray  (values 0, 1, 2)
    test_size : float
        Fraction of total data for test set (default 0.20).
    val_size : float
        Fraction of total data for validation set (default 0.10).
    random_state : int

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    # 1. Carve out test set (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2. Carve out validation set from remaining data (stratified)
    val_fraction_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    total = len(X)
    print(f"\n[INFO] Data split ({total:,} total):")
    print(f"  Train : {len(X_train):>6,}  ({100 * len(X_train) / total:.1f}%)")
    print(f"  Val   : {len(X_val):>6,}  ({100 * len(X_val) / total:.1f}%)")
    print(f"  Test  : {len(X_test):>6,}  ({100 * len(X_test) / total:.1f}%)")

    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        info = ", ".join(
            f"{INT_TO_LABEL.get(int(c), str(c))}={n}"
            for c, n in zip(unique, counts)
        )
        print(f"  {split_name} class counts: {info}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Compatibility helpers used by demo.py / evaluate.py / older scripts
# ---------------------------------------------------------------------------

def clean_text_single(text: str) -> str:
    """Preprocess a single text sample with the default pipeline."""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a dataset and normalise it to the current 3-class schema.
    Returns a dataframe with at least: text, label, risk_level.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    label_col = None
    for candidate in ("risk_level", "label", "class"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"No label column found in {filepath}. Available columns: {list(df.columns)}")

    text_col = "text" if "text" in df.columns else df.columns[0]
    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "risk_level"})

    mapping = {
        "low_risk": 0,
        "low": 0,
        "non-suicide": 0,
        "non_suicide": 0,
        "normal": 0,
        "moderate_risk": 1,
        "moderate": 1,
        "depression": 1,
        "high_risk": 2,
        "high": 2,
        "suicide": 2,
    }
    if df["risk_level"].dtype == object:
        df["risk_level"] = (
            df["risk_level"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True)
            .map(mapping)
        )

    df = df.dropna(subset=["risk_level"]).copy()
    df["label"] = df["risk_level"].astype(int)
    return df


def preprocess_dataframe(df: pd.DataFrame, for_transformer: bool = False) -> pd.DataFrame:
    """
    Normalise and clean a dataframe in a way that matches the rest of the codebase.
    """
    out = df.copy()
    out["text"] = out["text"].astype(str)
    if for_transformer:
        out["cleaned_text"] = out["text"].map(lambda t: expand_generation_language(t).strip())
    else:
        preprocessor = TextPreprocessor()
        out["cleaned_text"] = out["text"].map(preprocessor.preprocess)
    return out


def extract_tfidf_features(X_train, X_val, X_test, max_features: int = 50_000):
    """
    Backward-compatible TF-IDF helper.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(list(X_train))
    X_val_tfidf = vectorizer.transform(list(X_val))
    X_test_tfidf = vectorizer.transform(list(X_test))
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def extract_sentiment_features(texts):
    """
    Backward-compatible re-export so older callers can import from preprocess.py.
    """
    from src.features.sentiment_features import extract_sentiment_features as _extract
    return _extract(list(texts))


def generate_synthetic_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Backward-compatible synthetic dataframe generator returning text + label columns.
    """
    import random

    rng = random.Random(random_state)
    low_templates = [
        "Spent the afternoon with friends and felt calm for once.",
        "Coffee, music, and a quiet walk helped a lot today.",
        "Finished a project and I am proud of myself.",
    ]
    moderate_templates = [
        "I feel numb and disconnected from everyone lately.",
        "Everything feels heavy and I am struggling to get out of bed.",
        "I keep thinking I am a burden and nothing is getting better.",
    ]
    high_templates = [
        "I want to kill myself and I already have a plan.",
        "I do not want to be alive anymore and I have written goodbye messages.",
        "I have been thinking about ending my life tonight.",
    ]

    records = []
    for _ in range(max(1, n_samples // 3)):
        records.append({"text": rng.choice(low_templates), "label": 0, "risk_level": 0})
        records.append({"text": rng.choice(moderate_templates), "label": 1, "risk_level": 1})
        records.append({"text": rng.choice(high_templates), "label": 2, "risk_level": 2})

    return pd.DataFrame(records).sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _raw_dir       = os.path.join(_PROJECT_ROOT, "data", "raw")
    _synthetic_csv = os.path.join(_raw_dir, "synthetic_social_media_detection.csv")

    if not os.path.exists(_synthetic_csv):
        print("[INFO] Synthetic CSV not found — generating it now ...")
        _src_data = os.path.join(_PROJECT_ROOT, "src", "data")
        if _src_data not in sys.path:
            sys.path.insert(0, _src_data)
        from generate_synthetic import generate_dataset  # type: ignore
        generate_dataset(n_per_class=800)

    print("\n" + "=" * 60)
    print("DEMO: load_and_preprocess_data  (3-class)")
    print("=" * 60)

    X, y, le = load_and_preprocess_data(
        _synthetic_csv,
        text_col="text",
        label_col="risk_level",
    )

    print("\n" + "=" * 60)
    print("DEMO: split_data  (stratified 3-class)")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\n[INFO] First 3 training samples:")
    for text, label in zip(X_train[:3], y_train[:3]):
        decoded = le.inverse_transform([int(label)])[0]
        print(f"  [{decoded}] {text[:80]}")

    print("\n[INFO] Social-media cleaning demo:")
    _demo_texts = [
        "@JohnDoe RT this is a retweet #depressed #sad feeling low",
        "RT @user123: Great post about #mentalhealth awareness",
        "> quoted text in reddit\nI feel so empty and alone #worthless",
        "Just had my morning coffee ☕ #goodmorning feeling great!",
        "sooooooo tired of everything ugh cant do this anymore",
    ]
    _pp = TextPreprocessor(remove_stopwords=False, lemmatize=False)
    for raw in _demo_texts:
        cleaned_sm   = _pp.clean_social_media(raw)
        cleaned_full = _pp.preprocess(raw)
        print(f"  RAW  : {raw[:70]}")
        print(f"  SM   : {cleaned_sm[:70]}")
        print(f"  FULL : {cleaned_full[:70]}")
        print()

    print("[INFO] Demo complete.")
