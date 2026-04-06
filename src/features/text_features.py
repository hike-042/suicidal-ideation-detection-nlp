"""
text_features.py
----------------
Feature extraction classes for the Suicidal Ideation Detection project.

Classes
-------
TFIDFFeatures        – TF-IDF sparse matrix features
BagOfWordsFeatures   – Count-based sparse matrix features
Word2VecFeatures     – Dense document embeddings via gensim Word2Vec

Function
--------
get_combined_features – horizontally concatenate TF-IDF + W2V matrices
"""

import os
import pickle
import warnings
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# Optional gensim import (Word2Vec)
# ---------------------------------------------------------------------------
try:
    from gensim.models import Word2Vec  # type: ignore
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False
    warnings.warn(
        "gensim is not installed. Word2VecFeatures will raise ImportError on use. "
        "Install with: pip install gensim",
        ImportWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# TFIDFFeatures
# ---------------------------------------------------------------------------

class TFIDFFeatures:
    """
    TF-IDF sparse feature matrix.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.
    sublinear_tf : bool
        Apply sublinear TF scaling (log(1 + tf)).
    min_df : int or float
        Minimum document frequency for a term to be included.
    max_df : float
        Maximum document frequency fraction (removes corpus-wide terms).
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, train_texts: List[str]) -> sp.csr_matrix:
        """
        Fit the vectorizer on *train_texts* and return the transformed matrix.

        Parameters
        ----------
        train_texts : list of str

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_samples, vocab_size)
        """
        print(f"[TFIDFFeatures] Fitting on {len(train_texts):,} documents …")
        matrix = self._vectorizer.fit_transform(train_texts)
        self._fitted = True
        print(f"[TFIDFFeatures] Vocabulary size: {len(self._vectorizer.vocabulary_):,}")
        print(f"[TFIDFFeatures] Matrix shape: {matrix.shape}")
        return matrix

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """Transform *texts* using the already-fitted vectorizer."""
        self._assert_fitted()
        return self._vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return the list of feature (token) names."""
        self._assert_fitted()
        return self._vectorizer.get_feature_names_out().tolist()

    def save(self, path: str) -> None:
        """Pickle the fitted vectorizer to *path*."""
        self._assert_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._vectorizer, fh)
        print(f"[TFIDFFeatures] Saved to {path}")

    def load(self, path: str) -> "TFIDFFeatures":
        """Load a previously saved vectorizer from *path* (in-place)."""
        with open(path, "rb") as fh:
            self._vectorizer = pickle.load(fh)
        self._fitted = True
        print(f"[TFIDFFeatures] Loaded from {path}")
        return self

    # ------------------------------------------------------------------
    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TFIDFFeatures is not fitted yet. Call fit_transform() first.")


# ---------------------------------------------------------------------------
# BagOfWordsFeatures
# ---------------------------------------------------------------------------

class BagOfWordsFeatures:
    """
    Bag-of-Words (raw count) sparse feature matrix.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.
    min_df : int
        Minimum document frequency.
    max_df : float
        Maximum document frequency fraction.
    binary : bool
        If True, all non-zero counts are set to 1 (presence matrix).
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        binary: bool = False,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary

        self._vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, train_texts: List[str]) -> sp.csr_matrix:
        """Fit and transform training texts."""
        print(f"[BagOfWordsFeatures] Fitting on {len(train_texts):,} documents …")
        matrix = self._vectorizer.fit_transform(train_texts)
        self._fitted = True
        print(f"[BagOfWordsFeatures] Vocabulary size: {len(self._vectorizer.vocabulary_):,}")
        print(f"[BagOfWordsFeatures] Matrix shape: {matrix.shape}")
        return matrix

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """Transform *texts* using the fitted vectorizer."""
        self._assert_fitted()
        return self._vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names."""
        self._assert_fitted()
        return self._vectorizer.get_feature_names_out().tolist()

    def save(self, path: str) -> None:
        """Pickle the fitted vectorizer to *path*."""
        self._assert_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._vectorizer, fh)
        print(f"[BagOfWordsFeatures] Saved to {path}")

    def load(self, path: str) -> "BagOfWordsFeatures":
        """Load a previously saved vectorizer."""
        with open(path, "rb") as fh:
            self._vectorizer = pickle.load(fh)
        self._fitted = True
        print(f"[BagOfWordsFeatures] Loaded from {path}")
        return self

    # ------------------------------------------------------------------
    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BagOfWordsFeatures is not fitted yet. Call fit_transform() first."
            )


# ---------------------------------------------------------------------------
# Word2VecFeatures
# ---------------------------------------------------------------------------

class Word2VecFeatures:
    """
    Dense document embeddings produced by averaging word vectors from a
    gensim Word2Vec model trained on the corpus.

    Parameters
    ----------
    vector_size : int
        Dimensionality of word vectors.
    window : int
        Context window size.
    min_count : int
        Ignore words with total frequency lower than this.
    workers : int
        Number of training threads.
    epochs : int
        Number of training passes over the corpus.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ) -> None:
        if not _GENSIM_AVAILABLE:
            raise ImportError(
                "gensim is required for Word2VecFeatures. "
                "Install with: pip install gensim"
            )
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self._model: Optional["Word2Vec"] = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> "Word2VecFeatures":
        """
        Train a Word2Vec model on the tokenised *texts*.

        Parameters
        ----------
        texts : list of str
            Pre-tokenized or raw strings (will be split on whitespace).

        Returns
        -------
        self
        """
        if not _GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed.")

        sentences = [t.split() for t in texts]
        print(
            f"[Word2VecFeatures] Training Word2Vec on {len(sentences):,} documents "
            f"(vector_size={self.vector_size}, window={self.window}, "
            f"min_count={self.min_count}, epochs={self.epochs}) …"
        )
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=42,
        )
        vocab_size = len(self._model.wv)
        print(f"[Word2VecFeatures] Training complete. Vocabulary: {vocab_size:,} words.")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Produce document embeddings by averaging the word vectors for each text.

        Words not in the vocabulary are ignored.  Documents with no known
        words receive a zero vector.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray of shape (n_samples, vector_size)
        """
        self._assert_fitted()
        wv = self._model.wv  # type: ignore[union-attr]

        embeddings = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.split()
            known = [t for t in tokens if t in wv]
            if known:
                embeddings[i] = np.mean(wv[known], axis=0)

        print(f"[Word2VecFeatures] Embedded {len(texts):,} documents → shape {embeddings.shape}")
        return embeddings

    def save(self, path: str) -> None:
        """Save the Word2Vec model to *path* (gensim native format)."""
        self._assert_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._model.save(path)  # type: ignore[union-attr]
        print(f"[Word2VecFeatures] Saved to {path}")

    def load(self, path: str) -> "Word2VecFeatures":
        """Load a previously saved Word2Vec model."""
        if not _GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed.")
        self._model = Word2Vec.load(path)
        print(f"[Word2VecFeatures] Loaded from {path}")
        return self

    # ------------------------------------------------------------------
    def _assert_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Word2VecFeatures model is not trained yet. Call fit() first."
            )


# ---------------------------------------------------------------------------
# get_combined_features
# ---------------------------------------------------------------------------

def get_combined_features(
    tfidf_features: sp.csr_matrix,
    w2v_features: np.ndarray,
) -> sp.csr_matrix:
    """
    Horizontally concatenate a sparse TF-IDF matrix with dense Word2Vec
    embeddings into a single sparse matrix.

    Parameters
    ----------
    tfidf_features : scipy.sparse.csr_matrix
        Shape (n_samples, n_tfidf_features).
    w2v_features : np.ndarray
        Shape (n_samples, vector_size).

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_samples, n_tfidf_features + vector_size)
    """
    if tfidf_features.shape[0] != w2v_features.shape[0]:
        raise ValueError(
            f"Row count mismatch: TF-IDF has {tfidf_features.shape[0]} rows, "
            f"W2V has {w2v_features.shape[0]} rows."
        )

    w2v_sparse = sp.csr_matrix(w2v_features)
    combined = sp.hstack([tfidf_features, w2v_sparse], format="csr")
    print(
        f"[get_combined_features] Combined shape: {combined.shape} "
        f"(TF-IDF: {tfidf_features.shape[1]} + W2V: {w2v_features.shape[1]})"
    )
    return combined


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sample_texts = [
        "i feel hopeless and worthless nothing matters anymore",
        "today was a wonderful day i went hiking with friends",
        "i cant stop thinking about ending my life",
        "just baked a cake and watched a movie feeling great",
        "nobody cares about me i want to disappear",
        "my job is going well and i love my family",
    ] * 20  # repeat to pass min_count thresholds

    print("=" * 60)
    print("Smoke test: TFIDFFeatures")
    print("=" * 60)
    tfidf = TFIDFFeatures(max_features=500, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(_sample_texts)
    print(f"TF-IDF matrix: {X_tfidf.shape}")

    print("\n" + "=" * 60)
    print("Smoke test: BagOfWordsFeatures")
    print("=" * 60)
    bow = BagOfWordsFeatures(max_features=500)
    X_bow = bow.fit_transform(_sample_texts)
    print(f"BoW matrix: {X_bow.shape}")

    if _GENSIM_AVAILABLE:
        print("\n" + "=" * 60)
        print("Smoke test: Word2VecFeatures")
        print("=" * 60)
        w2v = Word2VecFeatures(vector_size=50, window=3, min_count=1, epochs=5)
        w2v.fit(_sample_texts)
        X_w2v = w2v.transform(_sample_texts)
        print(f"W2V embeddings: {X_w2v.shape}")

        print("\n" + "=" * 60)
        print("Smoke test: get_combined_features")
        print("=" * 60)
        X_combined = get_combined_features(X_tfidf, X_w2v)
        print(f"Combined: {X_combined.shape}")
    else:
        print("\n[SKIP] gensim not installed — Word2Vec tests skipped.")

    print("\n[INFO] All smoke tests passed.")
