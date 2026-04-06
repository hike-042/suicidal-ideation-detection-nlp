"""
download_data.py
----------------
Downloads and assembles a 3-class social media suicide ideation detection
dataset from multiple sources.

Three risk levels
-----------------
high_risk    (2) – explicit suicidal ideation / self-harm intent
                   Sources: r/SuicideWatch, Twitter #suicidal
moderate_risk (1) – depressive language, hopelessness, indirect signals
                   Sources: r/depression, r/mentalhealth
low_risk      (0) – normal social media posts / everyday conversation
                   Sources: r/teenagers, r/casualconversation, general tweets

Primary data sources
--------------------
1. Kaggle – nikhileswarkomati/suicide-watch
   Binary dataset (suicide / non-suicide).
   Remap:  suicide       → high_risk
           non-suicide with depression keywords → moderate_risk
           non-suicide otherwise                → low_risk

2. Kaggle (alternative) – suchintikasarkar/sentiment-analysis-for-mental-health
   Multi-label mental health dataset.
   Remap labels to the 3-class scheme (see _remap_mental_health_label).

3. PRAW / Reddit (fallback stub)
   If Kaggle is unavailable, a PRAW-based collection stub is provided.
   Requires:  pip install praw
   Credentials: set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
   environment variables or fill in the constants below.

4. Twitter (note)
   Academic API access is required for full-archive search.
   A tweepy-based stub is provided for keyword-based collection.
   Requires: pip install tweepy
   Credentials: set TWITTER_BEARER_TOKEN environment variable.

Output
------
data/raw/social_media_suicide_detection.csv
Columns: text, source (reddit/twitter), subreddit, risk_level
"""

import os
import re
import sys
import zipfile
import traceback
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
OUTPUT_CSV = os.path.join(RAW_DIR, "social_media_suicide_detection.csv")

KAGGLE_DATASET_PRIMARY = "nikhileswarkomati/suicide-watch"
KAGGLE_DATASET_ALT = "suchintikasarkar/sentiment-analysis-for-mental-health"

# Reddit / Twitter credentials (override via environment variables)
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "suicidal_ideation_scraper/0.1")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "YOUR_BEARER_TOKEN")

# ---------------------------------------------------------------------------
# Keyword lists for label remapping
# ---------------------------------------------------------------------------

# High-risk keywords used to up-grade ambiguous "non-suicide" posts
_HIGH_RISK_KEYWORDS = [
    "kill myself", "end my life", "end it all", "want to die",
    "commit suicide", "take my own life", "suicidal", "suicide note",
    "no reason to live", "planned how i", "written my note",
    "don't want to be alive", "wont be here", "won't be here",
    "attempting suicide", "overdose on purpose", "jump off",
]

# Moderate-risk keywords – depressive but no explicit suicidal plan
_MODERATE_RISK_KEYWORDS = [
    "feel empty", "so depressed", "can't go on", "worthless",
    "hopeless", "helpless", "feel like a burden", "feel nothing",
    "crying every day", "can't get out of bed", "haven't eaten",
    "nobody cares", "no one cares", "exhausted of living",
    "mental health", "anxiety attack", "panic attack", "self harm",
    "cutting myself", "feel so alone", "completely alone",
    "darkness", "numb inside", "pointless", "what's the point",
]


def _contains_keywords(text: str, keyword_list) -> bool:
    """Return True if *text* contains at least one keyword from *keyword_list*."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keyword_list)


# ---------------------------------------------------------------------------
# Mental-health multi-label → 3-class remap
# ---------------------------------------------------------------------------

# Maps labels from suchintikasarkar dataset to our 3 classes
_MENTAL_HEALTH_LABEL_MAP = {
    # high risk
    "suicidal": "high_risk",
    "suicide watch": "high_risk",
    "self harm": "high_risk",
    "crisis": "high_risk",
    # moderate risk
    "depression": "moderate_risk",
    "depressed": "moderate_risk",
    "anxiety": "moderate_risk",
    "ptsd": "moderate_risk",
    "bipolar": "moderate_risk",
    "mental illness": "moderate_risk",
    "stress": "moderate_risk",
    "ocd": "moderate_risk",
    "eating disorder": "moderate_risk",
    # low risk
    "normal": "low_risk",
    "casual": "low_risk",
    "positive": "low_risk",
    "happy": "low_risk",
    "neutral": "low_risk",
}


def _remap_mental_health_label(raw_label: str) -> str:
    """Map a raw mental-health dataset label to one of our 3 risk classes."""
    cleaned = raw_label.strip().lower()
    for key, mapped in _MENTAL_HEALTH_LABEL_MAP.items():
        if key in cleaned:
            return mapped
    # Default: if it looks positive-ish → low_risk, else moderate_risk
    return "low_risk" if any(w in cleaned for w in ["normal", "casual", "positive"]) else "moderate_risk"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_raw_dir() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[INFO] Raw data directory: {RAW_DIR}")


def _kaggle_available() -> bool:
    """Return True if the kaggle package is importable and credentials exist."""
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        print("[WARN] 'kaggle' package not installed. Run: pip install kaggle")
        return False
    except OSError as exc:
        print(f"[WARN] Kaggle credentials not found: {exc}")
        print("[WARN] Create ~/.kaggle/kaggle.json with your API key from kaggle.com/account")
        return False


def _find_csv(directory: str) -> Optional[str]:
    """Return the first CSV file found in *directory* (recursive)."""
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".csv"):
                return os.path.join(root, fname)
    return None


def _print_class_distribution(df: pd.DataFrame, label_col: str = "risk_level") -> None:
    print(f"\n[INFO] Class distribution ({len(df):,} total rows):")
    counts = df[label_col].value_counts()
    for label, cnt in counts.items():
        pct = 100.0 * cnt / len(df)
        print(f"  {label:>20} : {cnt:>6,}  ({pct:.1f}%)")


def _build_output_df(
    texts,
    risk_levels,
    sources=None,
    subreddits=None,
) -> pd.DataFrame:
    """Assemble a tidy DataFrame with the canonical output schema."""
    df = pd.DataFrame({
        "text": texts,
        "risk_level": risk_levels,
        "source": sources if sources is not None else ["unknown"] * len(texts),
        "subreddit": subreddits if subreddits is not None else [None] * len(texts),
    })
    df = df.dropna(subset=["text", "risk_level"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df["risk_level"] = df["risk_level"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy 1 – Primary Kaggle dataset (nikhileswarkomati/suicide-watch)
# ---------------------------------------------------------------------------

def _remap_suicide_watch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Re-map the binary suicide-watch dataset to our 3-class scheme.

    suicide     → high_risk
    non-suicide → moderate_risk  (if depressive/crisis keywords found)
                  low_risk       (otherwise)
    """
    records = []
    for _, row in df_raw.iterrows():
        text = str(row.get("text", row.iloc[0]))
        raw_class = str(row.get("class", row.iloc[1])).strip().lower()

        if "suicide" in raw_class and "non" not in raw_class:
            risk = "high_risk"
        elif _contains_keywords(text, _HIGH_RISK_KEYWORDS):
            risk = "high_risk"
        elif _contains_keywords(text, _MODERATE_RISK_KEYWORDS):
            risk = "moderate_risk"
        else:
            risk = "low_risk"

        records.append({
            "text": text,
            "risk_level": risk,
            "source": "reddit",
            "subreddit": "r/SuicideWatch" if risk == "high_risk" else None,
        })
    return pd.DataFrame(records)


def download_via_kaggle_primary() -> bool:
    """Download and process the primary Kaggle suicide-watch dataset."""
    print("[INFO] Attempting Kaggle primary dataset (suicide-watch) …")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET_PRIMARY,
            path=RAW_DIR,
            unzip=False,
            quiet=False,
        )

        zip_name = KAGGLE_DATASET_PRIMARY.split("/")[1] + ".zip"
        zip_path = os.path.join(RAW_DIR, zip_name)
        if not os.path.exists(zip_path):
            zips = [f for f in os.listdir(RAW_DIR) if f.endswith(".zip")]
            if zips:
                zip_path = os.path.join(RAW_DIR, zips[0])
            else:
                raise FileNotFoundError("Downloaded zip not found.")

        print(f"[INFO] Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        os.remove(zip_path)

        csv_path = _find_csv(RAW_DIR)
        if csv_path is None:
            raise FileNotFoundError("No CSV found after extraction.")

        df_raw = pd.read_csv(csv_path)
        print(f"[INFO] Raw dataset: {df_raw.shape}  columns: {list(df_raw.columns)}")

        # Normalise column names
        col_map = {}
        for col in df_raw.columns:
            low = col.lower().strip()
            if low in ("text", "post", "content", "message"):
                col_map[col] = "text"
            elif low in ("class", "label", "category", "target"):
                col_map[col] = "class"
        df_raw = df_raw.rename(columns=col_map)
        if "text" not in df_raw.columns or "class" not in df_raw.columns:
            df_raw.columns = ["text", "class"] + list(df_raw.columns[2:])

        df = _remap_suicide_watch(df_raw)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[INFO] Saved {len(df):,} rows to {OUTPUT_CSV}")
        _print_class_distribution(df)
        return True

    except Exception as exc:
        print(f"[ERROR] Kaggle primary download failed: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Strategy 2 – Alternative Kaggle dataset (suchintikasarkar/sentiment-analysis-for-mental-health)
# ---------------------------------------------------------------------------

def download_via_kaggle_alt() -> bool:
    """Download and process the alternative Kaggle mental-health dataset."""
    print("[INFO] Attempting Kaggle alternative dataset (sentiment-analysis-for-mental-health) …")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET_ALT,
            path=RAW_DIR,
            unzip=False,
            quiet=False,
        )

        zip_name = KAGGLE_DATASET_ALT.split("/")[1] + ".zip"
        zip_path = os.path.join(RAW_DIR, zip_name)
        if not os.path.exists(zip_path):
            zips = [f for f in os.listdir(RAW_DIR) if f.endswith(".zip")]
            if zips:
                zip_path = os.path.join(RAW_DIR, zips[0])
            else:
                raise FileNotFoundError("Downloaded zip not found.")

        print(f"[INFO] Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        os.remove(zip_path)

        csv_path = _find_csv(RAW_DIR)
        if csv_path is None:
            raise FileNotFoundError("No CSV found after extraction.")

        df_raw = pd.read_csv(csv_path)
        print(f"[INFO] Raw dataset: {df_raw.shape}  columns: {list(df_raw.columns)}")

        # Identify text and label columns (flexible)
        text_col = None
        label_col = None
        for col in df_raw.columns:
            low = col.lower().strip()
            if low in ("text", "statement", "post", "content", "message") and text_col is None:
                text_col = col
            elif low in ("status", "label", "category", "class", "mental_health_condition") and label_col is None:
                label_col = col

        if text_col is None or label_col is None:
            # Positional fallback
            text_col = df_raw.columns[0]
            label_col = df_raw.columns[1]
            print(f"[WARN] Columns not identified by name; using '{text_col}' and '{label_col}'.")

        records = []
        for _, row in df_raw.iterrows():
            text = str(row[text_col])
            raw_label = str(row[label_col])
            risk = _remap_mental_health_label(raw_label)
            records.append({
                "text": text,
                "risk_level": risk,
                "source": "reddit",
                "subreddit": None,
            })

        df = pd.DataFrame(records)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[INFO] Saved {len(df):,} rows to {OUTPUT_CSV}")
        _print_class_distribution(df)
        return True

    except Exception as exc:
        print(f"[ERROR] Kaggle alternative download failed: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Strategy 3 – PRAW Reddit fallback stub
# ---------------------------------------------------------------------------

def collect_reddit_via_praw(
    posts_per_subreddit: int = 500,
    save: bool = True,
) -> bool:
    """
    Collect posts from Reddit using PRAW (Python Reddit API Wrapper).

    Prerequisites
    -------------
    1. Install:  pip install praw
    2. Create a Reddit app at https://www.reddit.com/prefs/apps
       (select "script" type).
    3. Set environment variables:
           REDDIT_CLIENT_ID
           REDDIT_CLIENT_SECRET
           REDDIT_USER_AGENT

    Subreddit → risk_level mapping
    --------------------------------
    r/SuicideWatch   → high_risk
    r/depression     → moderate_risk
    r/mentalhealth   → moderate_risk
    r/teenagers      → low_risk
    r/casualconversation → low_risk

    Parameters
    ----------
    posts_per_subreddit : int
        How many hot/new posts to pull from each subreddit.
    save : bool
        If True, append results to OUTPUT_CSV.

    Returns
    -------
    bool – True on success.
    """
    print("[INFO] Attempting Reddit data collection via PRAW …")
    try:
        import praw  # type: ignore
    except ImportError:
        print("[ERROR] 'praw' package not installed. Run: pip install praw")
        print("[INFO]  Manual instructions:")
        print("        1. pip install praw")
        print("        2. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT env vars")
        print("        3. Re-run this script")
        return False

    if "YOUR_CLIENT_ID" in REDDIT_CLIENT_ID:
        print("[ERROR] Reddit credentials not configured.")
        print("        Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.")
        return False

    subreddit_config = {
        "SuicideWatch":        "high_risk",
        "depression":          "moderate_risk",
        "mentalhealth":        "moderate_risk",
        "teenagers":           "low_risk",
        "casualconversation":  "low_risk",
    }

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        read_only=True,
    )

    records = []
    for sub_name, risk_level in subreddit_config.items():
        print(f"[INFO] Fetching r/{sub_name} ({risk_level}) …")
        try:
            subreddit = reddit.subreddit(sub_name)
            posts_seen = 0
            for submission in subreddit.hot(limit=posts_per_subreddit * 2):
                body = (submission.selftext or "").strip()
                title = (submission.title or "").strip()
                text = f"{title}. {body}" if body else title
                if not text or text in ("[deleted]", "[removed]"):
                    continue
                records.append({
                    "text": text,
                    "risk_level": risk_level,
                    "source": "reddit",
                    "subreddit": f"r/{sub_name}",
                })
                posts_seen += 1
                if posts_seen >= posts_per_subreddit:
                    break
            print(f"  Collected {posts_seen} posts from r/{sub_name}")
        except Exception as exc:
            print(f"  [WARN] Failed to fetch r/{sub_name}: {exc}")

    if not records:
        print("[ERROR] No Reddit posts collected.")
        return False

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    if save:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[INFO] Saved {len(df):,} rows to {OUTPUT_CSV}")
        _print_class_distribution(df)

    return True


# ---------------------------------------------------------------------------
# Twitter stub (Academic API)
# ---------------------------------------------------------------------------

def collect_twitter_via_tweepy(
    max_results_per_query: int = 100,
) -> bool:
    """
    Collect tweets using tweepy's Academic Research Product Track API.

    NOTE: Twitter/X Academic API access must be applied for separately at
          https://developer.twitter.com/en/products/twitter-api/academic-research
          Standard/Basic tiers do NOT support historical full-archive search.

    Prerequisites
    -------------
    1. Install:  pip install tweepy
    2. Set environment variable:  TWITTER_BEARER_TOKEN

    Query → risk_level mapping
    --------------------------
    "#suicidal OR kill myself OR end my life"  → high_risk
    "#depressed OR feeling hopeless OR worthless OR self harm" → moderate_risk
    "#happy OR good morning OR weekend plans" → low_risk

    Parameters
    ----------
    max_results_per_query : int
        Number of tweets to fetch per search query (10–500 for Academic tier).

    Returns
    -------
    bool – True on success (always False in stub if credentials are missing).
    """
    print("[INFO] Attempting Twitter data collection via Tweepy …")
    try:
        import tweepy  # type: ignore
    except ImportError:
        print("[ERROR] 'tweepy' not installed. Run: pip install tweepy")
        return False

    if "YOUR_BEARER_TOKEN" in TWITTER_BEARER_TOKEN:
        print("[ERROR] Twitter bearer token not configured.")
        print("        Set the TWITTER_BEARER_TOKEN environment variable.")
        return False

    query_config = [
        (
            "(#suicidal OR \"kill myself\" OR \"end my life\" OR \"want to die\") "
            "lang:en -is:retweet",
            "high_risk",
        ),
        (
            "(#depressed OR \"feeling hopeless\" OR \"feel worthless\" OR \"self harm\") "
            "lang:en -is:retweet",
            "moderate_risk",
        ),
        (
            "(#happy OR #goodmorning OR \"weekend plans\" OR \"just had coffee\") "
            "lang:en -is:retweet",
            "low_risk",
        ),
    ]

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

    records = []
    for query, risk_level in query_config:
        print(f"[INFO] Searching Twitter: {risk_level} …")
        try:
            response = client.search_recent_tweets(
                query=query,
                max_results=min(max_results_per_query, 100),
                tweet_fields=["text", "created_at", "author_id"],
            )
            if response.data:
                for tweet in response.data:
                    records.append({
                        "text": tweet.text,
                        "risk_level": risk_level,
                        "source": "twitter",
                        "subreddit": None,
                    })
                print(f"  Collected {len(response.data)} tweets for {risk_level}")
            else:
                print(f"  No tweets returned for {risk_level}")
        except Exception as exc:
            print(f"  [WARN] Twitter query failed: {exc}")

    if not records:
        print("[ERROR] No tweets collected.")
        return False

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    # Merge with existing data if file already present
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset="text")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved {len(df):,} rows to {OUTPUT_CSV}")
    _print_class_distribution(df)
    return True


# ---------------------------------------------------------------------------
# Strategy 4 – Synthetic fallback
# ---------------------------------------------------------------------------

def download_via_synthetic() -> bool:
    """Generate a 3-class synthetic dataset as a last resort."""
    print("[INFO] Falling back to synthetic dataset generation …")
    try:
        try:
            from src.data.generate_synthetic import generate_dataset  # type: ignore
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from generate_synthetic import generate_dataset  # type: ignore

        records = generate_dataset(n_per_class=800, output_path=OUTPUT_CSV)
        df = pd.DataFrame(records)
        print(f"[INFO] Synthetic 3-class dataset saved to {OUTPUT_CSV} ({len(df):,} rows).")
        _print_class_distribution(df)
        return True

    except Exception as exc:
        print(f"[ERROR] Synthetic generation failed: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_raw_dir()

    if os.path.exists(OUTPUT_CSV):
        print(f"[INFO] Dataset already exists at {OUTPUT_CSV}.")
        print("[INFO] Delete it to re-download / regenerate.")
        df = pd.read_csv(OUTPUT_CSV)
        _print_class_distribution(df)
        return

    strategies = []

    if _kaggle_available():
        strategies.append(("Kaggle primary (suicide-watch)", download_via_kaggle_primary))
        strategies.append(("Kaggle alternative (mental-health)", download_via_kaggle_alt))

    strategies += [
        ("PRAW Reddit collector", collect_reddit_via_praw),
        ("Synthetic 3-class generation", download_via_synthetic),
    ]

    for name, fn in strategies:
        print(f"\n{'='*60}")
        print(f"[INFO] Strategy: {name}")
        print(f"{'='*60}")
        if fn():
            print(f"\n[SUCCESS] Dataset ready at: {OUTPUT_CSV}")
            return

    print("\n[FATAL] All strategies failed.")
    print("Manual options:")
    print(f"  1. Download from https://www.kaggle.com/datasets/{KAGGLE_DATASET_PRIMARY}")
    print(f"     and place the CSV at: {OUTPUT_CSV}")
    print("  2. Set up PRAW credentials and re-run with collect_reddit_via_praw()")
    print("  3. Set TWITTER_BEARER_TOKEN and re-run with collect_twitter_via_tweepy()")
    sys.exit(1)


if __name__ == "__main__":
    main()
