# Suicidal Ideation Detection using NLP & Machine Learning

> **IMPORTANT DISCLAIMER**  
> This project is intended **solely for research and educational purposes**.  
> It is **not** a clinical diagnostic tool and must **never** replace professional mental health assessment.  
> If you or someone you know is in crisis, please contact a mental health professional or a crisis hotline immediately.
>
> - **US:** 988 Suicide & Crisis Lifeline — call or text **988**  
> - **US:** Crisis Text Line — text **HOME** to **741741**  
> - **International resources:** https://www.iasp.info/resources/Crisis_Centres/

---

## Description

This project builds a text classification system that distinguishes posts expressing suicidal ideation from non-suicidal content. Using a dataset sourced from Reddit (r/SuicideWatch vs r/teenagers), the system explores multiple NLP and machine learning approaches:

- **Classical ML** (TF-IDF + Logistic Regression, SVM, Random Forest, Gradient Boosting)
- **Deep Learning** (LSTM, TextCNN with learned word embeddings)
- **Transformer** (DistilBERT fine-tuning)

The goal is to evaluate whether automated NLP methods can reliably flag potentially at-risk content, and to understand the strengths and limitations of each approach, particularly in safety-critical contexts where false negatives carry real-world consequences.

## Current Architecture

The repository now has two major layers:

- A risk pipeline that now distinguishes:
  - `LOW_RISK`
  - `MODERATE_RISK`
  - `HIGH_RISK_SELF_HARM`
  - `HIGH_RISK_HARM_TO_OTHERS`
- A FastAPI web app (`app/`) that runs a tiered agent workflow: keyword prefilter -> Haiku unified analysis -> Sonnet escalation for ambiguous or high-risk cases -> motivational support layer.
- A deterministic signal-analysis layer that reviews the text from multiple angles and synthesizes a system-level risk assessment.

Recent updates also add:

- generation-aware slang expansion for social-media English such as acronyms, meme phrasing, and youth shorthand
- compatibility wrappers so `train.py`, `evaluate.py`, and `demo.py` use the same preprocessing and label schema
- separate risk-specific word clouds and exploratory plots saved under `outputs/plots/`
- structured signal detection for explicit intent, planning, finality, self-harm, hopelessness, burden, isolation, dysregulation, and protective/help-seeking cues

## Quick Start

### Train on synthetic data

```bash
python train.py --synthetic --models classical
```

This now generates:

- class distribution plot
- text-length distribution plot
- separate word clouds for Low / Moderate / High risk text
- model metrics and saved artefacts

### Run the web app

```bash
python run.py --reload
```

If `ANTHROPIC_API_KEY` is not set, the app can still use its built-in keyword/fallback path for local testing.

### Website default: free OpenRouter

The website backend now defaults to an OpenRouter-only free setup:

- primary provider: OpenRouter
- fallback provider: OpenRouter
- default model: `openrouter/free`
- if you later add Anthropic, you can still override the provider order with env vars

Environment variables:

```bash
OPENROUTER_API_KEY=...
LLM_PRIMARY_PROVIDER=openrouter
LLM_FALLBACK_PROVIDER=openrouter
OPENROUTER_FAST_MODEL=openrouter/free
OPENROUTER_SMART_MODEL=openrouter/free
```

Start the website with:

```bash
python run.py --reload
```

To avoid pasting environment variables every time:

1. Copy `.env.example` to `.env`
2. Put your real `OPENROUTER_API_KEY` into `.env`
3. Run `python run.py --reload`

`run.py` now loads `.env` automatically if the file exists.
### Benchmark the active website system

You can benchmark the current four-class website pipeline with the included labeled benchmark set of 200+ examples covering:

- indirect self-harm
- vague threats
- sarcasm
- quotes, song, gaming, and fantasy language
- gen-z and slang variants
- vulgar but harmless text
- planning and finality language

Run the local heuristic benchmark with:

```bash
python benchmark_system.py --mode fallback
```

This evaluates:

- `LOW_RISK`
- `MODERATE_RISK`
- `HIGH_RISK_SELF_HARM`
- `HIGH_RISK_HARM_TO_OTHERS`

Outputs are saved under `outputs/benchmark/` as:

- benchmark JSON report
- per-example prediction CSV

The benchmark runner also reports:

- per-class recall / precision / F1
- category-level accuracy
- common failure slices by category
- mismatch examples for manual review

If your API keys are configured and you want to benchmark the full website path:

```bash
python benchmark_system.py --mode orchestrator
```

Optional Anthropic override:

```bash
ANTHROPIC_API_KEY=...
LLM_PRIMARY_PROVIDER=anthropic
LLM_FALLBACK_PROVIDER=openrouter
```

---

## Project Structure

```
Suicidal Ideation/
│
├── config.py                    # Centralised configuration
├── train.py                     # Main training CLI script
├── evaluate.py                  # Evaluation CLI script
├── demo.py                      # Interactive command-line demo
├── setup_nltk.py                # Download required NLTK corpora
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py        # Text cleaning, tokenisation, feature extraction
│   │   └── dataset.py           # Vocabulary, PyTorch Dataset, DataLoader builders
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py         # Sklearn wrapper for multiple classical classifiers
│   │   ├── lstm.py              # Bidirectional LSTM classifier
│   │   ├── text_cnn.py          # TextCNN classifier
│   │   └── transformer.py       # DistilBERT fine-tuning wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # PyTorch training loop (LSTM / CNN)
│   │   └── transformer_trainer.py  # HuggingFace-based training loop
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Accuracy, F1, AUC-ROC, classification reports
│   └── visualization/
│       ├── __init__.py
│       └── plots.py             # Confusion matrices, ROC curves, feature importance
│
├── data/
│   ├── raw/
│   │   └── suicide_detection.csv   # Kaggle dataset (download separately)
│   └── processed/                  # Auto-generated preprocessed artefacts
│
├── outputs/
│   ├── models/                  # Saved model weights and vectorisers
│   ├── plots/                   # Training curves, confusion matrices, ROC curves
│   └── results/                 # JSON metrics reports
│
└── notebooks/
    └── exploration.ipynb        # EDA and experimental notebooks
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/suicidal-ideation-detection.git
cd suicidal-ideation-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python setup_nltk.py
```

### 5. Kaggle API setup (for the real dataset)

1. Go to https://www.kaggle.com/account and click **Create New API Token**.  
2. Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows).  
3. Download the dataset:

```bash
kaggle datasets download -d nikhileswarkomati/suicide-watch
unzip suicide-watch.zip -d data/raw/
```

The CSV should end up at `data/raw/suicide_detection.csv`.

---

## Usage

### Quick start — synthetic data (no download required)

```bash
python train.py --synthetic --models classical
```

### Train all models on the real dataset

```bash
python train.py --data data/raw/suicide_detection.csv --models all
```

### Train only specific model groups

```bash
# Classical ML only
python train.py --data data/raw/suicide_detection.csv --models classical

# Deep learning (LSTM + TextCNN)
python train.py --data data/raw/suicide_detection.csv --models lstm --epochs 15

# Transformer (DistilBERT)
python train.py --data data/raw/suicide_detection.csv --models transformer --epochs 3
```

### Advanced training options

```bash
python train.py \
    --data data/raw/suicide_detection.csv \
    --models all \
    --epochs 10 \
    --batch-size 64 \
    --max-features 50000 \
    --output-dir outputs/ \
    --no-gpu
```

### Run the interactive demo

```bash
python demo.py
```

The demo loads the best available saved model automatically. If no model has been trained yet, it trains a fast logistic regression on synthetic data first.

### Evaluate a saved model

```bash
# Evaluate all classical models
python evaluate.py --data data/raw/suicide_detection.csv --model-type classical

# Evaluate a specific classical model
python evaluate.py \
    --data data/raw/suicide_detection.csv \
    --model-type classical \
    --model-name "Logistic Regression"

# Evaluate LSTM / TextCNN
python evaluate.py --data data/raw/suicide_detection.csv --model-type lstm

# Evaluate DistilBERT
python evaluate.py --data data/raw/suicide_detection.csv --model-type transformer
```

---

## Models

### Logistic Regression
A linear classifier trained on TF-IDF bag-of-words features (up to 50 000 unigrams and bigrams) combined with VADER sentiment scores. Fast to train, highly interpretable via feature coefficients, and surprisingly competitive as a baseline.

### Support Vector Machine (SVM)
A linear SVM (`LinearSVC`) operating on the same TF-IDF + sentiment feature matrix. SVMs are known to generalise well on high-dimensional sparse text data and often outperform Logistic Regression at the cost of slightly less probabilistic output.

### Random Forest
An ensemble of decision trees trained on TF-IDF features. Provides feature importance rankings useful for understanding which terms most distinguish the two classes. Slower to train than linear models but naturally handles non-linear interactions.

### Gradient Boosting
`HistGradientBoostingClassifier` from scikit-learn, offering strong performance without manual hyperparameter sensitivity. Operates on a dense sub-sample of TF-IDF features.

### LSTM (Bidirectional)
A two-layer bidirectional LSTM with learned 128-dimensional word embeddings. Processes each post sequentially, capturing long-range dependencies and contextual meaning that bag-of-words models miss. Trained with Adam optimiser and early stopping.

### TextCNN
A convolutional neural network applied over word embeddings using multiple filter widths (2, 3, 4 tokens). Effectively captures key phrases and local n-gram patterns. Trains faster than LSTM and often achieves comparable accuracy.

### DistilBERT
A distilled version of BERT pre-trained on large text corpora and fine-tuned here for binary classification. Captures deep contextual semantics through the full transformer attention mechanism. The highest-performing model but requires the most compute.

---

## Dataset

The dataset is sourced from Kaggle:  
**[Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)**

| Property | Value |
|---|---|
| Source | Reddit |
| Positive class | r/SuicideWatch posts (`suicide`) |
| Negative class | r/teenagers posts (`non-suicide`) |
| Approximate size | ~232,000 posts |
| Format | CSV with columns: `text`, `class` |
| Language | English |

The two subreddits represent a meaningful distinction: r/SuicideWatch is a support community where users frequently discuss suicidal thoughts, while r/teenagers is a general-interest community. This framing makes the task tractable but also introduces caveats — ironic or hyperbolic usage in the non-suicide class, and the absence of subtle, ambiguous cases.

---

## Results

The following table will be populated after running `train.py`. Values shown are representative benchmarks from the literature and initial experiments.

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | ~0.930 | ~0.930 | ~0.977 |
| SVM (Linear) | ~0.935 | ~0.935 | ~0.978 |
| Random Forest | ~0.910 | ~0.909 | ~0.960 |
| Gradient Boosting | ~0.920 | ~0.920 | ~0.968 |
| LSTM (Bidirectional) | ~0.945 | ~0.945 | ~0.983 |
| TextCNN | ~0.942 | ~0.942 | ~0.981 |
| DistilBERT (fine-tuned) | ~0.962 | ~0.962 | ~0.991 |

*Exact numbers depend on your hardware, random seed, and dataset version. Run `train.py` to reproduce.*

---

## Ethical Considerations

This project engages with sensitive mental health data and requires careful ethical attention:

1. **Data provenance and consent.** Reddit posts are public, but authors did not consent to use of their content in machine learning research. Researchers should review the platform's terms of service and consider the ethical implications of using mental health disclosures.

2. **False negatives are dangerous.** In a safety-critical context, failing to flag a post expressing suicidal ideation (false negative) is far more harmful than a false positive. Evaluation should weight recall on the positive class heavily, and any deployment must involve human review.

3. **Bias and generalisation.** The model is trained on Reddit users, a specific demographic. Performance on other populations, platforms, or languages is unknown and likely degraded. The model should not be applied outside its training distribution without rigorous re-evaluation.

4. **Not a diagnostic tool.** Automated text classification cannot diagnose suicidal ideation. Risk assessment requires clinical training, contextual knowledge, and ongoing human judgement. This system should be treated as a triage aid at most, never as a standalone decision-maker.

5. **Potential for harm from misuse.** A deployed system used to surveil individuals without their knowledge or consent raises serious privacy and autonomy concerns. Any deployment must be transparent, consensual where possible, and subject to oversight.

6. **Researcher wellbeing.** Working with datasets containing expressions of suicidal ideation can be emotionally taxing. Researchers should be aware of this and seek support if needed.

---

## Crisis Resources

If you or someone you know needs help:

| Resource | Contact |
|---|---|
| US — 988 Suicide & Crisis Lifeline | Call or text **988** |
| US — Crisis Text Line | Text **HOME** to **741741** |
| UK — Samaritans | Call **116 123** (free, 24/7) |
| Canada — Crisis Services Canada | Call **1-833-456-4566** |
| Australia — Lifeline | Call **13 11 14** |
| International directory | https://www.iasp.info/resources/Crisis_Centres/ |
| Emergency services | 911 (US) / 999 (UK) / 112 (EU) |

---

## License

This repository is released for research and educational use only. See `LICENSE` for details.



