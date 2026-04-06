# Project Audit

## What Was Checked

- top-level CLI scripts: `train.py`, `evaluate.py`, `demo.py`, `run.py`
- data and preprocessing modules under `src/data/`
- feature extraction modules under `src/features/`
- model code under `src/models/`
- evaluation and visualisation modules under `src/evaluation/` and `src/visualization/`
- FastAPI app, routes, agents, frontend assets under `app/`

## Issues Found

1. The repository mixed an older binary-classification layout with a newer 3-class risk pipeline.
2. Several scripts imported modules that no longer existed:
   - `src.data.dataset`
   - `src.models.lstm`
   - `src.models.text_cnn`
   - `src.training.trainer`
   - `src.training.transformer_trainer`
3. `src/data/preprocess.py` had a runtime bug where the `remove_stopwords` instance attribute shadowed the method with the same name.
4. `demo.py` still assumed binary `SUICIDE` vs `NON-SUICIDE` outputs.
5. `evaluate.py` still expected older helper functions and signatures.
6. Visualisation helpers assumed mostly binary labels and older function signatures.
7. Frontend helper `app/static/js/app.js` expected top-level risk fields while the current API returns nested `classification` fields.

## Fixes Applied

- added generation-aware slang expansion in `src/data/generation_lexicon.py`
- repaired preprocessing and added backward-compatible helper functions in `src/data/preprocess.py`
- added compatibility wrappers for dataset/model/training imports
- updated `demo.py` to the 3-class label scheme
- rewrote `evaluate.py` around the current 3-class pipeline
- extended plotting helpers to handle current calls and added separate per-risk word clouds
- updated `train.py` to save class-distribution, text-length, and per-risk word cloud plots
- updated `app/static/js/app.js` to consume the current API response shape
- refreshed `README.md` to match the current architecture
- added a deterministic multi-angle signal engine in `app/agents/signal_engine.py`
- integrated system-level signal synthesis into the website backend and frontend

## Remaining Risks

- the FastAPI app still depends on Anthropic availability for the full agent path
- no runtime verification was possible in this environment because Python was not available in `PATH`
- the project still contains some legacy files and wording that could be simplified further in a cleanup pass

## System Angles

The website now inspects text from multiple angles before synthesizing the final risk view:

- harm to others intent
- explicit intent
- planning / preparation
- finality / farewell language
- self-harm references
- hopelessness / meaninglessness
- burden / worthlessness
- isolation / disconnection
- emotional dysregulation
- help-seeking
- future orientation / protective language

This makes the site more of a structured risk-analysis system than a plain one-shot labeler.

## Expanded Classification Paths

- `LOW_RISK`
- `MODERATE_RISK`
- `HIGH_RISK_SELF_HARM`
- `HIGH_RISK_HARM_TO_OTHERS`

This prevents violent threats toward others from being incorrectly collapsed into the old self-harm-only taxonomy.
