# Email Classification and Priority Scoring System — Summary

## Overview
This project is a production-oriented NLP backend that ingests email content, classifies each message into one of four categories (`urgent`, `follow_up`, `spam`, `informational`), and computes a priority score (0–100). It includes dataset building with weak labeling, multiple baseline models, a transformer fine‑tuning pipeline, Gmail OAuth integration with polling, SQLite storage, and a web portal for monitoring.

## Methodology
- Combine real-world email corpora (Enron, TREC 2007, optional BC3) into a unified schema.
- Apply weak labeling rules to map messages to the four target classes.
- Train multiple models for comparison: TF‑IDF + linear classifiers and a transformer fine‑tune.
- Generate priority scores using a hybrid of model confidence and heuristic signals.
- Serve results via FastAPI, Gmail polling, and a lightweight portal.

## Data Pipeline
### Datasets
- Enron Email Dataset (general email content)
- TREC 2007 Public Spam Corpus (spam/ham signal)
- BC3 corpus (optional, follow‑up style threads)

### Download and Storage
- KaggleHub used for Enron and TREC 2007.
- Files are copied to:
  - `data/raw/enron/`
  - `data/raw/trec07/`
- Optional BC3 placed under `data/raw/bc3/`.

### Unified Schema
Each record is normalized to:
- `source_dataset`
- `subject`
- `body`
- `sender`
- `timestamp`
- `text` (merged subject + body)
- `original_label`
- `derived_label`
- `label_explanation`
- `label_source`

### Weak Labeling Rules
Rules live in `datasets/labeling_rules.py` and return:
- label
- explanation
- source

Rule logic:
- `spam`: from dataset labels or spam keyword hits
- `urgent`: urgency keywords and deadline phrases
- `follow_up`: reply prefixes and follow‑up phrases
- `informational`: fallback non‑spam, non‑urgent, non‑follow_up

## Preprocessing and Features
### Text Preprocessing
- Lowercasing
- Whitespace normalization
- Punctuation cleanup while preserving urgency markers
- Safe handling of missing values

### Structured Features
- urgency/follow‑up/spam keyword counts
- action phrase count
- exclamation count
- uppercase ratio
- subject length, body length
- sender importance
- reply‑prefix flag
- deadline flag

## Models
### Baseline Models (Comparative Study)
- TF‑IDF + LogisticRegression
- TF‑IDF + LinearSVC (calibrated)
- TF‑IDF + SGDClassifier (log‑loss)
- MiniLM embeddings + LogisticRegression

Artifacts:
- `models/artifacts/<model>.joblib`
- `models/artifacts/model_registry.json`
- `models/artifacts/model_comparison.csv`

### Transformer Fine‑Tuning
- DistilBERT fine‑tuning via `models/train_transformer.py`
- Defaults: 4 epochs, max length 256, batch size 16
- Checkpointing after each epoch
- Resume supported via `--resume`

Artifacts:
- `models/artifacts/distilbert/`
- `models/artifacts/distilbert/checkpoints/epoch_*.pt`
- `models/artifacts/distilbert/metrics.json`
- `models/artifacts/distilbert/loss_curve.png`
- `models/artifacts/distilbert/accuracy_curve.png`

## Priority Scoring
Priority score (0–100) combines:
- predicted class base weight
- model confidence
- urgency/follow‑up keyword intensity
- spam penalties
- deadline bonuses
- reply chain bonus
- sender importance

Output includes:
- `priority_score`
- `priority_band` (high/medium/low)
- `reasons` (explanations)
- `extracted_signals`

## Inference
### Scikit‑learn
- `models/predict.py` for baseline inference

### Transformer
- `models/transformer_predict.py` for transformer inference
- Loads base HF model + checkpoint weights
- Uses label map when available

## Gmail Integration (Polling)
### OAuth
- OAuth client secret JSON at `secrets/gmail_client_secret.json`
- Token stored at `secrets/gmail_token.json`
- Authorization via `python scripts/authorize_gmail.py`

### Polling
- Worker polls every `GMAIL_POLL_INTERVAL_MINUTES` (default 30)
- Query via `GMAIL_QUERY` (default `in:inbox`)
- Fetches new messages, runs transformer inference, stores results

### Storage
- SQLite at `data/processed/emails.db`
- Tables: `emails`, `state`

## API and Portal
### FastAPI Endpoints
- `GET /health`
- `GET /emails` (JSON list)
- `GET /portal` (glass UI)
- `POST /classify-email`
- `POST /classify-batch`

### Portal
- Uses `pexels-pixabay-33545.jpg` background
- Glassmorphism table
- Auto refresh by polling cadence

## Configuration (Environment)
Supported `.env` keys:
- `GMAIL_POLL_INTERVAL_MINUTES`
- `GMAIL_QUERY`
- `GMAIL_USER_ID`
- `GMAIL_CLIENT_SECRET_PATH`
- `GMAIL_TOKEN_PATH`
- `SQLITE_DB_PATH`
- `TRANSFORMER_BASE_MODEL`
- `TRANSFORMER_MODEL_DIR`
- `TRANSFORMER_CHECKPOINT_PATH`
- `TRANSFORMER_MAX_LENGTH`

## Results and Outputs
Artifacts are written to:
- `models/artifacts/` for baseline models and evaluation
- `models/artifacts/distilbert/` for transformer outputs
- `data/processed/emails.db` for live Gmail results

Metrics saved:
- `metrics_<model>.json`
- `classification_report_<model>.txt`
- `confusion_matrix_<model>.csv`
- `model_comparison.csv`

Transformer plots:
- `loss_curve.png`
- `accuracy_curve.png`

## Limitations
- Weak labels introduce noise in training targets
- Gmail polling is periodic, not real‑time
- Transformer inference depends on base model download unless fully saved locally

## Next Enhancements
- Add label map auto‑generation for clean class names
- Add real‑time Gmail push via Pub/Sub
- Add search and filtering in portal
- Swap to MiniLM / DistilBERT embeddings for faster inference

