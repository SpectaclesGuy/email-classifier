# Email Classification and Priority Scoring System

Production-oriented NLP backend for classifying incoming emails into `urgent`, `follow_up`, `spam`, or `informational` and computing a 0-100 priority score.

## Project Purpose
- Inbox triage for faster response handling
- Helpdesk or internal business inbox prioritization
- Extensible for future Gmail/Outlook integration

## Gmail Integration (Polling)
This build supports Gmail OAuth + polling every 30 minutes (configurable).

### Setup
1. Create a Google Cloud project and enable Gmail API.
2. Create OAuth client credentials (Desktop App) and download `client_secret.json`.
3. Place the file at:
```
secrets/gmail_client_secret.json
```
4. Authorize once:
```
python scripts/authorize_gmail.py
```
This will create:
```
secrets/gmail_token.json
```

### Configure Polling
Defaults can be overridden via env vars (see `.env.example`):
- `GMAIL_POLL_INTERVAL_MINUTES` (default 30)
- `GMAIL_QUERY` (default `in:inbox`)
- `GMAIL_USER_ID` (default `me`)
- `SQLITE_DB_PATH` (default `data/processed/emails.db`)

### Run API + Poller
```
uvicorn api.main:app --reload
```
Visit:
- Portal: `http://127.0.0.1:8000/portal`
- JSON feed: `http://127.0.0.1:8000/emails`

## Dataset Sources (KaggleHub)
This project supports KaggleHub downloads for Enron and TREC 2007.

Required Kaggle credentials:
- Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment.

Download datasets:
```
python datasets/download_datasets.py
```

This will populate:
```
data/raw/enron/
data/raw/trec07/
```

If you also use BC3, place it manually under `data/raw/bc3/` as CSV.

## Label Mapping Strategy
Public datasets do not perfectly match the target labels. This project uses a transparent weak-labeling approach:
- `spam`: direct spam labels from TREC or spam keyword matches
- `follow_up`: reply prefixes and follow-up phrases
- `urgent`: urgency keywords and deadline phrases
- `informational`: fallback non-spam, non-urgent, non-follow-up

Label rules live in `datasets/labeling_rules.py` and return both a label and an explanation.

## Setup
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Build Dataset
```
python datasets/build_dataset.py
```

## Train Models (Comparative Study)
This project trains multiple baseline models for comparison:
- `logreg`
- `linear_svc` (calibrated)
- `sgd` (log-loss)
- `minilm` (GPU-capable embeddings + LogisticRegression)

Run training:
```
python models/train.py
```

This writes:
- `models/artifacts/model_registry.json`
- `models/artifacts/logreg.joblib`
- `models/artifacts/linear_svc.joblib`
- `models/artifacts/sgd.joblib`
- `models/artifacts/minilm.joblib`
- `models/artifacts/email_classifier.joblib` (default)

## GPU Transformer Training (Option 2)
For full GPU training (fine-tuning a transformer classifier):

1. Install CUDA-enabled PyTorch from the official selector:
```
https://pytorch.org/get-started/locally/
```

2. Run the transformer trainer (defaults to 4 epochs):
```
python models/train_transformer.py --model distilbert-base-uncased --epochs 4 --batch-size 16 --max-length 256 --max-samples 100000
```

3. Resume from latest checkpoint:
```
python models/train_transformer.py --resume
```

Outputs:
- Model + tokenizer: `models/artifacts/distilbert/`
- Checkpoints: `models/artifacts/distilbert/checkpoints/`
- Metrics: `models/artifacts/distilbert/metrics.json`
- Plots: `models/artifacts/distilbert/loss_curve.png`, `models/artifacts/distilbert/accuracy_curve.png`

## Transformer Inference
Run inference using the fine-tuned transformer:
```
python scripts/run_transformer_inference.py
```

Or use the module directly:
```
python models/transformer_predict.py
```

## Evaluate Models
```
python models/evaluate.py
```

Evaluation outputs:
- `models/artifacts/metrics_<model>.json`
- `models/artifacts/classification_report_<model>.txt`
- `models/artifacts/confusion_matrix_<model>.csv`
- `models/artifacts/model_comparison.csv`

## Run API
```
uvicorn api.main:app --reload
```

## Example Request
```
curl -X POST http://127.0.0.1:8000/classify-email \
  -H "Content-Type: application/json" \
  -d '{"subject":"Urgent: client issue needs resolution today","body":"Please resolve this before EOD. The client is waiting.","sender":"manager@company.com","timestamp":"2026-04-06T10:30:00"}'
```

## Output
Model artifacts and evaluation results are stored under `models/artifacts/`.

## Limitations
- Weak labels can be noisy; results are intended as a baseline.
- Urgent and follow-up detection depends on keyword heuristics.
- Some datasets require manual normalization before use.

## Next Improvements
- Replace TF-IDF with transformer embeddings (MiniLM, DistilBERT)
- Add active learning and user-specific priority rules
- Integrate Gmail/Outlook ingestion pipeline

