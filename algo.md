# Algorithms And ML Workflow — Email Classification And Priority Scoring

**Purpose**
This document describes the scoring algorithm and the complete ML training workflow used in the project, with concrete examples for each stage.

**Scoring Algorithm (Priority Score 0–100)**
The score combines the predicted label, model confidence, and extracted signals from the email text and metadata. It is bounded to 0–100 and mapped to priority bands.

**Inputs**
- `predicted_label`: one of `urgent`, `follow_up`, `informational`, `spam`.
- `confidence`: model confidence in the prediction (0–1).
- `signals`: extracted features from subject/body/sender and reply indicators.

**Signals Extracted**
- `urgency_keyword_count`
- `followup_keyword_count`
- `spam_keyword_count`
- `action_phrase_count`
- `exclamation_count`
- `uppercase_ratio`
- `subject_length`
- `body_length`
- `sender_importance`
- `has_reply_prefix`
- `deadline_flag`

**Weights And Thresholds**
- Base weights:
- `urgent_base = 60`
- `follow_up_base = 40`
- `informational_base = 20`
- `spam_base = 2`
- Confidence weight: `confidence_weight = 25`
- Signal weights:
- `urgency_keyword_weight = 6`
- `followup_keyword_weight = 4`
- `spam_keyword_penalty = 8`
- `deadline_bonus = 8`
- `reply_chain_bonus = 5`
- `sender_bonus = 10`
- Priority bands:
- `high >= 80`
- `medium >= 50`
- otherwise `low`

**Algorithm Steps**
1. Start score at 0.
2. Add base weight for the predicted label.
3. Add `confidence * confidence_weight`.
4. Add or subtract signal contributions.
5. Clamp score to `[0, 100]`.
6. Assign priority band based on thresholds.
7. Return score, band, and reasons.

**Scoring Formula (Readable Form)**
- `score = base(predicted_label)`
- `score += confidence * 25`
- `score += urgency_keyword_count * 6`
- `score += followup_keyword_count * 4`
- `score -= spam_keyword_count * 8`
- `score += deadline_flag * 8`
- `score += has_reply_prefix * 5`
- `score += sender_importance * 10`
- `score = clamp(score, 0, 100)`
- `band = high if score >= 80; medium if score >= 50; else low`

**Scoring Example 1 (Urgent + Deadline)**
Input:
- `predicted_label = urgent`
- `confidence = 0.92`
- `urgency_keyword_count = 2`
- `followup_keyword_count = 0`
- `spam_keyword_count = 0`
- `deadline_flag = 1`
- `has_reply_prefix = 1`
- `sender_importance = 0.7`

Calculation:
- Base: `60`
- Confidence: `0.92 * 25 = 23.0`
- Urgency keywords: `2 * 6 = 12`
- Deadline: `1 * 8 = 8`
- Reply chain: `1 * 5 = 5`
- Sender importance: `0.7 * 10 = 7`
- Total: `60 + 23 + 12 + 8 + 5 + 7 = 115`
- Clamped: `100`
- Band: `high`

**Scoring Example 2 (Follow-Up With Mild Confidence)**
Input:
- `predicted_label = follow_up`
- `confidence = 0.55`
- `followup_keyword_count = 1`
- `deadline_flag = 0`
- `has_reply_prefix = 1`
- `sender_importance = 0`

Calculation:
- Base: `40`
- Confidence: `0.55 * 25 = 13.75`
- Follow-up keywords: `1 * 4 = 4`
- Reply chain: `1 * 5 = 5`
- Total: `40 + 13.75 + 4 + 5 = 62.75`
- Band: `medium`

**Scoring Example 3 (Spam With Keyword Penalty)**
Input:
- `predicted_label = spam`
- `confidence = 0.80`
- `spam_keyword_count = 3`
- `deadline_flag = 0`
- `has_reply_prefix = 0`

Calculation:
- Base: `2`
- Confidence: `0.80 * 25 = 20`
- Spam penalty: `3 * 8 = 24`
- Total: `2 + 20 - 24 = -2`
- Clamped: `0`
- Band: `low`

**Complete ML Training And Task Pipeline**
This section lists every ML step implemented in the project.

**1. Dataset Acquisition**
- Sources: Enron, TREC 2007, optional BC3.
- Download Enron and TREC using KaggleHub.

Example command:
```bash
python datasets/download_datasets.py
```

**2. Dataset Normalization**
- Columns are mapped to a unified schema.
- Missing columns are created and filled.
- Text is merged into a `text` field (`subject + body`).
- Reply prefix is detected (`re:`, `fwd:`).

Unified schema fields:
- `source_dataset`, `subject`, `body`, `sender`, `timestamp`, `text`, `original_label`, `derived_label`, `label_explanation`, `label_source`, `has_reply_prefix`.

**Example (Unified Record)**
```json
{
  "source_dataset": "trec07",
  "subject": "Re: Action required",
  "body": "Please resolve this by EOD.",
  "sender": "manager@company.com",
  "timestamp": null,
  "original_label": "ham",
  "has_reply_prefix": true,
  "text": "Re: Action required\nPlease resolve this by EOD."
}
```

**3. Weak Labeling Rules**
- Spam if original label indicates spam or spam keywords match.
- Urgent if urgency keywords match.
- Follow-up if reply prefix or follow-up phrases match.
- Informational otherwise.

**Labeling Example 1 (Spam)**
- Subject: "Limited time offer"
- Body: "Click here to claim your prize"
- Derived label: `spam`

**Labeling Example 2 (Urgent)**
- Subject: "Urgent: client issue"
- Body: "Resolve immediately"
- Derived label: `urgent`

**Labeling Example 3 (Follow-Up)**
- Subject: "Re: project update"
- Body: "Just checking in"
- Derived label: `follow_up`

**Labeling Example 4 (Informational)**
- Subject: "Team meeting notes"
- Body: "Minutes attached"
- Derived label: `informational`

**4. Feature Engineering**
- Text preprocessing: lowercasing, whitespace normalization, punctuation cleanup.
- Structured features: counts of urgency/follow-up/spam keywords, action phrases, exclamation count, uppercase ratio, subject/body length, sender importance, reply prefix, deadline flag.

**Feature Example**
Input:
- Subject: "Urgent: release approval"
- Body: "Please approve by EOD!!!"

Output signals (illustrative):
- `urgency_keyword_count = 1`
- `action_phrase_count = 1`
- `exclamation_count = 3`
- `deadline_flag = 1`
- `has_reply_prefix = 0`

**5. Model Training (Baselines)**
- TF-IDF with n-grams (1–2) + structured features.
- Classifiers: Logistic Regression, Linear SVC (calibrated), SGD (log-loss).

Example command:
```bash
python models/train.py
```

Artifacts created:
- `models/artifacts/logreg.joblib`
- `models/artifacts/linear_svc.joblib`
- `models/artifacts/sgd.joblib`
- `models/artifacts/email_classifier.joblib`
- `models/artifacts/model_registry.json`
- `models/artifacts/metadata.json`

**6. Embedding-Based Baseline (MiniLM)**
- Sentence-Transformers embeddings + Logistic Regression.
- Uses GPU if available.

Example command:
```bash
python models/train.py
```

Result:
- `models/artifacts/minilm.joblib` (if `sentence-transformers` is installed)

**7. Transformer Fine-Tuning**
- DistilBERT fine-tune with PyTorch.
- Checkpoints saved each epoch with metrics and plots.

Example command:
```bash
python models/train_transformer.py --model distilbert-base-uncased --epochs 4 --batch-size 16 --max-length 256
```

Artifacts created:
- `models/artifacts/distilbert/`
- `models/artifacts/distilbert/checkpoints/epoch_*.pt`
- `models/artifacts/distilbert/metrics.json`
- `models/artifacts/distilbert/loss_curve.png`
- `models/artifacts/distilbert/accuracy_curve.png`

**8. Evaluation**
- Accuracy, macro precision/recall/F1, classification report, confusion matrix.

Example command:
```bash
python models/evaluate.py
```

Artifacts created:
- `models/artifacts/metrics_<model>.json`
- `models/artifacts/classification_report_<model>.txt`
- `models/artifacts/confusion_matrix_<model>.csv`
- `models/artifacts/model_comparison.csv`

**9. Inference**
- Baseline inference uses the trained scikit-learn pipeline.
- Transformer inference loads fine-tuned checkpoint if available.

**Inference Example (API Payload)**
```json
{
  "subject": "Urgent: client issue needs resolution today",
  "body": "Please resolve this before EOD. The client is waiting.",
  "sender": "manager@company.com",
  "timestamp": "2026-04-06T10:30:00"
}
```

**Inference Example (Expected Output Shape)**
```json
{
  "predicted_category": "urgent",
  "confidence_score": 0.93,
  "priority_score": 100,
  "priority_band": "high",
  "explanation": [
    "Predicted class is urgent",
    "High model confidence",
    "Contains 2 urgency indicators",
    "Contains deadline-related phrase"
  ],
  "extracted_signals": {
    "urgency_keyword_count": 2,
    "followup_keyword_count": 0,
    "spam_keyword_count": 0,
    "action_phrase_count": 1,
    "exclamation_count": 0,
    "uppercase_ratio": 0.0,
    "subject_length": 43,
    "body_length": 62,
    "sender_importance": 0.7,
    "has_reply_prefix": 0,
    "deadline_flag": 1
  }
}
```

**10. Gmail Polling ML Loop**
- Gmail API fetches new messages on a schedule.
- Each message is decoded, inferred, scored, and stored in SQLite.

Example polling flow:
- Query constructed from `GMAIL_QUERY` and `last_checked_ts`.
- Fetch message IDs.
- For each message: decode body → predict → score → insert into DB.

**End-To-End Example (Minimal Local Run)**
```bash
python datasets/build_dataset.py
python models/train.py
uvicorn api.main:app --reload
```

**Where The Logic Lives**
- Scoring: `models/scoring.py`
- Feature extraction: `app/utils.py`
- Weak labeling: `datasets/labeling_rules.py`
- Training: `models/train.py`, `models/train_transformer.py`
- Evaluation: `models/evaluate.py`
- Inference: `models/predict.py`, `models/transformer_predict.py`
