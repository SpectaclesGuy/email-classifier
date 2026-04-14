# Project Details — Email Classification and Priority Scoring System

**Aim And Functionality**
- Build a production-oriented backend that ingests emails, classifies them into `urgent`, `follow_up`, `spam`, or `informational`, and assigns a 0–100 priority score.
- Provide an API plus a lightweight portal to view prioritized emails.
- Support Gmail OAuth polling to pull real inbox data at a configurable interval.

**High-Level Flow**
1. Datasets are downloaded (Enron, TREC 2007, optional BC3) and normalized into a single schema.
2. Weak-labeling rules map raw labels and keyword signals to the four target classes.
3. Models are trained (TF-IDF baselines, optional MiniLM embeddings, optional transformer fine-tune).
4. Inference produces a class + confidence and feeds a priority scoring module.
5. Gmail polling runs inference on new messages and stores results in SQLite.
6. FastAPI exposes `/classify-email`, `/classify-batch`, `/emails`, and `/portal`.

**Core Concepts (Project-Level)**
- Email triage and priority scoring: classification + heuristic scoring to rank messages.
- Weak labeling: deriving target labels from imperfect real-world datasets using transparent rules.
- Hybrid intelligence: combining ML predictions with rule-based signals for robust prioritization.
- End-to-end ML pipeline: data prep → training → evaluation → inference → serving.
- Explainability: storing reasons and extracted signals alongside predictions.

**Technical Stack And Components**
- Backend API: FastAPI + Uvicorn (`api/main.py`).
- Data processing: Pandas, NumPy, scikit-learn.
- ML artifacts: joblib pipelines + model registry (`models/artifacts/`).
- Transformers: Hugging Face `transformers` + PyTorch (optional fine-tuning and inference).
- Embeddings: `sentence-transformers` MiniLM (optional baseline).
- Gmail integration: Google OAuth + Gmail API (`integrations/`).
- Storage: SQLite with JSON-encoded explanation and signals (`storage/db.py`).
- Configuration: `.env` + `python-dotenv` (`app/config.py`).
- Portal UI: server-rendered HTML with static background asset.

**Backend Concepts (System Design)**
- REST endpoints with input validation using Pydantic schemas (`app/schemas.py`).
- Background worker thread for polling Gmail (`services/gmail_worker.py`).
- OAuth token lifecycle handling and refresh logic (`integrations/gmail_auth.py`).
- Data persistence with schema-driven tables and idempotent inserts (`storage/db.py`).
- Separation of concerns: API, services, integrations, models, datasets.
- Configurable runtime behavior via environment variables (polling interval, query, paths).

**Machine Learning Concepts**
- Text preprocessing: normalization, case-folding, punctuation cleanup (`app/utils.py`).
- Feature engineering: urgency/follow-up/spam keyword counts, exclamations, uppercase ratio, reply prefix, sender importance, and deadline flags.
- TF-IDF vectorization with n-grams + structured features (`models/train.py`).
- Linear classifiers: Logistic Regression, Linear SVC (calibrated), SGD (log-loss).
- Embedding-based baseline: MiniLM sentence embeddings + Logistic Regression.
- Transformer fine-tuning: DistilBERT with checkpointing, metrics, and plots (`models/train_transformer.py`).
- Confidence estimation: `predict_proba` or calibrated decision scores (`models/predict.py`).
- Evaluation: accuracy + macro F1, saved reports and comparison CSVs (`models/evaluate.py`).

**Priority Scoring Logic**
- Combines predicted class base weights, model confidence, keyword intensity, spam penalties, deadline bonuses, reply-chain bonuses, and sender importance.
- Produces a numeric score and a band (`high`, `medium`, `low`) with textual reasons.
- Designed to be tunable via weights in `app/config.py`.

**What Makes The Project Unique**
- Transparent weak-labeling strategy that includes explanations and sources for each label.
- Hybrid scoring: ML prediction + interpretable, business-driven signals.
- Dual-path ML: fast TF-IDF baselines and optional transformer fine-tuning for higher accuracy.
- Real inbox integration with Gmail polling and a live portal for operational use.
- Model registry + metadata for reproducible selection of the best model.

**Future Scope**
- Real-time Gmail push via Pub/Sub instead of polling.
- Active learning loop: capture user feedback to refine labels and scoring.
- Personalization: per-user sender importance and priority policies.
- Model monitoring: drift detection and periodic retraining automation.
- Multi-tenant support with separate SQLite or managed databases (PostgreSQL).
- Enhanced explainability with SHAP/LIME-style explanations for ML outputs.

**Key Files And Responsibilities**
- `api/main.py`: API endpoints, portal, and worker lifecycle.
- `datasets/build_dataset.py`: dataset normalization, weak labeling, train/val split.
- `datasets/labeling_rules.py`: heuristic label mapping rules.
- `models/train.py`: baseline training pipelines and registry.
- `models/train_transformer.py`: transformer fine-tuning + checkpoints.
- `models/transformer_predict.py`: transformer inference pipeline.
- `models/scoring.py`: priority scoring and banding logic.
- `services/gmail_polling.py`: Gmail fetch + inference + persistence.
- `storage/db.py`: SQLite schema and persistence helpers.
