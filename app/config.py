from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# Dataset files
UNIFIED_DATASET_PATH = PROCESSED_DIR / "email_dataset.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"

# Model artifacts
DEFAULT_MODEL_NAME = "logreg"
MODEL_PATH = ARTIFACTS_DIR / "email_classifier.joblib"
MODEL_REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF settings
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM_RANGE = (1, 2)

# Classifier hyperparameters
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
LOGREG_CLASS_WEIGHT = "balanced"
SGD_ALPHA = 1e-4
SGD_MAX_ITER = 2000

# Embedding model settings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

# Gmail polling settings
GMAIL_USER_ID = os.getenv("GMAIL_USER_ID", "me")
GMAIL_QUERY = os.getenv("GMAIL_QUERY", "in:inbox")
GMAIL_POLL_INTERVAL_MINUTES = int(os.getenv("GMAIL_POLL_INTERVAL_MINUTES", "30"))
GMAIL_CLIENT_SECRET_PATH = Path(
    os.getenv("GMAIL_CLIENT_SECRET_PATH", PROJECT_ROOT / "secrets" / "gmail_client_secret.json")
)
GMAIL_TOKEN_PATH = Path(os.getenv("GMAIL_TOKEN_PATH", PROJECT_ROOT / "secrets" / "gmail_token.json"))

# Transformer inference settings
TRANSFORMER_BASE_MODEL = os.getenv("TRANSFORMER_BASE_MODEL", "distilbert-base-uncased")
TRANSFORMER_MODEL_DIR = Path(os.getenv("TRANSFORMER_MODEL_DIR", ARTIFACTS_DIR / "distilbert"))
TRANSFORMER_CHECKPOINT_PATH = Path(
    os.getenv("TRANSFORMER_CHECKPOINT_PATH", ARTIFACTS_DIR / "distilbert" / "checkpoints" / "epoch_3.pt")
)
TRANSFORMER_MAX_LENGTH = int(os.getenv("TRANSFORMER_MAX_LENGTH", "256"))

# SQLite storage
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", PROCESSED_DIR / "emails.db"))

# Keyword lists for weak labeling and scoring
URGENT_KEYWORDS = [
    "urgent",
    "asap",
    "immediately",
    "today",
    "by eod",
    "deadline",
    "critical",
    "action required",
    "time-sensitive",
]
FOLLOWUP_KEYWORDS = [
    "following up",
    "gentle reminder",
    "just checking in",
    "any update",
    "circling back",
    "checking in",
    "reminder",
]
SPAM_KEYWORDS = [
    "free",
    "winner",
    "click here",
    "unsubscribe",
    "guaranteed",
    "limited time",
    "offer",
    "buy now",
    "work from home",
]
ACTION_PHRASES = [
    "please resolve",
    "need you to",
    "action required",
    "please review",
    "respond",
    "reply",
    "approve",
    "confirm",
]

# Weights for scoring
SCORE_WEIGHTS = {
    "urgent_base": 60,
    "follow_up_base": 40,
    "informational_base": 20,
    "spam_base": 2,
    "confidence_weight": 25,
    "urgency_keyword_weight": 6,
    "followup_keyword_weight": 4,
    "spam_keyword_penalty": 8,
    "deadline_bonus": 8,
    "reply_chain_bonus": 5,
    "sender_bonus": 10,
}

# Priority bands
PRIORITY_BANDS = {
    "high": 80,
    "medium": 50,
}

# Optional sender importance map
SENDER_IMPORTANCE = {
    "ceo@company.com": 1.0,
    "manager@company.com": 0.7,
}

# Reply prefixes
REPLY_PREFIXES = ("re:", "fwd:", "fw:")

# Logging
LOG_LEVEL = "INFO"

