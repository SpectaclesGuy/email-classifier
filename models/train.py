from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import logging

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from app.config import (
    ARTIFACTS_DIR,
    DEFAULT_MODEL_NAME,
    LOGREG_C,
    LOGREG_CLASS_WEIGHT,
    LOGREG_MAX_ITER,
    METADATA_PATH,
    MODEL_PATH,
    MODEL_REGISTRY_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    SGD_ALPHA,
    SGD_MAX_ITER,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from app.logging_config import setup_logging
from app.utils import extract_structured_features, get_structured_feature_names, preprocess_text
from models.embedding import EmbeddingTransformer, is_embedding_available

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


MODEL_NAMES: List[str] = ["logreg", "linear_svc", "sgd", "minilm"]


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_training_data() -> pd.DataFrame:
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Training data not found. Run datasets/build_dataset.py first.")
    return pd.read_csv(train_path, low_memory=False)


def load_validation_data() -> pd.DataFrame | None:
    val_path = PROCESSED_DIR / "val.csv"
    if not val_path.exists():
        return None
    return pd.read_csv(val_path, low_memory=False)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ["text", "subject", "body", "sender"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "has_reply_prefix" in df.columns:
        df["has_reply_prefix"] = df["has_reply_prefix"].fillna(False)
    return df


def build_tfidf_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=TFIDF_MIN_DF,
                    max_df=TFIDF_MAX_DF,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    preprocessor=preprocess_text,
                ),
                "text",
            ),
            (
                "struct",
                FunctionTransformer(extract_structured_features, validate=False),
                ["subject", "body", "sender", "has_reply_prefix"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_embedding_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "emb",
                EmbeddingTransformer(),
                "text",
            ),
            (
                "struct",
                FunctionTransformer(extract_structured_features, validate=False),
                ["subject", "body", "sender", "has_reply_prefix"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimator(name: str):
    if name == "logreg":
        return LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            class_weight=LOGREG_CLASS_WEIGHT,
            n_jobs=1,
        )
    if name == "linear_svc":
        base = LinearSVC(class_weight=LOGREG_CLASS_WEIGHT)
        return CalibratedClassifierCV(estimator=base, cv=3)
    if name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            alpha=SGD_ALPHA,
            max_iter=SGD_MAX_ITER,
            class_weight=LOGREG_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        )
    if name == "minilm":
        return LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            class_weight=LOGREG_CLASS_WEIGHT,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model name: {name}")


def build_pipeline(name: str) -> Pipeline:
    if name == "minilm":
        preprocessor = build_embedding_preprocessor()
    else:
        preprocessor = build_tfidf_preprocessor()
    estimator = build_estimator(name)
    return Pipeline([("features", preprocessor), ("clf", estimator)])


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    df = _clean_dataframe(load_training_data())
    val_df = load_validation_data()
    val_df = _clean_dataframe(val_df) if val_df is not None else None

    X_train = df[["text", "subject", "body", "sender", "has_reply_prefix"]]
    y_train = df["derived_label"]

    X_val = None
    y_val = None
    if val_df is not None and not val_df.empty:
        X_val = val_df[["text", "subject", "body", "sender", "has_reply_prefix"]]
        y_val = val_df["derived_label"]

    logger.info("Training samples: %s", len(df))
    if val_df is not None:
        logger.info("Validation samples: %s", len(val_df))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    registry: Dict[str, Dict[str, object]] = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "models": {},
        "best_model": DEFAULT_MODEL_NAME,
    }

    best_f1 = -1.0
    for name in _progress(MODEL_NAMES, desc="Training models", unit="model"):
        if name == "minilm" and not is_embedding_available():
            logger.warning("Skipping MiniLM model: sentence-transformers not installed.")
            continue

        logger.info("Training model: %s", name)
        pipeline = build_pipeline(name)
        pipeline.fit(X_train, y_train)

        model_path = ARTIFACTS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

        if name == DEFAULT_MODEL_NAME:
            joblib.dump(pipeline, MODEL_PATH)

        metrics = {}
        if X_val is not None and y_val is not None:
            preds = pipeline.predict(X_val)
            metrics = {
                "accuracy": float(accuracy_score(y_val, preds)),
                "macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
            }
            logger.info("%s metrics: %s", name, metrics)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                registry["best_model"] = name

        registry["models"][name] = {
            "path": str(model_path),
            "metrics": metrics,
        }

    metadata = {
        "labels": sorted(y_train.unique().tolist()),
        "structured_features": get_structured_feature_names(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    logger.info("Saved model registry to %s", MODEL_REGISTRY_PATH)


if __name__ == "__main__":
    main()

