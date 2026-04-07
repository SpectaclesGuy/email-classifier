from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


import json
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from app.config import DEFAULT_MODEL_NAME, MODEL_PATH, MODEL_REGISTRY_PATH
from app.utils import merge_subject_body
from models.scoring import score_email
from models.embedding import EmbeddingTransformer

_MODEL_CACHE: Dict[str, object] = {}


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum(axis=-1, keepdims=True)


def _predict_with_confidence(model, X: pd.DataFrame) -> Tuple[List[str], List[float]]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = model.classes_[np.argmax(probs, axis=1)]
        confs = np.max(probs, axis=1)
        return preds.tolist(), confs.tolist()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        probs = _softmax(scores)
        preds = model.classes_[np.argmax(probs, axis=1)]
        confs = np.max(probs, axis=1)
        return preds.tolist(), confs.tolist()

    preds = model.predict(X)
    return preds.tolist(), [0.5 for _ in preds]


def _load_registry() -> Dict[str, Any]:
    if MODEL_REGISTRY_PATH.exists():
        return json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"models": {}, "best_model": DEFAULT_MODEL_NAME}


def _resolve_model_path(model_name: str | None) -> Path:
    if model_name:
        candidate = Path(model_name)
        if candidate.exists():
            return candidate
        return Path("models") / "artifacts" / f"{model_name}.joblib"

    if MODEL_REGISTRY_PATH.exists():
        registry = _load_registry()
        best = registry.get("best_model", DEFAULT_MODEL_NAME)
        models = registry.get("models", {})
        if best in models:
            return Path(models[best]["path"])

    return MODEL_PATH


def _load_model(model_path: Path):
    key = str(model_path)
    if key not in _MODEL_CACHE:
        if not model_path.exists():
            raise FileNotFoundError("Model artifact not found. Run models/train.py first.")
        _MODEL_CACHE[key] = joblib.load(model_path)
    return _MODEL_CACHE[key]


def predict_email(
    subject: str,
    body: str,
    sender: str | None = None,
    timestamp: str | None = None,
    thread_id: str | None = None,
    has_reply_prefix: bool | None = None,
    model_name: str | None = None,
) -> Dict[str, Any]:
    model_path = _resolve_model_path(model_name)
    model = _load_model(model_path)
    text = merge_subject_body(subject, body)

    X = pd.DataFrame(
        [
            {
                "text": text,
                "subject": subject,
                "body": body,
                "sender": sender,
                "has_reply_prefix": has_reply_prefix,
            }
        ]
    )

    preds, confs = _predict_with_confidence(model, X)
    predicted = preds[0]
    confidence = float(confs[0])

    score_data = score_email(
        subject=subject,
        body=body,
        sender=sender,
        has_reply_prefix=has_reply_prefix,
        predicted_label=predicted,
        confidence=confidence,
    )

    return {
        "predicted_category": predicted,
        "confidence_score": round(confidence, 4),
        "priority_score": score_data["priority_score"],
        "priority_band": score_data["priority_band"],
        "explanation": score_data["reasons"],
        "extracted_signals": score_data["signals"],
    }


def predict_batch(emails: List[Dict[str, Any]], model_name: str | None = None) -> List[Dict[str, Any]]:
    return [
        predict_email(
            subject=email["subject"],
            body=email["body"],
            sender=email.get("sender"),
            timestamp=email.get("timestamp"),
            thread_id=email.get("thread_id"),
            has_reply_prefix=email.get("has_reply_prefix"),
            model_name=model_name,
        )
        for email in emails
    ]





