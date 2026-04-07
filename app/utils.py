from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import (
    ACTION_PHRASES,
    FOLLOWUP_KEYWORDS,
    REPLY_PREFIXES,
    SENDER_IMPORTANCE,
    SPAM_KEYWORDS,
    URGENT_KEYWORDS,
)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"#$%&'()*+,./;<=>\[\\\]^_`{|}~]")
_DEADLINE_RE = re.compile(
    r"\b(by\s+eod|by\s+eob|deadline|due\s+date|by\s+\w+day|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def preprocess_text(text: str) -> str:
    text = normalize_text(text.lower())
    text = _PUNCT_RE.sub(" ", text)
    text = normalize_text(text)
    return text


def merge_subject_body(subject: Optional[str], body: Optional[str]) -> str:
    subject = subject or ""
    body = body or ""
    combined = f"{subject.strip()}\n{body.strip()}".strip()
    return combined


def has_reply_prefix(subject: Optional[str]) -> bool:
    if not subject:
        return False
    lowered = subject.strip().lower()
    return any(lowered.startswith(prefix) for prefix in REPLY_PREFIXES)


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered)


def uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    tokens = [t for t in re.split(r"\s+", text) if t]
    if not tokens:
        return 0.0
    upper = [t for t in tokens if len(t) > 1 and t.isupper()]
    return len(upper) / max(len(tokens), 1)


def extract_deadline_flag(text: str) -> int:
    if not text:
        return 0
    return 1 if _DEADLINE_RE.search(text) else 0


def extract_sender_importance(sender: Optional[str]) -> float:
    if not sender:
        return 0.0
    key = sender.strip().lower()
    return float(SENDER_IMPORTANCE.get(key, 0.0))


def extract_signals(
    subject: Optional[str],
    body: Optional[str],
    sender: Optional[str],
    has_reply_prefix_value: Optional[bool],
) -> Dict[str, float]:
    merged = merge_subject_body(subject, body)
    processed = preprocess_text(merged)
    urgency_count = count_keyword_hits(processed, URGENT_KEYWORDS)
    followup_count = count_keyword_hits(processed, FOLLOWUP_KEYWORDS)
    spam_count = count_keyword_hits(processed, SPAM_KEYWORDS)
    action_count = count_keyword_hits(processed, ACTION_PHRASES)
    exclamations = merged.count("!")
    reply_prefix = has_reply_prefix_value if has_reply_prefix_value is not None else has_reply_prefix(subject)

    return {
        "urgency_keyword_count": urgency_count,
        "followup_keyword_count": followup_count,
        "spam_keyword_count": spam_count,
        "action_phrase_count": action_count,
        "exclamation_count": float(exclamations),
        "uppercase_ratio": uppercase_ratio(merged),
        "subject_length": float(len(subject or "")),
        "body_length": float(len(body or "")),
        "sender_importance": extract_sender_importance(sender),
        "has_reply_prefix": float(bool(reply_prefix)),
        "deadline_flag": float(extract_deadline_flag(merged)),
    }


def extract_structured_features(df: pd.DataFrame) -> np.ndarray:
    features: List[List[float]] = []
    for _, row in df.iterrows():
        signals = extract_signals(
            row.get("subject"),
            row.get("body"),
            row.get("sender"),
            row.get("has_reply_prefix"),
        )
        features.append([
            signals["urgency_keyword_count"],
            signals["followup_keyword_count"],
            signals["spam_keyword_count"],
            signals["action_phrase_count"],
            signals["exclamation_count"],
            signals["uppercase_ratio"],
            signals["subject_length"],
            signals["body_length"],
            signals["sender_importance"],
            signals["has_reply_prefix"],
            signals["deadline_flag"],
        ])
    return np.asarray(features, dtype=float)


def get_structured_feature_names() -> List[str]:
    return [
        "urgency_keyword_count",
        "followup_keyword_count",
        "spam_keyword_count",
        "action_phrase_count",
        "exclamation_count",
        "uppercase_ratio",
        "subject_length",
        "body_length",
        "sender_importance",
        "has_reply_prefix",
        "deadline_flag",
    ]



