from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import FOLLOWUP_KEYWORDS, SPAM_KEYWORDS, URGENT_KEYWORDS
from app.utils import has_reply_prefix, merge_subject_body, preprocess_text


@dataclass
class LabelDecision:
    derived_label: str
    reasons: List[str]
    source: str


URGENCY_PATTERNS = [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in URGENT_KEYWORDS]
FOLLOWUP_PATTERNS = [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in FOLLOWUP_KEYWORDS]
SPAM_PATTERNS = [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in SPAM_KEYWORDS]


SPAM_LABELS = {"spam", "1", "true"}
HAM_LABELS = {"ham", "0", "false", "legit"}


def _matches_any(text: str, patterns: List[re.Pattern]) -> List[str]:
    matches = []
    for pattern in patterns:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def _normalize_label(value: object | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def derive_label(
    subject: Optional[str],
    body: Optional[str],
    original_label: Optional[object],
    has_reply_prefix_value: Optional[bool],
) -> LabelDecision:
    merged = merge_subject_body(subject, body)
    processed = preprocess_text(merged)

    normalized = _normalize_label(original_label)
    if normalized:
        if normalized in SPAM_LABELS:
            return LabelDecision("spam", ["Original label indicates spam"], "original_label")
        if normalized in HAM_LABELS:
            # Keep going to apply heuristic non-spam labels.
            pass

    spam_hits = _matches_any(processed, SPAM_PATTERNS)
    if spam_hits:
        return LabelDecision("spam", ["Matched spam keyword"], "heuristic")

    urgent_hits = _matches_any(processed, URGENCY_PATTERNS)
    if urgent_hits:
        return LabelDecision("urgent", ["Matched urgency keyword"], "heuristic")

    reply_prefix = has_reply_prefix_value if has_reply_prefix_value is not None else has_reply_prefix(subject)
    followup_hits = _matches_any(processed, FOLLOWUP_PATTERNS)
    if reply_prefix or followup_hits:
        reasons = []
        if reply_prefix:
            reasons.append("Subject has reply prefix")
        if followup_hits:
            reasons.append("Matched follow-up phrase")
        return LabelDecision("follow_up", reasons, "heuristic")

    return LabelDecision("informational", ["Fallback non-spam, non-urgent, non-follow-up"], "heuristic")


def derive_label_with_explanations(
    subject: Optional[str],
    body: Optional[str],
    original_label: Optional[object],
    has_reply_prefix_value: Optional[bool],
) -> Tuple[str, str, str]:
    decision = derive_label(subject, body, original_label, has_reply_prefix_value)
    return decision.derived_label, "; ".join(decision.reasons), decision.source

