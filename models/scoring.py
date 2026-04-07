from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.config import PRIORITY_BANDS, SCORE_WEIGHTS
from app.utils import extract_signals


def compute_priority_score(
    predicted_label: str,
    confidence: float,
    signals: Dict[str, float],
) -> Tuple[int, List[str], str]:
    reasons: List[str] = []
    score = 0.0

    if predicted_label == "urgent":
        score += SCORE_WEIGHTS["urgent_base"]
        reasons.append("Predicted class is urgent")
    elif predicted_label == "follow_up":
        score += SCORE_WEIGHTS["follow_up_base"]
        reasons.append("Predicted class is follow_up")
    elif predicted_label == "informational":
        score += SCORE_WEIGHTS["informational_base"]
        reasons.append("Predicted class is informational")
    elif predicted_label == "spam":
        score += SCORE_WEIGHTS["spam_base"]
        reasons.append("Predicted class is spam")

    score += confidence * SCORE_WEIGHTS["confidence_weight"]
    if confidence >= 0.8:
        reasons.append("High model confidence")

    urgency_hits = signals.get("urgency_keyword_count", 0.0)
    followup_hits = signals.get("followup_keyword_count", 0.0)
    spam_hits = signals.get("spam_keyword_count", 0.0)

    if urgency_hits:
        score += urgency_hits * SCORE_WEIGHTS["urgency_keyword_weight"]
        reasons.append(f"Contains {int(urgency_hits)} urgency indicators")
    if followup_hits:
        score += followup_hits * SCORE_WEIGHTS["followup_keyword_weight"]
        reasons.append(f"Contains {int(followup_hits)} follow-up indicators")
    if spam_hits:
        score -= spam_hits * SCORE_WEIGHTS["spam_keyword_penalty"]
        reasons.append("Spam indicators reduce priority")

    if signals.get("deadline_flag"):
        score += SCORE_WEIGHTS["deadline_bonus"]
        reasons.append("Contains deadline-related phrase")

    if signals.get("has_reply_prefix"):
        score += SCORE_WEIGHTS["reply_chain_bonus"]
        reasons.append("Part of a reply chain")

    sender_bonus = signals.get("sender_importance", 0.0)
    if sender_bonus:
        score += sender_bonus * SCORE_WEIGHTS["sender_bonus"]
        reasons.append("Sender is marked as important")

    score = max(0.0, min(100.0, score))

    if score >= PRIORITY_BANDS["high"]:
        band = "high"
    elif score >= PRIORITY_BANDS["medium"]:
        band = "medium"
    else:
        band = "low"

    return int(round(score)), reasons, band


def score_email(
    subject: str,
    body: str,
    sender: str | None,
    has_reply_prefix: bool | None,
    predicted_label: str,
    confidence: float,
) -> Dict[str, Any]:
    signals = extract_signals(subject, body, sender, has_reply_prefix)
    score, reasons, band = compute_priority_score(predicted_label, confidence, signals)
    return {
        "priority_score": score,
        "priority_band": band,
        "reasons": reasons,
        "signals": signals,
    }



