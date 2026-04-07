from models.scoring import compute_priority_score


def test_scoring_urgent_high_confidence():
    score, reasons, band = compute_priority_score(
        predicted_label="urgent",
        confidence=0.95,
        signals={
            "urgency_keyword_count": 2,
            "followup_keyword_count": 0,
            "spam_keyword_count": 0,
            "deadline_flag": 1,
            "has_reply_prefix": 0,
            "sender_importance": 0,
        },
    )
    assert score >= 80
    assert band == "high"
    assert any("urgent" in r.lower() for r in reasons)


def test_scoring_spam_low():
    score, _, band = compute_priority_score(
        predicted_label="spam",
        confidence=0.9,
        signals={
            "urgency_keyword_count": 0,
            "followup_keyword_count": 0,
            "spam_keyword_count": 2,
            "deadline_flag": 0,
            "has_reply_prefix": 0,
            "sender_importance": 0,
        },
    )
    assert score <= 20
    assert band == "low"



