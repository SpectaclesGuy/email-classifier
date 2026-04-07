from app.utils import has_reply_prefix, merge_subject_body, preprocess_text


def test_preprocess_text_preserves_markers():
    text = "URGENT: Please respond ASAP!!!"
    processed = preprocess_text(text)
    assert "urgent" in processed
    assert "asap" in processed


def test_merge_subject_body():
    combined = merge_subject_body("Hello", "World")
    assert "Hello" in combined
    assert "World" in combined


def test_reply_prefix():
    assert has_reply_prefix("Re: status update") is True
    assert has_reply_prefix("Fwd: report") is True
    assert has_reply_prefix("Update") is False



