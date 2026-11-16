import pytest
from app.utils.pii_anonymizer import PIIAnonymizer, get_pii_anonymizer

def test_pii_redaction_and_restoration():
    anonymizer = PIIAnonymizer()
    raw_text = (
        "Customer Rajesh Kumar (rajesh.k@example.com, +91 98765 43210) reported an issue with "
        "Order #88410. He attempted a payment with card 4111 2222 3333 4444 which was charged twice."
    )

    redacted, mapping = anonymizer.redact(raw_text)

    # Assert PII is redacted
    assert "rajesh.k@example.com" not in redacted
    assert "+91 98765 43210" not in redacted
    assert "4111 2222 3333 4444" not in redacted
    assert "88410" not in redacted

    # Assert placeholders are present
    assert "[EMAIL_1]" in redacted
    assert "[PHONE_1]" in redacted
    assert "[ORDER_ID_1]" in redacted
    assert "[CARD_1]" in redacted

    # Test restoration
    restored = anonymizer.restore(redacted, mapping)
    assert restored == raw_text

def test_pii_logging_redaction():
    anonymizer = get_pii_anonymizer()
    text = "User admin@company.com phoned from 9876543210 regarding card 4111 2222 3333 4444 issue."
    safe_log = anonymizer.redact_for_logging(text)
    assert "admin@company.com" not in safe_log
    assert "[EMAIL_REDACTED]" in safe_log
    assert "9876543210" not in safe_log
    assert "4111 2222 3333 4444" not in safe_log
    assert "[CARD_REDACTED]" in safe_log
