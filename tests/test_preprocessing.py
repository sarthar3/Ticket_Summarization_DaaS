import pytest
from app.preprocessing.cleaner import TicketPreprocessor, TicketData
from app.preprocessing.token_stats import calculate_token_statistics

def test_ticket_cleaner_basic():
    preprocessor = TicketPreprocessor()
    raw_text = "  User unable  to login \n\n\n Error code ERR-902   "
    cleaned = preprocessor.clean_text(raw_text)
    assert cleaned == "User unable to login \n\n Error code ERR-902"

def test_ticket_validator_success():
    preprocessor = TicketPreprocessor()
    raw_dict = {
        "ticket_id": "T001",
        "ticket_text": "Need help resetting password.",
        "summary": "Password reset help."
    }
    data = preprocessor.validate_and_normalize(raw_dict)
    assert isinstance(data, TicketData)
    assert data.ticket_id == "T001"
    assert data.ticket_text == "Need help resetting password."
    assert data.summary == "Password reset help."
    assert data.sector == "General Support"

def test_ticket_validator_missing_fields():
    preprocessor = TicketPreprocessor()
    
    # Auto generates ticket_id if omitted
    data = preprocessor.validate_and_normalize({"ticket_text": "No ID provided"})
    assert data.ticket_id.startswith("TICK-")

    # Raises ValueError if ticket_text is missing or empty
    with pytest.raises(ValueError, match="ticket_text"):
        preprocessor.validate_and_normalize({"ticket_id": "T002", "ticket_text": ""})

def test_token_statistics():
    texts = [
        "word " * 10,  # 10 words
        "word " * 50,  # 50 words
        "word " * 100 # 100 words
    ]
    stats = calculate_token_statistics(texts, max_context_limit=40)
    assert stats.total_samples == 3
    assert stats.min_length == 10
    assert stats.max_length == 100
    assert stats.mean_length == 53.333333333333336
    assert stats.p50_length == 50.0
    assert stats.exceeding_limit_count == 2
    assert round(stats.exceeding_limit_percentage, 1) == 66.7


def test_insufficient_balance_classification_and_routing():
    preprocessor = TicketPreprocessor()
    ticket_dict = {
        "ticket_text": "My transaction of rs 500 is continuously getting failed as my balance in account is rs 100",
        "sector": "Banking & Finance"
    }
    data = preprocessor.validate_and_normalize(ticket_dict)
    assert data.priority == "P3 · Medium"
    assert "exceeds" in data.priority_reason.lower() or "balance" in data.priority_reason.lower()
    assert data.domain == "Banking & Financial Services"
    assert data.issue_type == "Transaction Failure / Insufficient Funds"
    assert "insufficient available balance" in data.customer_impact.lower() or "available account balance" in data.customer_impact.lower()
    assert len(data.recommended_actions) >= 2

    # Verify routing message does not contain '(None)'
    is_teacher, routing_reason = preprocessor.detect_distribution(
        ticket_text=data.ticket_text,
        sector=data.sector,
        priority_label=data.priority
    )
    assert "(None)" not in routing_reason
    assert "Student Model" in routing_reason


def test_ticket_types_classification():
    preprocessor = TicketPreprocessor()
    test_cases = [
        ("I ordered size 8 shoes but accidentally selected the wrong size. Please change to size 9 before shipping.", "user_error", "CLIENT_SIDE"),
        ("Package tracking for Order #88410 has had no movement for 5 days. Carrier says they haven't received it.", "third_party", "CLIENT_SIDE"),
        ("Requesting physical SIM replacement to eSIM QR code profile for new device.", "customer_request", "CLIENT_SIDE"),
        ("What are your branch opening hours and fee structure for corporate savings accounts?", "general_inquiry", "CLIENT_SIDE"),
        ("Mobile banking app crashes with invalid session token and HTTP 503 gateway timeout.", "company_defect", "COMPANY_SIDE"),
    ]
    for text, expected_type, expected_origin in test_cases:
        tt, sev, conf, rec, prob, c_conf = preprocessor.classify_ticket_type_and_severity(text)
        assert tt == expected_type, f"Expected {expected_type} for text '{text}', got {tt}"
        orig, guid, esc = preprocessor.classify_origin_and_guidance(text, "", "", sev)
        assert orig == expected_origin, f"Expected origin {expected_origin} for type {expected_type}, got {orig}"
        if expected_type == "company_defect":
            assert esc is not None, "Company defect should have escalation details"
        else:
            assert guid is not None, f"Non-defect ticket {expected_type} should have customer guidance steps"


