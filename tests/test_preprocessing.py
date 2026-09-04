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
    with pytest.raises(ValueError, match="ticket_id"):
        preprocessor.validate_and_normalize({"ticket_text": "No ID here"})

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
