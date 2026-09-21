from app.evaluation.failure_analysis import FailureAnalyzer

def test_detect_hallucinated_entities():
    analyzer = FailureAnalyzer()
    ticket_text = "User experienced login issue on account T001."
    summary_clean = "User faced login error on ticket T001."
    summary_hallucinated = "User charged $1,499 twice on ERR-902."

    assert len(analyzer.detect_hallucinated_entities(ticket_text, summary_clean)) == 0
    hallucinations = analyzer.detect_hallucinated_entities(ticket_text, summary_hallucinated)
    assert len(hallucinations) > 0
    assert any("1,499" in h for h in hallucinations)
    assert any("ERR-902" in h for h in hallucinations)

def test_detect_repetition():
    analyzer = FailureAnalyzer()
    clean_text = "User cannot log in after the application update."
    repeating_text = "User cannot log in User cannot log in User cannot log in"
    
    assert analyzer.detect_repetition(clean_text) is False
    assert analyzer.detect_repetition(repeating_text) is True

def test_detect_truncation():
    analyzer = FailureAnalyzer()
    complete_text = "Summary of ticket issue."
    incomplete_text = "Summary of ticket issue"
    
    assert analyzer.detect_truncation(complete_text) is False
    assert analyzer.detect_truncation(incomplete_text) is True

def test_audit_record():
    analyzer = FailureAnalyzer()
    ticket_text = "Password reset link failed for customer."
    summary_bad = "Password reset link failed Password reset link failed"
    
    report = analyzer.audit_record("T001", ticket_text, summary_bad)
    assert report.has_failure is True
    assert "Repetition Loop" in report.failure_types or "Truncation Anomaly" in report.failure_types
