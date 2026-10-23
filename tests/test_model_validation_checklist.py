import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.api.routes import get_pipeline
from app.inference.pipeline import SummarizationPipeline, PipelineResponse, StructuredSummaryDetails
from app.inference.student_model import StudentModelWrapper
from app.utils.history_store import clear_history
from app.preprocessing.cleaner import TicketPreprocessor, TicketData

client = TestClient(app)
cleaner = TicketPreprocessor()

OUTAGE_COMPLAINT = (
    "Our organization's internet banking portal, mobile banking application, and payment gateway "
    "have all stopped working since 10:30 AM. More than 500 customers are unable to complete transactions. "
    "Several users report HTTP 503 Service Unavailable errors. This is affecting business operations "
    "and requires immediate investigation."
)

ROUTINE_QUERY = (
    "Customer requesting information about annual percentage yield on 2-year fixed deposits "
    "and eligibility criteria for senior citizen premium rates. No urgent account or transaction issues."
)


def test_cleaner_critical_outage_priority_and_details():
    """Item 1: 500+ users and 3 services down must classify as P1 · Critical."""
    prio, reason = cleaner.classify_priority_details(OUTAGE_COMPLAINT)
    assert prio == "P1 · Critical"
    assert "outage" in reason.lower()
    assert "503" in reason or "widespread" in reason.lower()


def test_cleaner_normal_query_priority_details():
    """Item 2: Routine query with zero service disruption must classify as Low."""
    prio, reason = cleaner.classify_priority_details(ROUTINE_QUERY)
    assert prio == "P4 / Low"
    assert "Standard customer support inquiry" in reason


def test_cleaner_domain_and_issue_classification_no_billing_hallucination():
    """Item 3: Domain is Cloud Infrastructure, Issue Type is Multi-Service Outage / HTTP 503 without billing dispute."""
    domain, issue_type, urgency, impact, actions = cleaner.classify_domain_and_issue(
        OUTAGE_COMPLAINT,
        sector="Cloud Infrastructure"
    )
    assert domain == "Cloud Infrastructure"
    assert issue_type == "Multi-Service Outage / HTTP 503"
    assert "billing" not in issue_type.lower()
    assert "dispute" not in issue_type.lower()
    assert "500" in impact
    assert len(actions) == 4
    assert any("SRE" in a or "incident response" in a for a in actions)


def test_routing_safety_escalates_p1_outages():
    """Item 5: Routing safety policy evaluates severity, confidence, and novelty."""
    is_unseen, reason = cleaner.detect_distribution(
        ticket_text=OUTAGE_COMPLAINT,
        domain="Cloud Infrastructure",
        priority_label="P1 · Critical",
        issue_type="Multi-Service Outage / HTTP 503"
    )
    assert is_unseen is True
    assert "Critical Severity Safety Escalation" in reason


@pytest.fixture
def mock_pipeline():
    mock_model = MagicMock(spec=StudentModelWrapper)
    mock_model.is_loaded = True
    mock_model.model_name = "mock-qwen-1.5b"
    mock_model.precision = "float16"
    mock_model.device_setting = "cpu"

    pipeline = MagicMock(spec=SummarizationPipeline)
    pipeline.model_wrapper = mock_model
    pipeline.run.return_value = PipelineResponse(
        ticket_id="TICK-1017",
        customer_name="Karan",
        customer_email="dei@gmail.com",
        customer_phone="8300661376",
        summary="Major multi-service outage affecting banking services for 500+ users due to HTTP 503.",
        structured_summary=StructuredSummaryDetails(
            core_issue="Multiple banking services unavailable (HTTP 503).",
            customer_intent="Cloud Infrastructure / Multi-Service Outage / HTTP 503 (P1 · Critical)",
            domain="Cloud Infrastructure",
            issue_type="Multi-Service Outage / HTTP 503",
            urgency="P1 · Critical",
            customer_impact="More than 500 customers affected; banking access disrupted.",
            recommended_actions=[
                "Escalate to the incident response / SRE team.",
                "Investigate application health and recent deployments.",
                "Check logs and infrastructure health.",
                "Restore service and communicate updates."
            ],
            key_action_items="Escalate to SRE team immediately."
        ),
        priority="P1 · Critical",
        model="mock-qwen-1.5b",
        latency_ms=125.0,
        input_tokens=60,
        output_tokens=30,
        route_used="teacher",
        route_reason="Severity Escalation: P1 outage detected -> routed to Teacher LLM (Groq 120B) for executive review."
    )
    return pipeline


def test_api_outage_summarize_response_structure(mock_pipeline):
    """Verifies API response enforces validated schema with separated fields."""
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        response = client.post("/summarize?route=auto", json={
            "customer_name": "Karan",
            "customer_email": "dei@gmail.com",
            "customer_phone": "8300661376",
            "sector": "Cloud Infrastructure",
            "ticket_text": OUTAGE_COMPLAINT
        })
        assert response.status_code == 200
        data = response.json()

        assert data["priority"] == "P1 · Critical"
        assert "latency_breakdown" in data
        assert "inference_ms" in data["latency_breakdown"]
        assert data["structured_summary"]["domain"] == "Cloud Infrastructure"
        assert data["structured_summary"]["issue_type"] == "Multi-Service Outage / HTTP 503"
        assert "customer_impact" in data["structured_summary"]
        assert len(data["structured_summary"]["recommended_actions"]) == 4
    finally:
        app.dependency_overrides.clear()


def test_unverified_outage_report_flag(mock_pipeline):
    """Item 7: An uncorroborated single P1 outage ticket triggers Unverified Outage Report."""
    clear_history()
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        # Submit single outage ticket
        res1 = client.post("/summarize?route=auto", json={
            "customer_name": "Karan",
            "sector": "Cloud Infrastructure",
            "ticket_text": OUTAGE_COMPLAINT
        })
        assert res1.status_code == 200

        # Check company issues
        report_res = client.get("/common-issues")
        assert report_res.status_code == 200
        report = report_res.json()
        assert report["has_unverified_outage_report"] is True
        assert report["company_status"] == "UNVERIFIED_OUTAGE_REPORT"
    finally:
        app.dependency_overrides.clear()


def test_latency_benchmark_endpoint(mock_pipeline):
    """Item 6: Benchmark endpoint returns median and P95 latency stats across multiple runs."""
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        res = client.post("/evaluate/latency-benchmark", json={"runs": 2, "include_teacher": False})
        assert res.status_code == 200
        data = res.json()
        assert data["runs_executed"] == 2
        assert "student_stats" in data
        assert "median_ms" in data["student_stats"]
        assert "p95_ms" in data["student_stats"]
        assert "breakdown" in data["student_stats"]
    finally:
        app.dependency_overrides.clear()
