import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_pipeline
from app.inference.pipeline import SummarizationPipeline, PipelineResponse, StructuredSummaryDetails
from app.inference.student_model import StudentModelWrapper
from app.utils.history_store import get_next_ticket_id, save_history_item, clear_history, get_all_history
from app.analytics.issue_clusterer import get_issue_clusterer, IssueClusterer

@pytest.fixture
def mock_pipeline():
    mock_model = MagicMock(spec=StudentModelWrapper)
    mock_model.is_loaded = True
    mock_model.model_name = "mock-qwen-1.5b"
    mock_model.precision = "float16"
    mock_model.device_setting = "cuda"

    pipeline = MagicMock(spec=SummarizationPipeline)
    pipeline.model_wrapper = mock_model
    pipeline.run.return_value = PipelineResponse(
        ticket_id="TICK-1001",
        customer_name="Rajesh Kumar",
        customer_email="rajesh@example.com",
        customer_phone="+91 98401 23456",
        summary="User unable to log in after v4.2 update.",
        structured_summary=StructuredSummaryDetails(
            core_issue="Unable to log in to mobile app after update.",
            customer_intent="Banking / Support (High Priority)",
            key_action_items="Reset session token and review auth logs."
        ),
        priority="High",
        model="mock-qwen-1.5b",
        latency_ms=90.0,
        input_tokens=50,
        output_tokens=15
    )
    return pipeline

def test_next_ticket_id_generation():
    clear_history()
    next_id = get_next_ticket_id()
    assert next_id.startswith("TICK-")
    assert next_id == "TICK-1001"

    # Save a record with TICK-1001
    save_history_item(
        ticket_id="TICK-1001",
        ticket_text="Customer reported login issue.",
        summary="Login issue reported.",
        model="test-model",
        latency_ms=50.0,
        customer_name="Customer 1"
    )

    next_id_after = get_next_ticket_id()
    assert next_id_after == "TICK-1002"

def test_customer_metadata_persistence():
    clear_history()
    saved = save_history_item(
        ticket_id="",  # Auto-assign
        ticket_text="Payment was deducted twice during checkout.",
        summary="Double billing glitch during checkout.",
        model="test-model",
        latency_ms=45.0,
        customer_name="Priya Sharma",
        customer_email="priya@example.com",
        customer_phone="+91 98402 34567"
    )

    assert saved["ticket_id"] == "TICK-1001"
    assert saved["customer_name"] == "Priya Sharma"
    assert saved["customer_email"] == "priya@example.com"
    assert saved["customer_phone"] == "+91 98402 34567"
    assert "created_at_formatted" in saved

def test_issue_clustering_and_company_outage_detection():
    clusterer = IssueClusterer()

    sample_tickets = [
        {
            "ticket_id": "TICK-1001",
            "customer_name": "Rajesh Kumar",
            "customer_email": "rajesh@example.com",
            "customer_phone": "+91 98401 23456",
            "ticket_text": "Unable to login to mobile banking app following v4.2 update with invalid session.",
            "summary": "Customer cannot login to banking app post v4.2 update.",
            "priority": "Critical"
        },
        {
            "ticket_id": "TICK-1002",
            "customer_name": "Priya Sharma",
            "customer_email": "priya@example.com",
            "customer_phone": "+91 98402 34567",
            "ticket_text": "Mobile banking app crashes and gives invalid session error on login attempt.",
            "summary": "App login fails with invalid session token error.",
            "priority": "Critical"
        },
        {
            "ticket_id": "TICK-1003",
            "customer_name": "Karthik Raman",
            "customer_email": "karthik@example.com",
            "customer_phone": "+91 98403 45678",
            "ticket_text": "Since morning update, app shows invalid session token during sign in.",
            "summary": "Login blocked by invalid session token across staff accounts.",
            "priority": "High"
        },
        {
            "ticket_id": "TICK-1004",
            "customer_name": "Deepak Verma",
            "customer_email": "deepak@example.com",
            "customer_phone": "+91 98407 89012",
            "ticket_text": "Package tracking for Order #88410 is stuck in transit at carrier warehouse.",
            "summary": "Order tracking stuck for 5 days.",
            "priority": "Medium"
        }
    ]

    report = clusterer.cluster_tickets(sample_tickets)
    assert report.total_tickets_analyzed == 4
    assert report.company_outages_detected >= 1
    assert report.has_active_outage is True

    # First cluster should be authentication outage with 3 affected users
    top_cluster = report.clusters[0]
    assert top_cluster.affected_count == 3
    assert top_cluster.severity == "CRITICAL_OUTAGE"
    assert top_cluster.is_company_wide_issue is True
    assert "Rajesh Kumar" in top_cluster.common_issue_summary
    assert "Priya Sharma" in top_cluster.common_issue_summary
    assert "company" in top_cluster.common_issue_summary.lower()

def test_api_common_issues_and_next_id(mock_pipeline):
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    client = TestClient(app)

    # 1. Test next-id endpoint
    resp_id = client.get("/tickets/next-id")
    assert resp_id.status_code == 200
    assert "next_ticket_id" in resp_id.json()

    # 2. Test seed-demo endpoint
    seed_resp = client.post("/tickets/seed-demo")
    assert seed_resp.status_code == 200
    seed_data = seed_resp.json()
    assert seed_data["total_tickets_analyzed"] == 8
    assert seed_data["company_outages_detected"] >= 2
    assert seed_data["has_active_outage"] is True
    assert len(seed_data["clusters"]) >= 3

    # 3. Test common-issues GET endpoint
    comm_resp = client.get("/common-issues")
    assert comm_resp.status_code == 200
    comm_data = comm_resp.json()
    assert comm_data["total_tickets_analyzed"] == 8

    # 4. Test summarize with auto ID and customer info
    sum_payload = {
        "customer_name": "Test Customer",
        "customer_email": "test@domain.com",
        "customer_phone": "+91 99999 88888",
        "ticket_text": "Unable to login after upgrade."
    }
    sum_resp = client.post("/summarize", json=sum_payload)
    assert sum_resp.status_code == 200
    sum_json = sum_resp.json()
    assert sum_json["ticket_id"].startswith("TICK-")
    assert sum_json["customer_name"] == "Test Customer"
    assert sum_json["customer_email"] == "test@domain.com"

    app.dependency_overrides.clear()
