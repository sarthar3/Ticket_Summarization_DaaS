import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing.cleaner import TicketPreprocessor
from app.utils.officials_store import get_all_officials, add_official, delete_official

client = TestClient(app)

def test_classify_origin_client_side():
    pre = TicketPreprocessor()
    # Ticket describing insufficient balance and payment failure
    ticket_text = "My debit card payment was declined due to insufficient balance in my checking account. How do I transfer funds?"
    origin, guidance, escalation = pre.classify_origin_and_guidance(
        ticket_text=ticket_text,
        issue_type="Card Payment Failure / Insufficient Funds",
        domain="Banking & Financial Services",
        priority="Low"
    )

    assert origin == "CLIENT_SIDE"
    assert guidance is not None
    assert len(guidance["steps"]) > 0
    assert "balance" in guidance["message_template"].lower() or "deposit" in guidance["message_template"].lower()
    assert escalation is None

def test_classify_origin_company_side():
    pre = TicketPreprocessor()
    # Ticket describing 503 outage affecting 500+ users
    ticket_text = "Internet banking portal and payment gateway return HTTP 503 Service Unavailable for more than 500 customers."
    origin, guidance, escalation = pre.classify_origin_and_guidance(
        ticket_text=ticket_text,
        issue_type="Multi-Service Outage / HTTP 503",
        domain="Cloud Infrastructure",
        priority="P1 · Critical"
    )

    assert origin == "COMPANY_SIDE"
    assert guidance is None
    assert escalation is not None
    assert escalation["target_official"]["email"] is not None
    assert "SRE" in escalation["target_official"]["designation"] or "Infra" in escalation["target_official"]["department"]
    assert "HTTP 503" in escalation["email_subject"] or "Outage" in escalation["email_subject"]

def test_officials_crud_endpoints():
    # 1. Fetch current officials
    res = client.get("/officials")
    assert res.status_code == 200
    data = res.json()
    assert "officials" in data
    assert len(data["officials"]) >= 5

    # 2. Add a new company official
    new_official = {
        "name": "Priya Nair",
        "designation": "Director of Customer Trust & Security",
        "email": "priya.nair@company-test.com",
        "department": "Cybersecurity & Fraud"
    }
    add_res = client.post("/officials", json=new_official)
    assert add_res.status_code == 200
    added = add_res.json()["official"]
    assert added["name"] == "Priya Nair"
    assert added["email"] == "priya.nair@company-test.com"
    official_id = added["id"]

    # 3. Verify official is listed
    list_res = client.get("/officials")
    ids = [o["id"] for o in list_res.json()["officials"]]
    assert official_id in ids

    # 4. Delete the added test official
    del_res = client.delete(f"/officials/{official_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify official is removed
    post_del_res = client.get("/officials")
    post_del_ids = [o["id"] for o in post_del_res.json()["officials"]]
    assert official_id not in post_del_ids

def test_escalate_email_endpoint():
    payload = {
        "official_name": "Vikram Sharma",
        "official_email": "vikram.sre@company.com",
        "official_designation": "Head of SRE & Cloud Infrastructure",
        "subject": "[CRITICAL OUTAGE] Gateway 503 Spike across Banking APIs",
        "body_text": "High volume of 503 Service Unavailable errors reported by 500+ users. Upstream cluster pod memory exhausted.",
        "urgency": "CRITICAL / P1"
    }
    res = client.post("/officials/escalate-email", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"].upper() == "DISPATCHED"
    assert "Vikram Sharma" in data["message"]
    assert data["details"]["recipient"] == "vikram.sre@company.com"
    assert data["details"]["urgency"] == "CRITICAL / P1"

def test_teacher_buffer_stats_endpoint():
    res = client.get("/teacher/buffer-stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_unseen_samples" in data
    assert "buffer_path" in data
    assert "latest_samples" in data

def test_apply_cluster_solution_endpoint():
    # First seed demo tickets so clusters exist
    seed_res = client.post("/tickets/seed-demo")
    assert seed_res.status_code == 200

    # Get clusters
    issues_res = client.get("/common-issues")
    assert issues_res.status_code == 200
    clusters = issues_res.json()["clusters"]
    assert len(clusters) > 0

    target_cluster = clusters[0]
    cid = target_cluster["cluster_id"]

    # Apply unified master solution
    solution_text = "Master Resolution: Re-provisioned load balancer connection pool and refreshed ingress security tokens across all pods."
    apply_res = client.post(f"/clusters/{cid}/apply-solution", json={"solution_text": solution_text})
    assert apply_res.status_code == 200
    result = apply_res.json()
    assert result["status"].upper() == "APPLIED"
    assert result["cluster_id"] == cid
    assert result["unified_solution"] == solution_text
    assert result["affected_count"] >= 1

    # Verify common issues report now reflects the applied master solution
    updated_issues = client.get("/common-issues").json()
    updated_cluster = next((c for c in updated_issues["clusters"] if c["cluster_id"] == cid), None)
    assert updated_cluster is not None
    assert updated_cluster["resolution_applied"] is True
    assert updated_cluster["unified_solution"] == solution_text
