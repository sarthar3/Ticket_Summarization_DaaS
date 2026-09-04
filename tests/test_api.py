from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_pipeline
from app.inference.pipeline import SummarizationPipeline, PipelineResponse
from app.inference.student_model import StudentModelWrapper

def test_api_health_and_summarize():
    mock_model = MagicMock(spec=StudentModelWrapper)
    mock_model.is_loaded = True
    mock_model.model_name = "mock-qwen-1.5b"
    mock_model.precision = "float16"
    mock_model.device_setting = "cuda"

    mock_pipeline = MagicMock(spec=SummarizationPipeline)
    mock_pipeline.model_wrapper = mock_model
    mock_pipeline.run.return_value = PipelineResponse(
        ticket_id="T001",
        summary="User experiencing connection drops in Austin.",
        model="mock-qwen-1.5b",
        latency_ms=85.4,
        input_tokens=60,
        output_tokens=15
    )

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline

    client = TestClient(app)

    # Test GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    assert health_json["status"] == "healthy"
    assert health_json["model_loaded"] is True
    assert health_json["model_name"] == "mock-qwen-1.5b"

    # Test POST /summarize
    summarize_payload = {
        "ticket_id": "T001",
        "ticket_text": "Fiber connection keeps dropping every 20 minutes in Austin."
    }
    sum_resp = client.post("/summarize", json=summarize_payload)
    assert sum_resp.status_code == 200
    sum_json = sum_resp.json()
    assert sum_json["ticket_id"] == "T001"
    assert sum_json["summary"] == "User experiencing connection drops in Austin."
    assert sum_json["model"] == "mock-qwen-1.5b"
    assert sum_json["latency_ms"] == 85.4
    assert sum_json["input_tokens"] == 60
    assert sum_json["output_tokens"] == 15

    # Clean up dependency overrides
    app.dependency_overrides.clear()

def test_api_invalid_payload():
    client = TestClient(app)
    # Missing required ticket_id
    invalid_payload = {
        "ticket_text": "Missing ticket_id"
    }
    resp = client.post("/summarize", json=invalid_payload)
    assert resp.status_code == 422  # Unprocessable Entity (Pydantic validation failure)
