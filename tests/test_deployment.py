from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_pipeline
from app.inference.pipeline import SummarizationPipeline
from app.inference.student_model import StudentModelWrapper
from app.inference.quantization import QuantizationConfig, load_deployment_config

def test_quantization_config_init():
    cfg_fp16 = QuantizationConfig(precision="fp16")
    assert cfg_fp16.precision == "fp16"

    cfg_int4 = QuantizationConfig(precision="int4")
    assert cfg_int4.precision == "int4"

def test_load_deployment_config():
    cfg = load_deployment_config()
    assert isinstance(cfg, dict)
    assert "server" in cfg or "hardware" in cfg or cfg == {}

def test_api_metrics_endpoint():
    mock_model = MagicMock(spec=StudentModelWrapper)
    mock_model.is_loaded = True
    mock_model.model_name = "mock-model"
    mock_model.precision = "float16"
    mock_model.device_setting = "cuda"

    mock_pipeline = MagicMock(spec=SummarizationPipeline)
    mock_pipeline.model_wrapper = mock_model

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    client = TestClient(app)

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_json = metrics_resp.json()
    assert "total_requests" in metrics_json
    assert "successful_requests" in metrics_json
    assert "failed_requests" in metrics_json
    assert "average_latency_ms" in metrics_json

    app.dependency_overrides.clear()
