from unittest.mock import MagicMock
from app.config.settings import get_settings
from app.inference.pipeline import SummarizationPipeline, PipelineResponse
from app.inference.student_model import StudentModelWrapper

def test_pipeline_format_prompt():
    settings = get_settings()
    pipeline = SummarizationPipeline(settings=settings)
    formatted = pipeline.format_prompt("Sample ticket content")
    assert "Sample ticket content" in formatted
    assert "Summary:" in formatted

def test_pipeline_post_processing():
    settings = get_settings()
    pipeline = SummarizationPipeline(settings=settings)
    raw = "Summary :- User reported payment failure on credit card."
    cleaned = pipeline.post_process_summary(raw)
    assert cleaned == "User reported payment failure on credit card."

def test_pipeline_mock_run():
    settings = get_settings()
    mock_model = MagicMock(spec=StudentModelWrapper)
    mock_model.model_name = "mock-student-model"
    mock_model.generate.return_value = ("Generated summary result.", 45, 12, 120.5)

    pipeline = SummarizationPipeline(settings=settings, model_wrapper=mock_model)
    raw_ticket = {
        "ticket_id": "TICK-TEST",
        "ticket_text": "Customer needs password reset support urgently."
    }
    response = pipeline.run(raw_ticket)

    assert isinstance(response, PipelineResponse)
    assert response.ticket_id == "TICK-TEST"
    assert response.summary == "Generated summary result."
    assert response.model == "mock-student-model"
    assert response.input_tokens == 45
    assert response.output_tokens == 12
    assert response.latency_ms == 120.5
