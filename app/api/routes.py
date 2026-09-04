from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends
from app.api.schemas import SummarizeRequest, SummarizeResponse, HealthResponse
from app.inference.pipeline import SummarizationPipeline
from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global pipeline instance initialized during app startup
pipeline_instance: Optional[SummarizationPipeline] = None

def get_pipeline() -> SummarizationPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        settings = get_settings()
        pipeline_instance = SummarizationPipeline(settings=settings)
    return pipeline_instance

@router.get("/health", response_model=HealthResponse, summary="Check service and model readiness")
def health_check(pipeline: SummarizationPipeline = Depends(get_pipeline)):
    """Returns operational health status and model readiness info."""
    model_wrapper = pipeline.model_wrapper
    is_ready = model_wrapper.is_loaded
    
    return HealthResponse(
        status="healthy" if is_ready else "initializing",
        model_loaded=is_ready,
        model_name=model_wrapper.model_name,
        precision=model_wrapper.precision,
        device=model_wrapper.device_setting
    )

@router.post("/summarize", response_model=SummarizeResponse, summary="Summarize support ticket")
def summarize_ticket(
    payload: SummarizeRequest,
    pipeline: SummarizationPipeline = Depends(get_pipeline)
):
    """Processes a support ticket and returns a concise summary with latency & token usage metadata."""
    try:
        raw_ticket = payload.model_dump()
        response = pipeline.run(raw_ticket)
        return SummarizeResponse(
            ticket_id=response.ticket_id,
            summary=response.summary,
            model=response.model,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens
        )
    except ValueError as ve:
        logger.warning(f"Validation error for request ticket_id '{payload.ticket_id}': {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request payload: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Inference error processing ticket_id '{payload.ticket_id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the summarization request."
        )
