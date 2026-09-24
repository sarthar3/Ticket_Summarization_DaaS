from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Depends, Query
from app.api.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    HealthResponse,
    MetricsResponse,
    HistoryItem,
    HistoryListResponse
)
from app.inference.pipeline import SummarizationPipeline
from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger
from app.utils.history_store import (
    save_history_item,
    get_all_history,
    get_history_by_ticket_id,
    clear_history
)

logger = get_logger(__name__)

router = APIRouter()

# Global pipeline instance initialized during app startup
pipeline_instance: Optional[SummarizationPipeline] = None

# Operational Metrics Counters
metrics_counter = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0
}

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

@router.get("/metrics", response_model=MetricsResponse, summary="Retrieve operational metrics")
def get_metrics():
    """Returns live request counts, token processing metrics, and average latency."""
    total = metrics_counter["total_requests"]
    avg_latency = (metrics_counter["total_latency_ms"] / total) if total > 0 else 0.0

    return MetricsResponse(
        total_requests=metrics_counter["total_requests"],
        successful_requests=metrics_counter["successful_requests"],
        failed_requests=metrics_counter["failed_requests"],
        average_latency_ms=round(avg_latency, 2),
        total_input_tokens=metrics_counter["total_input_tokens"],
        total_output_tokens=metrics_counter["total_output_tokens"]
    )

@router.get("/history", response_model=HistoryListResponse, summary="Retrieve ticket summarization history")
def get_history(limit: Optional[int] = Query(None, description="Max items to return")):
    """Returns list of past ticket summarization records stored in history."""
    items = get_all_history(limit=limit)
    return HistoryListResponse(
        total_count=len(items),
        items=[HistoryItem(**item) for item in items]
    )

@router.get("/history/{ticket_id}", response_model=HistoryItem, summary="Get ticket history by ID")
def get_ticket_history(ticket_id: str):
    """Returns historical record for a specific ticket_id."""
    item = get_history_by_ticket_id(ticket_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No history record found for ticket_id '{ticket_id}'"
        )
    return HistoryItem(**item)

@router.delete("/history", summary="Clear ticket summarization history")
def delete_history():
    """Clears all historical ticket summarization records."""
    clear_history()
    return {"message": "History cleared successfully."}

@router.post("/summarize", response_model=SummarizeResponse, summary="Summarize support ticket")
def summarize_ticket(
    payload: SummarizeRequest,
    pipeline: SummarizationPipeline = Depends(get_pipeline)
):
    """Processes a support ticket, saves to history, and returns a concise summary with metadata."""
    metrics_counter["total_requests"] += 1
    try:
        raw_ticket = payload.model_dump()
        response = pipeline.run(raw_ticket)

        metrics_counter["successful_requests"] += 1
        metrics_counter["total_latency_ms"] += response.latency_ms
        metrics_counter["total_input_tokens"] += response.input_tokens
        metrics_counter["total_output_tokens"] += response.output_tokens

        struct_dict = response.structured_summary.model_dump() if response.structured_summary else None

        # Persist to history
        save_history_item(
            ticket_id=response.ticket_id,
            ticket_text=payload.ticket_text,
            summary=response.summary,
            model=response.model,
            latency_ms=response.latency_ms,
            sector=payload.sector,
            intent=payload.intent,
            priority=response.priority,
            structured_summary=struct_dict
        )

        return SummarizeResponse(
            ticket_id=response.ticket_id,
            summary=response.summary,
            structured_summary=struct_dict,
            priority=response.priority,
            model=response.model,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens
        )
    except ValueError as ve:
        metrics_counter["failed_requests"] += 1
        logger.warning(f"Validation error for request ticket_id '{payload.ticket_id}': {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request payload: {str(ve)}"
        )
    except Exception as e:
        metrics_counter["failed_requests"] += 1
        logger.error(f"Inference error processing ticket_id '{payload.ticket_id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the summarization request."
        )

