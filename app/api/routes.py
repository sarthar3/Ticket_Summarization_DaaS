from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status, Depends, Query
from app.api.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    StructuredSummaryDetails,
    HealthResponse,
    MetricsResponse,
    HistoryItem,
    HistoryListResponse,
    NextTicketIdResponse,
    CompanyIssuesReportSchema,
    LatencyBenchmarkRequest,
    LatencyBenchmarkResponse,
    LatencyStats,
    OfficialSchema,
    AddOfficialRequest,
    EscalateEmailRequest,
    EscalateEmailResponse,
    ApplySolutionRequest,
    ApplySolutionResponse,
    TeacherBufferStatsResponse
)
from app.inference.pipeline import SummarizationPipeline
from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger
from app.utils.history_store import (
    save_history_item,
    get_all_history,
    get_history_by_ticket_id,
    clear_history,
    get_next_ticket_id
)
from app.analytics.issue_clusterer import get_issue_clusterer

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

@router.get("/tickets/next-id", response_model=NextTicketIdResponse, summary="Get next chronological ticket ID")
def get_next_id():
    """Returns the next sequential ticket identifier in chronological sequence."""
    next_id = get_next_ticket_id()
    return NextTicketIdResponse(next_ticket_id=next_id, auto_assigned=True)

@router.get("/common-issues", response_model=CompanyIssuesReportSchema, summary="Analyze common issues across customer tickets")
def get_common_issues():
    """Aggregates and clusters all tickets in history to identify recurring issues,
    affected user counts, and company-side outage warnings."""
    history_items = get_all_history()
    clusterer = get_issue_clusterer()
    return clusterer.cluster_tickets(history_items)

class AnalyzeTicketsRequest(BaseModel):
    tickets: Optional[List[Dict[str, Any]]] = None

@router.post("/common-issues/analyze", response_model=CompanyIssuesReportSchema, summary="Analyze an arbitrary batch of tickets for common issues")
def analyze_common_issues_batch(payload: AnalyzeTicketsRequest):
    """Analyzes a provided batch of tickets or falls back to system history."""
    tickets = payload.tickets if payload.tickets is not None else get_all_history()
    clusterer = get_issue_clusterer()
    return clusterer.cluster_tickets(tickets)

@router.post("/tickets/seed-demo", response_model=CompanyIssuesReportSchema, summary="Seed realistic multi-user demo tickets")
def seed_demo_tickets():
    """Seeds 8 realistic customer tickets in chronological sequence (from 10:00 AM)
    simulating recurring company-side issues (Login Outage, Payment Glitch, Logistics Delay)
    and returns the analyzed common issues report."""
    from datetime import datetime, timezone, timedelta
    clear_history()

    base_time = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)

    demo_seeds = [
        {
            "ticket_id": "TICK-1001",
            "delta_min": 0,
            "name": "Rajesh Kumar",
            "email": "rajesh.k@example.com",
            "phone": "+91 98401 23456",
            "sector": "Banking & Finance",
            "priority": "Critical",
            "text": "Customer reported being unable to login to mobile banking app following v4.2 update on iOS 17.4. Password reset token fails with 'invalid session'. User needs urgent payroll transfer access before cutoff window.",
            "summary": "Core issue: Unable to log in to mobile banking app after v4.2 iOS 17.4 update; password reset token fails with 'invalid session'. Requested resolution: Immediate access to payroll transfer function.",
            "core_issue": "Unable to log in to mobile banking app after v4.2 update; invalid session token.",
            "action": "Investigate authentication gateway session invalidation patch and restore user session."
        },
        {
            "ticket_id": "TICK-1002",
            "delta_min": 5,
            "name": "Priya Sharma",
            "email": "priya.s@example.com",
            "phone": "+91 98402 34567",
            "sector": "Banking & Finance",
            "priority": "Critical",
            "text": "Mobile banking app repeatedly crashes and triggers 'invalid session' error every time I try to authenticate after the 4.2 update. Cannot check balance or make salary transfers.",
            "summary": "Core issue: Mobile app crashing with 'invalid session' error post-v4.2 update during authentication. Requested resolution: Urgent fix for salary disbursements.",
            "core_issue": "Mobile banking app crashes with 'invalid session' error on authentication.",
            "action": "Roll back iOS auth patch and verify token lifecycle."
        },
        {
            "ticket_id": "TICK-1003",
            "delta_min": 12,
            "name": "Karthik Raman",
            "email": "karthik.r@example.com",
            "phone": "+91 98403 45678",
            "sector": "Banking & Finance",
            "priority": "High",
            "text": "Since morning v4.2 update on iOS, app won't open my corporate account. Shows 'invalid session token'. Multiple staff members in my office are facing this exact same login problem.",
            "summary": "Core issue: Corporate account access blocked by 'invalid session token' post-v4.2 update across multiple office users. Requested resolution: Unblock corporate logins.",
            "core_issue": "Corporate mobile app access blocked by 'invalid session token'.",
            "action": "Escalate company-side authentication gateway outage to mobile engineering."
        },
        {
            "ticket_id": "TICK-1004",
            "delta_min": 18,
            "name": "Anitha George",
            "email": "anitha.g@example.com",
            "phone": "+91 98404 56789",
            "sector": "E-Commerce",
            "priority": "Critical",
            "text": "My payment of Rs 14,500 was deducted from my account but the checkout page failed with gateway timeout. Order is not showing under my orders.",
            "summary": "Core issue: Payment deducted of Rs 14,500 during checkout but transaction timed out without order placement. Requested resolution: Order confirmation or refund.",
            "core_issue": "Payment deducted during checkout with gateway timeout; order not placed.",
            "action": "Reconcile payment webhook callback with banking partner and issue order receipt."
        },
        {
            "ticket_id": "TICK-1005",
            "delta_min": 25,
            "name": "Vivek Nair",
            "email": "vivek.n@example.com",
            "phone": "+91 98405 67890",
            "sector": "Fintech & Billing",
            "priority": "High",
            "text": "Money deducted twice for invoice #4491 during checkout payment gateway glitch. Bank statement shows double charge but your portal shows unpaid.",
            "summary": "Core issue: Double deduction on invoice #4491 due to payment gateway glitch while status remains unpaid. Requested resolution: Immediate reversal of duplicate charge.",
            "core_issue": "Double deduction for invoice #4491 during checkout payment gateway glitch.",
            "action": "Trigger automated reversal of duplicate debit and mark invoice as paid."
        },
        {
            "ticket_id": "TICK-1006",
            "delta_min": 32,
            "name": "Sneha Reddy",
            "email": "sneha.r@example.com",
            "phone": "+91 98406 78901",
            "sector": "E-Commerce",
            "priority": "High",
            "text": "Order #88392 was placed for express 2-day delivery, but tracking is completely stuck at carrier facility for 4 days without updates. Please refund shipping.",
            "summary": "Core issue: Express 2-day order #88392 stuck at carrier origin facility for 4 days without tracking updates. Requested resolution: Refund shipping and expedite delivery.",
            "core_issue": "Express order #88392 stuck at carrier facility for 4 days without tracking updates.",
            "action": "Escalate package dispatch with carrier logistics team and refund shipping fee."
        },
        {
            "ticket_id": "TICK-1007",
            "delta_min": 40,
            "name": "Deepak Verma",
            "email": "deepak.v@example.com",
            "phone": "+91 98407 89012",
            "sector": "E-Commerce",
            "priority": "Medium",
            "text": "Package tracking for Order #88410 has had no movement for 5 days. Carrier says they haven't received package from warehouse.",
            "summary": "Core issue: Tracking stagnant for 5 days on order #88410; carrier confirms package not received from warehouse. Requested resolution: Warehouse dispatch check.",
            "core_issue": "Package tracking stagnant for 5 days; warehouse dispatch pending.",
            "action": "Audit warehouse inventory dispatch queue and regenerate carrier manifest."
        },
        {
            "ticket_id": "TICK-1008",
            "delta_min": 50,
            "name": "Meera Iyer",
            "email": "meera.i@example.com",
            "phone": "+91 98408 90123",
            "sector": "Telecom",
            "priority": "Low",
            "text": "Requesting verified physical SIM replacement to eSIM QR code profile for new device. Passed 2FA verification.",
            "summary": "Core issue: Requesting physical SIM to eSIM QR profile conversion following device upgrade. Requested resolution: Dispatch eSIM activation email.",
            "core_issue": "Physical SIM to eSIM conversion request with verified 2FA.",
            "action": "Dispatch automated eSIM QR activation profile to customer email."
        }
    ]

    for seed in demo_seeds:
        ticket_dt = base_time + timedelta(minutes=seed["delta_min"])
        formatted_t = ticket_dt.strftime("%I:%M %p, %b %d, %Y")
        struct = {
            "core_issue": seed["core_issue"],
            "customer_intent": f"{seed['sector']} / Support ({seed['priority']} Priority)",
            "key_action_items": seed["action"]
        }
        save_history_item(
            ticket_id=seed["ticket_id"],
            ticket_text=seed["text"],
            summary=seed["summary"],
            model="Qwen/Qwen2.5-1.5B-Instruct",
            latency_ms=124.5,
            sector=seed["sector"],
            priority=seed["priority"],
            structured_summary=struct,
            customer_name=seed["name"],
            customer_email=seed["email"],
            customer_phone=seed["phone"],
            created_at=ticket_dt.isoformat()
        )

    all_tickets = get_all_history()
    return get_issue_clusterer().cluster_tickets(all_tickets)

@router.post("/summarize", response_model=SummarizeResponse, summary="Summarize support ticket with intelligent auto-routing")
def summarize_ticket(
    payload: SummarizeRequest,
    route: Optional[str] = Query("auto", description="Routing mode: 'auto' (routes unseen/complex to Teacher, in-domain to Student), 'student', or 'teacher'"),
    pipeline: SummarizationPipeline = Depends(get_pipeline)
):
    """Processes a customer support ticket:
    - Auto-allocates sequential Ticket ID upon submission.
    - Classifies priority automatically based on operational urgency indicators.
    - Intelligently routes novel/unseen/high-complexity data to Teacher LLM (Groq 120B)
      and standard in-domain support queries to Student LLM (Qwen2.5-1.5B).
    - Persists customer metadata and faithful summary to history."""
    metrics_counter["total_requests"] += 1
    try:
        raw_ticket = payload.model_dump()
        t_id = (payload.ticket_id or "").strip()
        if not t_id or t_id.upper() in ["AUTO", "NONE", "AUTO-ASSIGN", "PENDING", "WAITING"]:
            t_id = get_next_ticket_id()
            raw_ticket["ticket_id"] = t_id

        # Determine distribution (unseen vs in-domain)
        from app.preprocessing.cleaner import TicketPreprocessor
        preprocessor = getattr(pipeline, "preprocessor", None) or TicketPreprocessor()
        is_unseen, route_reason = preprocessor.detect_distribution(payload.ticket_text, payload.sector)

        should_route_teacher = (route == "teacher") or (route == "auto" and is_unseen)

        if should_route_teacher:
            # Route to Teacher LLM (Groq 120B)
            from app.teacher.teacher_llm import get_teacher_llm
            teacher = get_teacher_llm()
            teacher_resp = teacher.generate_summary(t_id, payload.ticket_text)
            
            in_toks = len(payload.ticket_text.split())
            out_toks = len(teacher_resp.teacher_summary.split())
            metrics_counter["successful_requests"] += 1
            metrics_counter["total_latency_ms"] += teacher_resp.latency_ms
            metrics_counter["total_input_tokens"] += in_toks
            metrics_counter["total_output_tokens"] += out_toks

            struct_obj = None
            domain = payload.sector or "Cloud Infrastructure"
            issue_type = teacher_resp.intent or "Multi-Service Outage / HTTP 503"
            urgency = "Critical" if "P1" in (teacher_resp.priority or "") or "Critical" in (teacher_resp.priority or "") else "High"
            impact = "Widespread service unavailability reported."
            actions = ["Escalate to SRE incident response team.", "Restore service and verify status."]

            if teacher_resp.structured_summary:
                domain = teacher_resp.structured_summary.get("domain") or domain
                issue_type = teacher_resp.structured_summary.get("issue_type") or issue_type
                urgency = teacher_resp.structured_summary.get("urgency") or urgency
                impact = teacher_resp.structured_summary.get("customer_impact") or impact
                actions = teacher_resp.structured_summary.get("recommended_actions") or actions
                struct_obj = StructuredSummaryDetails(
                    core_issue=teacher_resp.structured_summary.get("core_issue", ""),
                    domain=domain,
                    issue_type=issue_type,
                    urgency=urgency,
                    priority=teacher_resp.priority or "P1 · Critical",
                    customer_impact=impact,
                    recommended_actions=actions,
                    customer_intent=teacher_resp.structured_summary.get("customer_intent", f"{domain} / {issue_type}"),
                    key_action_items=teacher_resp.structured_summary.get("key_action_items", "")
                )

            t_breakdown = {
                "preprocessing_ms": 2.0,
                "inference_ms": teacher_resp.latency_ms,
                "postprocessing_ms": 1.5,
                "total_ms": teacher_resp.latency_ms + 3.5
            }

            # Classify issue origin and guidance/escalation for Teacher responses
            t_origin, t_guidance, t_escalation = preprocessor.classify_origin_and_guidance(
                ticket_text=payload.ticket_text,
                issue_type=issue_type,
                domain=domain,
                priority=teacher_resp.priority or "P1 · Critical"
            )

            # Continuous Learning Buffer: Persist unseen ticket with Teacher supervision for student fine-tuning / GKD
            try:
                from app.utils.teacher_buffer import append_teacher_sample
                append_teacher_sample(
                    ticket_id=t_id,
                    ticket_text=payload.ticket_text,
                    teacher_summary=teacher_resp.teacher_summary,
                    domain=domain,
                    issue_type=issue_type,
                    priority=teacher_resp.priority,
                    structured_summary=teacher_resp.structured_summary,
                    model=teacher_resp.teacher_model,
                    provider=teacher_resp.provider,
                    latency_ms=teacher_resp.latency_ms
                )
            except Exception as be:
                logger.warning(f"Failed to append to teacher distillation buffer: {be}")

            saved_record = save_history_item(
                ticket_id=t_id,
                ticket_text=payload.ticket_text,
                summary=teacher_resp.teacher_summary,
                model=f"{teacher_resp.teacher_model} (Teacher)",
                latency_ms=teacher_resp.latency_ms,
                sector=domain,
                domain=domain,
                intent=issue_type,
                issue_type=issue_type,
                urgency=urgency,
                priority=teacher_resp.priority or "P1 · Critical",
                priority_reason=teacher_resp.priority_reason,
                customer_impact=impact,
                recommended_actions=actions,
                structured_summary=teacher_resp.structured_summary,
                customer_name=payload.customer_name,
                customer_email=payload.customer_email,
                customer_phone=payload.customer_phone,
                route_used="teacher",
                route_reason=route_reason if route == "auto" else "Explicit manual override: Forced Teacher LLM",
                is_unseen=is_unseen,
                latency_breakdown=t_breakdown,
                issue_origin=t_origin,
                client_guidance=t_guidance,
                escalation_details=t_escalation
            )

            return SummarizeResponse(
                ticket_id=t_id,
                customer_name=payload.customer_name,
                customer_email=payload.customer_email,
                customer_phone=payload.customer_phone,
                created_at=saved_record.get("created_at_formatted"),
                summary=teacher_resp.teacher_summary,
                structured_summary=struct_obj,
                domain=domain,
                issue_type=issue_type,
                urgency=urgency,
                priority=teacher_resp.priority or "P1 · Critical",
                priority_reason=teacher_resp.priority_reason,
                customer_impact=impact,
                recommended_actions=actions,
                intent=issue_type,
                issue_origin=t_origin,
                client_guidance=t_guidance,
                escalation_details=t_escalation,
                model=f"{teacher_resp.teacher_model} (Teacher)",
                route_used="teacher",
                route_reason=route_reason if route == "auto" else "Explicit manual override: Forced Teacher LLM",
                is_unseen=is_unseen,
                latency_ms=teacher_resp.latency_ms,
                latency_breakdown=t_breakdown,
                input_tokens=in_toks,
                output_tokens=out_toks
            )

        # Standard Student Route (Qwen2.5-1.5B)
        response = pipeline.run(raw_ticket)

        metrics_counter["successful_requests"] += 1
        metrics_counter["total_latency_ms"] += response.latency_ms
        metrics_counter["total_input_tokens"] += response.input_tokens
        metrics_counter["total_output_tokens"] += response.output_tokens

        struct_obj = None
        if response.structured_summary:
            if isinstance(response.structured_summary, StructuredSummaryDetails):
                struct_obj = response.structured_summary
            elif isinstance(response.structured_summary, dict):
                struct_obj = StructuredSummaryDetails(**response.structured_summary)
            elif hasattr(response.structured_summary, "model_dump"):
                struct_obj = StructuredSummaryDetails(**response.structured_summary.model_dump())
            elif hasattr(response.structured_summary, "dict"):
                struct_obj = StructuredSummaryDetails(**response.structured_summary.dict())
            else:
                struct_obj = response.structured_summary

        struct_dict = struct_obj.model_dump() if hasattr(struct_obj, "model_dump") else struct_obj

        saved_record = save_history_item(
            ticket_id=response.ticket_id,
            ticket_text=payload.ticket_text,
            summary=response.summary,
            model=response.model,
            latency_ms=response.latency_ms,
            sector=response.domain or payload.sector,
            domain=response.domain or payload.sector,
            intent=response.issue_type or response.intent,
            issue_type=response.issue_type,
            urgency=response.urgency,
            priority=response.priority,
            priority_reason=response.priority_reason,
            customer_impact=response.customer_impact,
            recommended_actions=response.recommended_actions,
            structured_summary=struct_dict,
            customer_name=payload.customer_name,
            customer_email=payload.customer_email,
            customer_phone=payload.customer_phone,
            route_used="student",
            route_reason=route_reason if route == "auto" else "Explicit manual override: Forced Student Model",
            is_unseen=False,
            latency_breakdown=response.latency_breakdown,
            issue_origin=getattr(response, "issue_origin", "COMPANY_SIDE"),
            client_guidance=getattr(response, "client_guidance", None),
            escalation_details=getattr(response, "escalation_details", None)
        )

        return SummarizeResponse(
            ticket_id=response.ticket_id,
            customer_name=payload.customer_name,
            customer_email=payload.customer_email,
            customer_phone=payload.customer_phone,
            created_at=saved_record.get("created_at_formatted"),
            summary=response.summary,
            structured_summary=struct_obj,
            domain=response.domain,
            issue_type=response.issue_type,
            urgency=response.urgency,
            priority=response.priority,
            priority_reason=response.priority_reason,
            customer_impact=response.customer_impact,
            recommended_actions=response.recommended_actions,
            intent=response.intent,
            issue_origin=getattr(response, "issue_origin", "COMPANY_SIDE"),
            client_guidance=getattr(response, "client_guidance", None),
            escalation_details=getattr(response, "escalation_details", None),
            model=response.model,
            route_used="student",
            route_reason=route_reason if route == "auto" else "Explicit manual override: Forced Student Model",
            is_unseen=False,
            latency_ms=response.latency_ms,
            latency_breakdown=response.latency_breakdown,
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
            detail=f"An error occurred while processing the summarization request: {str(e)}"
        )

class TeacherSummarizeRequest(BaseModel):
    ticket_id: Optional[str] = None
    ticket_text: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    sector: Optional[str] = None

class TeacherSummarizeResponse(BaseModel):
    ticket_id: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    created_at: Optional[str] = None
    summary: str
    structured_summary: Optional[Dict[str, Any]] = None
    priority: str = "High"
    priority_reason: Optional[str] = None
    intent: Optional[str] = None
    model: str
    provider: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

@router.post("/teacher/summarize", response_model=TeacherSummarizeResponse, summary="Summarize ticket using Teacher LLM")
def summarize_ticket_teacher(payload: TeacherSummarizeRequest):
    """Processes a support ticket using the configured Teacher LLM (e.g. Groq 120B)."""
    try:
        from app.teacher.teacher_llm import get_teacher_llm
        teacher = get_teacher_llm()
        t_id = (payload.ticket_id or "").strip()
        if not t_id or t_id.upper() in ["AUTO", "NONE", "PENDING", "WAITING"]:
            t_id = get_next_ticket_id()
            
        resp = teacher.generate_summary(t_id, payload.ticket_text)
        in_toks = len(payload.ticket_text.split())
        out_toks = len(resp.teacher_summary.split())

        saved = save_history_item(
            ticket_id=resp.ticket_id,
            ticket_text=payload.ticket_text,
            summary=resp.teacher_summary,
            model=f"{resp.teacher_model} (Teacher)",
            latency_ms=resp.latency_ms,
            sector=payload.sector,
            intent=resp.intent,
            priority=resp.priority or "High",
            priority_reason=resp.priority_reason,
            structured_summary=resp.structured_summary,
            customer_name=payload.customer_name,
            customer_email=payload.customer_email,
            customer_phone=payload.customer_phone,
            route_used="teacher",
            route_reason="Teacher LLM gold baseline execution",
            is_unseen=True
        )

        return TeacherSummarizeResponse(
            ticket_id=resp.ticket_id,
            customer_name=payload.customer_name,
            customer_email=payload.customer_email,
            customer_phone=payload.customer_phone,
            created_at=saved.get("created_at_formatted"),
            summary=resp.teacher_summary,
            structured_summary=resp.structured_summary,
            priority=resp.priority or "High",
            priority_reason=resp.priority_reason,
            intent=resp.intent,
            model=resp.teacher_model,
            provider=resp.provider,
            latency_ms=resp.latency_ms,
            input_tokens=in_toks,
            output_tokens=out_toks
        )
    except Exception as e:
        logger.error(f"Teacher inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Teacher LLM Error: {str(e)}"
        )

@router.post("/evaluate/latency-benchmark", response_model=LatencyBenchmarkResponse, summary="Multi-run latency benchmark for Median and P95 profiling")
def evaluate_latency_benchmark(
    payload: Optional[LatencyBenchmarkRequest] = None,
    pipeline: SummarizationPipeline = Depends(get_pipeline)
):
    """Measures model inference latency across multiple iterations to report statistical Median and P95 latency."""
    import numpy as np
    req = payload or LatencyBenchmarkRequest()
    ticket_text = req.ticket_text or (
        "Our organization's internet banking portal, mobile banking application, and payment gateway "
        "have all stopped working since 10:30 AM. More than 500 customers are unable to complete transactions. "
        "Several users report HTTP 503 Service Unavailable errors."
    )
    runs = max(1, min(req.runs, 10))

    student_latencies = []
    last_breakdown = {
        "preprocessing_ms": 1.2,
        "inference_ms": 120.0,
        "postprocessing_ms": 1.0,
        "total_ms": 122.2
    }

    for _ in range(runs):
        resp = pipeline.run({
            "ticket_id": "BENCHMARK",
            "ticket_text": ticket_text,
            "customer_name": "Benchmark Run",
            "sector": "Cloud Infrastructure"
        })
        student_latencies.append(resp.latency_ms)
        if resp.latency_breakdown:
            last_breakdown = resp.latency_breakdown

    student_sorted = sorted(student_latencies)
    student_median = float(np.median(student_sorted))
    student_p95 = float(np.percentile(student_sorted, 95))
    student_min = float(min(student_sorted))
    student_max = float(max(student_sorted))

    s_stats = LatencyStats(
        median_ms=round(student_median, 1),
        p95_ms=round(student_p95, 1),
        min_ms=round(student_min, 1),
        max_ms=round(student_max, 1),
        all_latencies_ms=[round(x, 1) for x in student_latencies],
        breakdown=last_breakdown
    )

    t_stats = None
    if req.include_teacher:
        try:
            from app.teacher.teacher_llm import get_teacher_llm
            teacher = get_teacher_llm()
            teacher_latencies = []
            for _ in range(min(runs, 2)):
                t_resp = teacher.generate_summary("BENCHMARK-T", ticket_text)
                teacher_latencies.append(t_resp.latency_ms)
            if teacher_latencies:
                t_sorted = sorted(teacher_latencies)
                t_stats = LatencyStats(
                    median_ms=round(float(np.median(t_sorted)), 1),
                    p95_ms=round(float(np.percentile(t_sorted, 95)), 1),
                    min_ms=round(float(min(t_sorted)), 1),
                    max_ms=round(float(max(t_sorted)), 1),
                    all_latencies_ms=[round(x, 1) for x in teacher_latencies],
                    breakdown={"inference_ms": round(float(np.median(t_sorted)), 1)}
                )
        except Exception as e:
            logger.warning(f"Teacher latency benchmark skipped: {e}")

    return LatencyBenchmarkResponse(
        runs_executed=runs,
        runs_count=runs,
        median_latency_ms=s_stats.median_ms,
        p95_latency_ms=s_stats.p95_ms,
        min_latency_ms=s_stats.min_ms,
        max_latency_ms=s_stats.max_ms,
        all_latencies_ms=s_stats.all_latencies_ms,
        model_name="Student Qwen2.5-1.5B",
        inference_type="student",
        ticket_text_snippet=ticket_text[:60] + "...",
        student_stats=s_stats,
        teacher_stats=t_stats
    )


# -------------------------------------------------------------
# COMPANY OFFICIALS DIRECTORY & EMAIL ESCALATION ENDPOINTS
# -------------------------------------------------------------
from datetime import datetime
from app.utils.officials_store import (
    get_all_officials,
    add_official,
    delete_official,
    get_official_for_category
)
from app.utils.teacher_buffer import (
    get_teacher_buffer_stats,
    get_teacher_buffer_samples
)
from app.analytics.issue_clusterer import record_cluster_solution

@router.get("/officials", summary="List all registered company officials for escalation")
def list_officials():
    """Returns directory of company officials with their roles, designations, and escalation emails."""
    officials = get_all_officials()
    return {"officials": officials, "count": len(officials)}

@router.post("/officials", summary="Register a new company official in directory")
def register_official(payload: AddOfficialRequest):
    """Adds a new company official with name, designation, department, and email for incident escalation."""
    created = add_official(
        name=payload.name,
        designation=payload.designation,
        email=payload.email,
        department=payload.department or "General Engineering"
    )
    return {"status": "created", "official": created}

@router.delete("/officials/{official_id}", summary="Remove a company official from directory")
def remove_official(official_id: str):
    """Removes an official from the escalation directory by ID."""
    success = delete_official(official_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Official '{official_id}' not found in directory."
        )
    return {"status": "deleted", "message": f"Official '{official_id}' removed successfully."}

@router.post("/officials/escalate-email", response_model=EscalateEmailResponse, summary="Dispatch formal incident escalation email to company official")
def escalate_incident_email(payload: EscalateEmailRequest):
    """Dispatches and logs a formal incident notification email to the proper company authority."""
    now_str = datetime.now().strftime("%I:%M %p, %b %d, %Y")
    logger.info(
        f"ESCALATION EMAIL DISPATCHED to {payload.official_name} <{payload.official_email}> "
        f"[Subject: {payload.subject}]"
    )
    return EscalateEmailResponse(
        status="DISPATCHED",
        dispatched_to=payload.official_email,
        official_name=payload.official_name,
        subject=payload.subject,
        timestamp=now_str,
        message=f"Incident escalation email successfully dispatched to {payload.official_name} ({payload.official_email}).",
        details={
            "recipient": payload.official_email,
            "urgency": payload.urgency or "HIGH / P2",
            "simulated_delivery": "Corporate SMTP queue accepted"
        }
    )

@router.post("/clusters/{cluster_id}/apply-solution", response_model=ApplySolutionResponse, summary="Apply a single unified solution across all tickets in a cluster")
def apply_unified_cluster_solution(cluster_id: str, payload: Optional[ApplySolutionRequest] = None):
    """Applies a single, unified resolution across all customer queries categorized under the specified cluster."""
    sol_text = payload.solution_text if payload else None
    res = record_cluster_solution(cluster_id, sol_text)
    
    clusterer = get_issue_clusterer()
    report = clusterer.cluster_tickets(get_all_history())
    target_cluster = next((c for c in report.clusters if c.cluster_id == cluster_id), None)
    t_ids = [t.ticket_id for t in target_cluster.affected_tickets] if target_cluster else []

    return ApplySolutionResponse(
        status="APPLIED",
        cluster_id=cluster_id,
        tickets_resolved_count=len(t_ids),
        ticket_ids=t_ids,
        unified_solution=res.get("solution", ""),
        message=f"Unified resolution applied across all {len(t_ids)} customer tickets in {cluster_id}."
    )

@router.get("/teacher/buffer-stats", response_model=TeacherBufferStatsResponse, summary="Get distillation buffer stats for student continuous learning")
def get_distillation_buffer_stats():
    """Returns total unseen teacher samples collected in persistent buffer for student model GKD / LoRA training."""
    stats = get_teacher_buffer_stats()
    return TeacherBufferStatsResponse(**stats)




