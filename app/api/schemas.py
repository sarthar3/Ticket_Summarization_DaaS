from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

class SummarizeRequest(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Unique ticket identifier; auto-generated chronologically if omitted", json_schema_extra={"example": "TICK-1001"})
    customer_name: Optional[str] = Field(None, description="Full customer name", json_schema_extra={"example": "Rajesh Kumar"})
    customer_email: Optional[str] = Field(None, description="Customer contact email", json_schema_extra={"example": "rajesh.k@example.com"})
    customer_phone: Optional[str] = Field(None, description="Customer phone number", json_schema_extra={"example": "+91 98765 43210"})
    ticket_text: str = Field(..., description="Full raw customer support ticket text / complaint query", json_schema_extra={"example": "Customer reported being unable to login to mobile banking app following v4.2 update."})
    sector: Optional[str] = Field(None, description="Industry sector (e.g. Banking, E-commerce, Telecom)")
    intent: Optional[str] = Field(None, description="Ticket intent category")
    category: Optional[str] = Field(None, description="Ticket classification category")
    priority: Optional[str] = Field(None, description="Ticket priority level; auto-classified if omitted")

class StructuredSummaryDetails(BaseModel):
    core_issue: str = Field(..., description="Main technical problem reported by customer", json_schema_extra={"example": "Multiple banking services returning HTTP 503 Service Unavailable."})
    domain: Optional[str] = Field("Cloud Infrastructure", description="Target industry/technical domain", json_schema_extra={"example": "Cloud Infrastructure"})
    issue_type: Optional[str] = Field("Multi-Service Outage / HTTP 503", description="Specific technical problem type", json_schema_extra={"example": "Multi-Service Outage / HTTP 503"})
    urgency: Optional[str] = Field("Critical", description="Operational urgency level", json_schema_extra={"example": "Critical"})
    priority: Optional[str] = Field("P1 · Critical", description="Deterministic priority classification", json_schema_extra={"example": "P1 · Critical"})
    customer_impact: Optional[str] = Field(None, description="Quantified customer & business operational impact", json_schema_extra={"example": "More than 500 customers affected; banking access disrupted."})
    recommended_actions: Optional[List[str]] = Field(None, description="Explicit action items for incident response / engineering")
    customer_intent: str = Field(..., description="Customer intent or priority context summary", json_schema_extra={"example": "Cloud Infrastructure / Multi-Service Outage / HTTP 503 (P1 · Critical)"})
    key_action_items: str = Field(..., description="Recommended resolution or follow-up steps", json_schema_extra={"example": "Escalate to SRE team and restore service."})

class SummarizeResponse(BaseModel):
    ticket_id: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    created_at: Optional[str] = None
    summary: str
    structured_summary: Optional[StructuredSummaryDetails] = None
    domain: Optional[str] = None
    issue_type: Optional[str] = None
    urgency: Optional[str] = None
    priority: str
    priority_reason: Optional[str] = None
    customer_impact: Optional[str] = None
    recommended_actions: Optional[List[str]] = None
    intent: Optional[str] = None
    issue_origin: Optional[str] = "COMPANY_SIDE"  # "CLIENT_SIDE" or "COMPANY_SIDE"
    client_guidance: Optional[Dict[str, Any]] = None
    escalation_details: Optional[Dict[str, Any]] = None
    model: str
    route_used: Optional[str] = None
    route_reason: Optional[str] = None
    is_unseen: Optional[bool] = False
    latency_ms: float
    latency_breakdown: Optional[Dict[str, float]] = None
    input_tokens: int
    output_tokens: int

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool
    model_name: str
    precision: str
    device: str

class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int

class HistoryItem(BaseModel):
    ticket_id: str
    timestamp: str
    created_at_formatted: Optional[str] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    ticket_text: str
    sector: Optional[str] = None
    domain: Optional[str] = None
    intent: Optional[str] = None
    issue_type: Optional[str] = None
    urgency: Optional[str] = None
    priority: Optional[str] = None
    priority_reason: Optional[str] = None
    customer_impact: Optional[str] = None
    recommended_actions: Optional[List[str]] = None
    issue_origin: Optional[str] = "COMPANY_SIDE"
    client_guidance: Optional[Dict[str, Any]] = None
    escalation_details: Optional[Dict[str, Any]] = None
    summary: str
    structured_summary: Optional[StructuredSummaryDetails] = None
    model: str
    route_used: Optional[str] = None
    route_reason: Optional[str] = None
    is_unseen: Optional[bool] = False
    latency_ms: float
    latency_breakdown: Optional[Dict[str, float]] = None

class LatencyBenchmarkRequest(BaseModel):
    ticket_text: Optional[str] = Field(
        default="Our organization's internet banking portal, mobile banking application, and payment gateway have all stopped working since 10:30 AM. More than 500 customers are unable to complete transactions. Several users report HTTP 503 Service Unavailable errors.",
        description="Ticket text to benchmark across multiple iterations"
    )
    runs: int = Field(3, description="Number of execution runs for statistical aggregation (median & P95)")
    route: str = Field("student", description="Target route to benchmark ('student' or 'teacher')")
    include_teacher: bool = Field(False, description="Whether to include Teacher model in benchmark")

class LatencyStats(BaseModel):
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    all_latencies_ms: List[float]
    breakdown: Optional[Dict[str, float]] = None

class LatencyBenchmarkResponse(BaseModel):
    runs_executed: int = Field(default=3)
    runs_count: Optional[int] = Field(default=3)
    median_latency_ms: Optional[float] = Field(default=0.0)
    p95_latency_ms: Optional[float] = Field(default=0.0)
    min_latency_ms: Optional[float] = Field(default=0.0)
    max_latency_ms: Optional[float] = Field(default=0.0)
    all_latencies_ms: Optional[List[float]] = Field(default_factory=list)
    model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct"
    inference_type: Optional[str] = "student"
    ticket_text_snippet: Optional[str] = None
    student_stats: Optional[LatencyStats] = None
    teacher_stats: Optional[LatencyStats] = None
    note: str = "Statistical multi-run benchmark measuring median and P95 latency separate from cold-starts."

class HistoryListResponse(BaseModel):
    total_count: int
    items: List[HistoryItem]

class NextTicketIdResponse(BaseModel):
    next_ticket_id: str
    auto_assigned: bool = True
    info: str = "Chronological next ticket sequence ID"

class AffectedTicketSchema(BaseModel):
    ticket_id: str
    timestamp: str
    formatted_time: str
    customer_name: str
    customer_email: str
    customer_phone: str
    priority: str
    core_issue: str
    summary: str
    raw_query: str

class CommonIssueClusterSchema(BaseModel):
    cluster_id: str
    issue_title: str
    category: str
    affected_count: int
    affected_pct: float
    severity: str
    is_company_wide_issue: bool
    issue_origin: Optional[str] = "COMPANY_SIDE"
    common_issue_summary: str
    company_root_cause_diagnosis: str
    recommended_action: str
    unified_solution: Optional[str] = None
    escalation_official: Optional[Dict[str, Any]] = None
    escalation_email_subject: Optional[str] = None
    escalation_email_body: Optional[str] = None
    client_guidance: Optional[Dict[str, Any]] = None
    resolution_applied: Optional[bool] = False
    resolution_timestamp: Optional[str] = None
    affected_tickets: List[AffectedTicketSchema]

class CompanyIssuesReportSchema(BaseModel):
    total_tickets_analyzed: int
    total_customers_affected: int
    unique_issue_clusters_count: int
    company_outages_detected: int
    has_active_outage: bool
    has_unverified_outage_report: Optional[bool] = False
    unverified_outage_count: Optional[int] = 0
    company_status: Optional[str] = "HEALTHY"
    executive_brief: str
    generated_at: str
    clusters: List[CommonIssueClusterSchema]
    chronological_queue: List[AffectedTicketSchema]

class OfficialSchema(BaseModel):
    id: str
    name: str
    designation: str
    email: str
    department: str
    is_primary: Optional[bool] = False

class AddOfficialRequest(BaseModel):
    name: str = Field(..., description="Full name of company official")
    designation: str = Field(..., description="Role/title, e.g. Head of SRE")
    email: str = Field(..., description="Corporate email address")
    department: Optional[str] = Field("General Engineering", description="Department / functional domain")

class EscalateEmailRequest(BaseModel):
    cluster_id: Optional[str] = None
    ticket_id: Optional[str] = None
    official_name: str
    official_email: str
    official_designation: Optional[str] = None
    subject: str
    body: Optional[str] = None
    body_text: Optional[str] = None
    urgency: Optional[str] = None

class EscalateEmailResponse(BaseModel):
    status: str = "DISPATCHED"
    dispatched_to: str
    official_name: str
    subject: str
    timestamp: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ApplySolutionRequest(BaseModel):
    cluster_id: Optional[str] = None
    solution_text: Optional[str] = None

class ApplySolutionResponse(BaseModel):
    status: str = "APPLIED"
    cluster_id: str
    tickets_resolved_count: int
    affected_count: Optional[int] = None
    ticket_ids: List[str]
    unified_solution: str
    message: str

    def model_post_init(self, __context: Any) -> None:
        if self.affected_count is None:
            self.affected_count = self.tickets_resolved_count

class TeacherBufferStatsResponse(BaseModel):
    buffer_file: str
    buffer_path: Optional[str] = None
    total_unseen_collected: int
    total_unseen_samples: Optional[int] = None
    ready_for_distillation: bool
    recommended_batch_for_gkd: int
    recent_samples: List[Dict[str, Any]] = []
    latest_samples: Optional[List[Dict[str, Any]]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.buffer_path is None:
            self.buffer_path = self.buffer_file
        if self.total_unseen_samples is None:
            self.total_unseen_samples = self.total_unseen_collected
        if self.latest_samples is None:
            self.latest_samples = self.recent_samples

