from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class SummarizeRequest(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Unique ticket identifier; auto-generated if omitted", json_schema_extra={"example": "TICK-1001"})
    ticket_text: str = Field(..., description="Full raw customer support ticket text", json_schema_extra={"example": "User unable to log in after update."})
    sector: Optional[str] = Field(None, description="Industry sector (e.g. Fintech, E-commerce)")
    intent: Optional[str] = Field(None, description="Ticket intent category")
    category: Optional[str] = Field(None, description="Ticket classification category")
    priority: Optional[str] = Field(None, description="Ticket priority level; auto-classified if omitted")

class StructuredSummaryDetails(BaseModel):
    core_issue: str = Field(..., description="Main technical problem reported by customer", json_schema_extra={"example": "Unable to login to mobile banking app following v4.2 update on iOS 17.4."})
    customer_intent: str = Field(..., description="Customer intent or priority context", json_schema_extra={"example": "Urgent technical support / Payroll access."})
    key_action_items: str = Field(..., description="Recommended resolution or follow-up steps", json_schema_extra={"example": "Reset authentication credentials and grant temporary payroll access."})

class SummarizeResponse(BaseModel):
    ticket_id: str
    summary: str
    structured_summary: Optional[StructuredSummaryDetails] = None
    priority: str
    model: str
    latency_ms: float
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
    ticket_text: str
    sector: Optional[str] = None
    intent: Optional[str] = None
    priority: Optional[str] = None
    summary: str
    structured_summary: Optional[StructuredSummaryDetails] = None
    model: str
    latency_ms: float

class HistoryListResponse(BaseModel):
    total_count: int
    items: list[HistoryItem]

