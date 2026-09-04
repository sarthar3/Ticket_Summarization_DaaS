from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class SummarizeRequest(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier", json_schema_extra={"example": "TICK-1001"})
    ticket_text: str = Field(..., description="Full raw customer support ticket text", json_schema_extra={"example": "User unable to log in after update."})
    sector: Optional[str] = Field(None, description="Industry sector (e.g. Fintech, E-commerce)")
    intent: Optional[str] = Field(None, description="Ticket intent category")
    category: Optional[str] = Field(None, description="Ticket classification category")
    priority: Optional[str] = Field(None, description="Ticket priority level")

class SummarizeResponse(BaseModel):
    ticket_id: str
    summary: str
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
