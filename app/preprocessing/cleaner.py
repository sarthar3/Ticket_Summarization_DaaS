import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

class TicketData(BaseModel):
    ticket_id: str
    ticket_text: str
    summary: Optional[str] = None
    sector: Optional[str] = "General Support"
    intent: Optional[str] = "Inquiry"
    category: Optional[str] = "General"
    priority: str = "Medium"

class TicketPreprocessor:
    """Preprocesses raw support tickets according to configurable limits and rules."""

    def __init__(self, max_input_tokens: int = 1024, max_output_tokens: int = 256, truncation_side: str = "right"):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.truncation_side = truncation_side

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # Replace multiple whitespaces/newlines with single spaces while preserving line breaks where helpful
        cleaned = text.strip()
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def classify_priority(self, ticket_text: str) -> str:
        """Automatically judges ticket priority based on complaint urgency indicators."""
        text_lower = ticket_text.lower()
        
        urgent_high_keywords = [
            "urgent", "asap", "emergency", "critical", "unable to login", "cannot login",
            "outage", "payroll", "double billing", "security breach", "denied", "error 500",
            "stuck", "failed", "crash", "unauthorized", "down", "refund", "system error"
        ]
        low_keywords = [
            "gdpr", "deletion", "feedback", "suggestion", "general inquiry",
            "documentation", "question"
        ]

        for kw in urgent_high_keywords:
            if kw in text_lower:
                return "High"
        for kw in low_keywords:
            if kw in text_lower:
                return "Low"
        return "Medium"

    def generate_ticket_id(self) -> str:
        """Automatically generates a unique ticket ID."""
        date_str = datetime.now().strftime("%Y%m%d")
        short_code = uuid.uuid4().hex[:5].upper()
        return f"TICK-{date_str}-{short_code}"

    def validate_and_normalize(self, raw_ticket: Dict[str, Any]) -> TicketData:
        """Validates ticket dictionary structure, generates missing ticket_id, and auto-classifies priority."""
        ticket_id = str(raw_ticket.get("ticket_id") or "").strip()
        if not ticket_id:
            ticket_id = self.generate_ticket_id()

        ticket_text = raw_ticket.get("ticket_text")
        if ticket_text is None or str(ticket_text).strip() == "":
            raise ValueError(f"Ticket '{ticket_id}' missing or has empty 'ticket_text'.")

        cleaned_text = self.clean_text(str(ticket_text))
        summary = self.clean_text(str(raw_ticket["summary"])) if raw_ticket.get("summary") else None

        # Auto-classify priority if not explicitly provided
        provided_priority = str(raw_ticket.get("priority") or "").strip()
        priority = provided_priority if provided_priority else self.classify_priority(cleaned_text)

        return TicketData(
            ticket_id=ticket_id,
            ticket_text=cleaned_text,
            summary=summary,
            sector=raw_ticket.get("sector") or "General Support",
            intent=raw_ticket.get("intent") or "Inquiry",
            category=raw_ticket.get("category") or "General",
            priority=priority
        )
