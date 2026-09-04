import re
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
    priority: Optional[str] = "Medium"

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

    def validate_and_normalize(self, raw_ticket: Dict[str, Any]) -> TicketData:
        """Validates ticket dictionary structure and normalizes missing values."""
        ticket_id = str(raw_ticket.get("ticket_id", "")).strip()
        if not ticket_id:
            raise ValueError("Ticket missing required field 'ticket_id'.")

        ticket_text = raw_ticket.get("ticket_text")
        if ticket_text is None or str(ticket_text).strip() == "":
            raise ValueError(f"Ticket '{ticket_id}' missing or has empty 'ticket_text'.")

        cleaned_text = self.clean_text(str(ticket_text))
        summary = self.clean_text(str(raw_ticket["summary"])) if raw_ticket.get("summary") else None

        return TicketData(
            ticket_id=ticket_id,
            ticket_text=cleaned_text,
            summary=summary,
            sector=raw_ticket.get("sector") or "General Support",
            intent=raw_ticket.get("intent") or "Inquiry",
            category=raw_ticket.get("category") or "General",
            priority=raw_ticket.get("priority") or "Medium"
        )
