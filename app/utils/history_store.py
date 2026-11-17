import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "history.json"
_history_lock = Lock()

def _ensure_history_file():
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

import re

def get_next_ticket_id() -> str:
    """Calculates the next chronological sequential ticket ID (e.g. TICK-1001, TICK-1002)."""
    _ensure_history_file()
    with _history_lock:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    existing_nums = []
    for item in history:
        t_id = str(item.get("ticket_id", "")).strip()
        match = re.match(r"^(?:TICK|TCK)-(\d+)$", t_id, re.IGNORECASE)
        if match:
            try:
                existing_nums.append(int(match.group(1)))
            except ValueError:
                pass

    if existing_nums:
        next_num = max(existing_nums) + 1
    else:
        # Default starting sequence
        next_num = 1001

    return f"TICK-{next_num}"

def save_history_item(
    ticket_id: str,
    ticket_text: str,
    summary: str,
    model: str,
    latency_ms: float,
    sector: Optional[str] = None,
    domain: Optional[str] = None,
    intent: Optional[str] = None,
    issue_type: Optional[str] = None,
    urgency: Optional[str] = None,
    priority: Optional[str] = None,
    priority_reason: Optional[str] = None,
    customer_impact: Optional[str] = None,
    recommended_actions: Optional[List[str]] = None,
    structured_summary: Optional[Dict[str, Any]] = None,
    customer_name: Optional[str] = None,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
    created_at: Optional[str] = None,
    route_used: Optional[str] = None,
    route_reason: Optional[str] = None,
    is_unseen: Optional[bool] = False,
    latency_breakdown: Optional[Dict[str, float]] = None,
    issue_origin: Optional[str] = "COMPANY_SIDE",
    client_guidance: Optional[Dict[str, Any]] = None,
    escalation_details: Optional[Dict[str, Any]] = None,
    ticket_type: Optional[str] = None,
    severity: Optional[str] = None,
    confidence_score: Optional[float] = None,
    probable_cause: Optional[str] = None,
    cause_confidence: Optional[float] = None,
    status: Optional[str] = "new",
    teacher_summary: Optional[str] = None,
    teacher_latency_ms: Optional[float] = None,
    teacher_tokens: Optional[int] = None,
    teacher_model: Optional[str] = None,
    **extra_kwargs
) -> Dict[str, Any]:
    """Appends a new summarization record with customer details and routing metadata to history JSON file."""
    _ensure_history_file()

    # Auto-assign ticket ID if omitted or AUTO
    t_id = (ticket_id or "").strip()
    if not t_id or t_id.upper() in ["AUTO", "NONE", "AUTO-ASSIGN", "PENDING", "WAITING"]:
        t_id = get_next_ticket_id()

    now_utc = datetime.now(timezone.utc)
    timestamp_iso = created_at or now_utc.isoformat()
    formatted_time = now_utc.strftime("%I:%M %p, %b %d, %Y")

    # Ensure ticket_type and issue_origin are accurately set if missing
    if not ticket_type:
        try:
            from app.preprocessing.cleaner import TicketPreprocessor
            pre = TicketPreprocessor()
            tt, sev, conf, action_rec, prob_cause, cause_conf = pre.classify_ticket_type_and_severity(ticket_text, priority or "Medium", domain or sector or "General Support", issue_type or intent or "")
            ticket_type = tt
            severity = severity or sev
            confidence_score = confidence_score or conf
            probable_cause = probable_cause or prob_cause
            cause_confidence = cause_confidence or cause_conf
            if issue_origin is None or issue_origin == "COMPANY_SIDE" or ticket_type != "company_defect":
                orig, guid, esc = pre.classify_origin_and_guidance(ticket_text, issue_type or "", domain or "", priority or "")
                issue_origin = orig
                if client_guidance is None:
                    client_guidance = guid
                if escalation_details is None:
                    escalation_details = esc
        except Exception as e:
            logger.warning(f"Error auto-classifying ticket: {e}")
            ticket_type = ticket_type or "customer_request"
            if ticket_type != "company_defect":
                issue_origin = "CLIENT_SIDE"
    
    item = {
        "ticket_id": t_id,
        "timestamp": timestamp_iso,
        "created_at_formatted": formatted_time,
        "customer_name": (customer_name or "").strip() or "Valued Customer",
        "customer_email": (customer_email or "").strip() or "customer@example.com",
        "customer_phone": (customer_phone or "").strip() or "+91 98765 43210",
        "ticket_text": ticket_text,
        "sector": domain or sector,
        "domain": domain or sector,
        "intent": issue_type or intent,
        "issue_type": issue_type or intent,
        "urgency": urgency,
        "priority": priority,
        "priority_reason": priority_reason,
        "customer_impact": customer_impact,
        "recommended_actions": recommended_actions,
        "issue_origin": issue_origin or ("COMPANY_SIDE" if ticket_type == "company_defect" else "CLIENT_SIDE"),
        "client_guidance": client_guidance,
        "escalation_details": escalation_details,
        "ticket_type": ticket_type,
        "severity": severity or "medium",
        "confidence_score": confidence_score if confidence_score is not None else 0.92,
        "probable_cause": probable_cause,
        "cause_confidence": cause_confidence,
        "status": status or "new",
        "summary": summary,
        "structured_summary": structured_summary,
        "model": model,
        "latency_ms": latency_ms,
        "latency_breakdown": latency_breakdown,
        "route_used": route_used,
        "route_reason": route_reason,
        "is_unseen": is_unseen,
        "teacher_summary": teacher_summary or extra_kwargs.get("teacher_summary"),
        "teacher_latency_ms": teacher_latency_ms or extra_kwargs.get("teacher_latency_ms"),
        "teacher_tokens": teacher_tokens or extra_kwargs.get("teacher_tokens"),
        "teacher_model": teacher_model or extra_kwargs.get("teacher_model")
    }
    
    with _history_lock:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
            
        history.insert(0, item)  # Latest items first
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
            
    return item

def get_all_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves stored ticket summarization history."""
    _ensure_history_file()
    with _history_lock:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    if limit and limit > 0:
        return history[:limit]
    return history

def get_history_by_ticket_id(ticket_id: str) -> Optional[Dict[str, Any]]:
    """Finds history item for a specific ticket_id."""
    history = get_all_history()
    for item in history:
        if item.get("ticket_id") == ticket_id:
            return item
    return None

def update_history_item(ticket_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Updates specific fields of an existing ticket record in history."""
    _ensure_history_file()
    with _history_lock:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

        target = None
        for item in history:
            if item.get("ticket_id") == ticket_id:
                item.update(updates)
                target = item
                break

        if target is not None:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return target

def clear_history() -> bool:
    """Clears all history records."""
    _ensure_history_file()
    with _history_lock:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    return True
