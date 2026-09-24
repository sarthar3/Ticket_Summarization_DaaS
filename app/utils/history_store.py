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

def save_history_item(
    ticket_id: str,
    ticket_text: str,
    summary: str,
    model: str,
    latency_ms: float,
    sector: Optional[str] = None,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    structured_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Appends a new summarization record to history JSON file."""
    _ensure_history_file()
    
    item = {
        "ticket_id": ticket_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_text": ticket_text,
        "sector": sector,
        "intent": intent,
        "priority": priority,
        "summary": summary,
        "structured_summary": structured_summary,
        "model": model,
        "latency_ms": latency_ms
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

def clear_history() -> bool:
    """Clears all history records."""
    _ensure_history_file()
    with _history_lock:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    return True
