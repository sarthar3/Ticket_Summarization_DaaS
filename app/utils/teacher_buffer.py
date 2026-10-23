import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from threading import Lock

BUFFER_FILE = Path(__file__).resolve().parents[2] / "data" / "teacher_unseen_buffer.jsonl"
_buffer_lock = Lock()

def _ensure_buffer_file():
    BUFFER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not BUFFER_FILE.exists():
        BUFFER_FILE.touch()

def append_teacher_sample(
    ticket_id: str,
    ticket_text: str,
    teacher_summary: str,
    domain: Optional[str] = None,
    issue_type: Optional[str] = None,
    priority: Optional[str] = None,
    structured_summary: Optional[Dict[str, Any]] = None,
    model: str = "openai/gpt-oss-120b",
    provider: str = "groq",
    latency_ms: float = 0.0
) -> Dict[str, Any]:
    """Appends an unseen/out-of-distribution ticket with gold Teacher LLM supervision
    to the persistent distillation buffer for subsequent student fine-tuning / GKD."""
    _ensure_buffer_file()

    record = {
        "ticket_id": ticket_id,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "ticket_text": ticket_text,
        "summary": teacher_summary,
        "teacher_summary": teacher_summary,
        "domain": domain or "General",
        "sector": domain or "General",
        "issue_type": issue_type or "Novel Domain Pattern",
        "intent": issue_type or "Novel Domain Pattern",
        "priority": priority or "P1 · Critical",
        "teacher_model": model,
        "teacher_provider": provider,
        "teacher_latency_ms": latency_ms,
        "structured_summary": structured_summary or {}
    }

    with _buffer_lock:
        with open(BUFFER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return record

def get_teacher_buffer_stats() -> Dict[str, Any]:
    """Returns the total number of collected teacher distillation samples available for student learning."""
    _ensure_buffer_file()
    count = 0
    recent_samples = []

    with _buffer_lock:
        try:
            with open(BUFFER_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    l = line.strip()
                    if l:
                        count += 1
                        if len(recent_samples) < 5:
                            recent_samples.append(json.loads(l))
        except Exception:
            pass

    return {
        "buffer_file": str(BUFFER_FILE.name),
        "total_unseen_collected": count,
        "ready_for_distillation": count >= 1,
        "recommended_batch_for_gkd": 5,
        "recent_samples": recent_samples
    }

def get_teacher_buffer_samples(limit: int = 50) -> List[Dict[str, Any]]:
    """Reads collected teacher samples from buffer."""
    _ensure_buffer_file()
    samples = []
    with _buffer_lock:
        try:
            with open(BUFFER_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    l = line.strip()
                    if l:
                        samples.append(json.loads(l))
        except Exception:
            pass
    return samples[-limit:]
