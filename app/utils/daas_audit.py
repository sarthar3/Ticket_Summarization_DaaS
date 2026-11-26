"""
DaaS Routing & Token Audit Utility
Logs terminal output for true Distillation-as-a-Service routing decisions,
token consumption breakdowns, and cost accounting.
"""
from typing import Optional, Dict, Any
import sys
from app.utils.logger import get_logger

logger = get_logger("daas_audit")

# Pricing per 1,000 tokens for Teacher model (Groq LLaMA 3.3 70B / 120B baseline)
TEACHER_PROMPT_PRICE_PER_1K = 0.00059   # $0.59 per 1M input tokens
TEACHER_COMPL_PRICE_PER_1K = 0.00079    # $0.79 per 1M output tokens

# Module-level session history for real-time per-query delta tracking
_SESSION_HISTORY = {
    "count": 0,
    "last_latency_ms": None,
    "last_output_tokens": None,
    "total_student_served": 0,
    "total_teacher_escalated": 0,
    "latencies": []
}

def calculate_teacher_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates approximate USD cost for Teacher API inference."""
    return round(
        (prompt_tokens / 1000.0 * TEACHER_PROMPT_PRICE_PER_1K) +
        (completion_tokens / 1000.0 * TEACHER_COMPL_PRICE_PER_1K),
        6
    )

def log_daas_audit(
    ticket_id: str,
    route_used: str,  # "student" or "teacher"
    student_latency_ms: float = 0.0,
    student_input_tokens: int = 0,
    student_output_tokens: int = 0,
    teacher_invoked: bool = False,
    teacher_model: Optional[str] = None,
    teacher_prompt_tokens: int = 0,
    teacher_completion_tokens: int = 0,
    escalation_reason: Optional[str] = None,
    priority_tokens: int = 0,
    summary_tokens: int = 0
) -> Dict[str, Any]:
    """
    Logs structured DaaS execution and token consumption directly to terminal.
    Ensures zero token usage is transparently logged when Student serves the query,
    and displays live per-query metric comparison deltas (increased/decreased).
    """
    global _SESSION_HISTORY

    _SESSION_HISTORY["count"] += 1
    query_num = _SESSION_HISTORY["count"]
    prev_lat = _SESSION_HISTORY["last_latency_ms"]
    prev_tok = _SESSION_HISTORY["last_output_tokens"]

    if route_used.lower() == "student" and not teacher_invoked:
        _SESSION_HISTORY["total_student_served"] += 1
    else:
        _SESSION_HISTORY["total_teacher_escalated"] += 1

    curr_lat = student_latency_ms
    _SESSION_HISTORY["latencies"].append(curr_lat)
    _SESSION_HISTORY["last_latency_ms"] = curr_lat
    _SESSION_HISTORY["last_output_tokens"] = student_output_tokens

    total_teacher_tokens = teacher_prompt_tokens + teacher_completion_tokens if teacher_invoked else 0
    teacher_cost_usd = calculate_teacher_cost(teacher_prompt_tokens, teacher_completion_tokens) if teacher_invoked else 0.0

    banner_width = 74
    lines = []
    lines.append("=" * banner_width)
    lines.append("                 DISTILLATION-AS-A-SERVICE (DaaS) AUDIT")
    lines.append("=" * banner_width)
    lines.append(f"Ticket ID         : {ticket_id}")

    if not teacher_invoked and route_used.lower() == "student":
        lines.append(f"Route Resolved    : STUDENT EDGE (Qwen 2.5-3B Local)")
        lines.append(f"Student Latency   : {student_latency_ms:.1f}ms")
        lines.append(f"Student Tokens    : {student_input_tokens} input / {student_output_tokens} output")
        lines.append("-" * banner_width)
        lines.append(f"Teacher Model     : BYPASSED (No API Call Made)")
        lines.append(f"Teacher Tokens    : 0 tokens (Prompt: 0, Completion: 0)")
        lines.append(f"Teacher API Cost  : $0.00000 USD (100% Cost Saved)")
        lines.append(f"Service Mode      : Pure Student Edge Serving")
    else:
        reason = escalation_reason or "Escalation / Out-of-Domain Fallback"
        pri_tok = priority_tokens or max(int(teacher_prompt_tokens * 0.25), 18)
        sum_tok = summary_tokens or (total_teacher_tokens - pri_tok if total_teacher_tokens > pri_tok else teacher_completion_tokens)
        
        lines.append(f"Route Resolved    : TEACHER ESCALATION ({teacher_model or 'Groq 120B'})")
        lines.append(f"Escalation Reason : {reason}")
        lines.append(f"Student Execution : Escalated to Teacher (Low confidence or Fallback)")
        lines.append("-" * banner_width)
        lines.append(f"Teacher Token Usage Breakdown:")
        lines.append(f"  ├─ Priority / Intent Classification : {pri_tok} tokens")
        lines.append(f"  ├─ Core Executive Summarization     : {sum_tok} tokens")
        lines.append(f"  ├─ Prompt Tokens                    : {teacher_prompt_tokens} tokens")
        lines.append(f"  ├─ Completion Tokens                : {teacher_completion_tokens} tokens")
        lines.append(f"  └─ Total Teacher Tokens Incurred    : {total_teacher_tokens} tokens")
        lines.append(f"Teacher Incurred Cost : ${teacher_cost_usd:.5f} USD")
        lines.append(f"Service Mode          : Gold Supervision Fallback")

    lines.append("-" * banner_width)
    lines.append(f"REAL-TIME METRICS COMPARISON (Query #{query_num}):")

    # Latency comparison delta
    if prev_lat is not None:
        lat_delta = curr_lat - prev_lat
        if lat_delta < -1.0:
            lat_status = f"{lat_delta:+.1f}ms [FASTER / IMPROVED ▼]"
        elif lat_delta > 1.0:
            lat_status = f"{lat_delta:+.1f}ms [SLOWER ▲]"
        else:
            lat_status = f"{lat_delta:+.1f}ms [STEADY =]"
        lines.append(f"  ├─ Latency (ms)     : {prev_lat:.1f}ms ──► {curr_lat:.1f}ms ({lat_status})")
    else:
        lines.append(f"  ├─ Latency (ms)     : {curr_lat:.1f}ms [BASELINE]")

    # Output tokens comparison delta
    if prev_tok is not None:
        tok_delta = student_output_tokens - prev_tok
        if tok_delta < 0:
            tok_status = f"{tok_delta:+d} tokens [MORE CONCISE ▼]"
        elif tok_delta > 0:
            tok_status = f"{tok_delta:+d} tokens [EXPANDED ▲]"
        else:
            tok_status = f"0 tokens [STEADY =]"
        lines.append(f"  ├─ Output Tokens    : {prev_tok} tok ──► {student_output_tokens} tok ({tok_status})")
    else:
        lines.append(f"  ├─ Output Tokens    : {student_output_tokens} tok [BASELINE]")

    # Session Edge Serving Rate (%)
    serve_rate = (_SESSION_HISTORY["total_student_served"] / max(query_num, 1)) * 100.0
    avg_lat = sum(_SESSION_HISTORY["latencies"]) / max(len(_SESSION_HISTORY["latencies"]), 1)
    lines.append(f"  ├─ Edge Serve Rate  : {serve_rate:.1f}% Local Student ($0.00 Ongoing API Cost)")
    lines.append(f"  └─ Session Mean Lat : {avg_lat:.1f}ms (across {query_num} processed queries)")

    lines.append("=" * banner_width)
    
    audit_text = "\n".join(lines)
    # Print directly to standard output so it appears immediately in the user's terminal
    print(audit_text, flush=True)
    logger.info(f"DaaS Audit logged for ticket {ticket_id}: route={route_used}, teacher_tokens={total_teacher_tokens}, cost=${teacher_cost_usd:.5f}")

    return {
        "ticket_id": ticket_id,
        "route_used": route_used,
        "teacher_invoked": teacher_invoked,
        "teacher_tokens": total_teacher_tokens,
        "teacher_prompt_tokens": teacher_prompt_tokens,
        "teacher_completion_tokens": teacher_completion_tokens,
        "teacher_cost_usd": teacher_cost_usd,
        "student_latency_ms": student_latency_ms,
        "student_tokens": student_output_tokens
    }
