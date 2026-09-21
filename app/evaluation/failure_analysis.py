import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TicketFailureReport(BaseModel):
    ticket_id: str
    has_failure: bool
    failure_types: List[str]
    details: List[str]

class AggregateFailureSummary(BaseModel):
    total_audited: int
    total_failures: int
    failure_rate_pct: float
    category_counts: Dict[str, int]
    ticket_reports: List[TicketFailureReport]

class FailureAnalyzer:
    """Automated failure detection engine for LLM summarization outputs."""

    def __init__(self):
        # Regex pattern matchers for entities
        self.code_pattern = re.compile(r"\b(?:ERR-\d+|HTTP \d{3}|\d{3} \b|#\w+[-\w]*)\b", re.IGNORECASE)
        self.currency_pattern = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", re.IGNORECASE)
        self.number_pattern = re.compile(r"\b\d+\b")

    def detect_repetition(self, text: str) -> bool:
        """Detects n-gram repetitions or looping sentences."""
        words = text.lower().split()
        if len(words) < 6:
            return False
        # Check 3-gram repetition
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        return len(trigrams) != len(set(trigrams))

    def detect_hallucinated_entities(self, ticket_text: str, summary: str) -> List[str]:
        """Detects entities (codes, currency amounts, numbers) present in summary but absent in ticket text."""
        hallucinations = []
        ticket_lower = ticket_text.lower()
        
        # Check currency amounts
        summary_currencies = set(self.currency_pattern.findall(summary))
        for curr in summary_currencies:
            if curr.lower() not in ticket_lower:
                hallucinations.append(f"Hallucinated currency amount: {curr}")

        # Check codes / IDs
        summary_codes = set(self.code_pattern.findall(summary))
        for code in summary_codes:
            if code.lower() not in ticket_lower:
                hallucinations.append(f"Hallucinated code/ID: {code}")

        return hallucinations

    def detect_truncation(self, summary: str) -> bool:
        """Detects incomplete sentences at the end of generated output."""
        summary_trimmed = summary.strip()
        if not summary_trimmed:
            return True
        last_char = summary_trimmed[-1]
        return last_char not in [".", "!", "?", "\"", "'"]

    def audit_record(self, ticket_id: str, ticket_text: str, summary: str) -> TicketFailureReport:
        failure_types: List[str] = []
        details: List[str] = []

        # 1. Hallucination Check
        hallucinations = self.detect_hallucinated_entities(ticket_text, summary)
        if hallucinations:
            failure_types.append("Hallucination")
            details.extend(hallucinations)

        # 2. Repetition Check
        if self.detect_repetition(summary):
            failure_types.append("Repetition Loop")
            details.append("Detected repetitive n-gram or sentence loop.")

        # 3. Truncation Check
        if self.detect_truncation(summary):
            failure_types.append("Truncation Anomaly")
            details.append("Summary ends abruptly without complete terminal punctuation.")

        # 4. Length Anomaly Check
        if len(summary.strip()) < 15:
            failure_types.append("Length Defect")
            details.append(f"Summary excessively short ({len(summary.strip())} chars).")

        has_failure = len(failure_types) > 0
        return TicketFailureReport(
            ticket_id=ticket_id,
            has_failure=has_failure,
            failure_types=failure_types,
            details=details
        )

    def analyze_dataset_results(self, raw_results: List[Dict[str, Any]]) -> AggregateFailureSummary:
        reports: List[TicketFailureReport] = []
        category_counts: Dict[str, int] = {
            "Hallucination": 0,
            "Repetition Loop": 0,
            "Truncation Anomaly": 0,
            "Length Defect": 0
        }

        for item in raw_results:
            t_id = item.get("ticket_id", "UNKNOWN")
            t_text = item.get("ticket_text") or item.get("generated_summary") or ""
            g_summary = item.get("generated_summary", "")

            report = self.audit_record(t_id, t_text, g_summary)
            reports.append(report)

            if report.has_failure:
                for f_type in report.failure_types:
                    category_counts[f_type] = category_counts.get(f_type, 0) + 1

        total_audited = len(raw_results)
        total_failures = sum(1 for r in reports if r.has_failure)
        failure_rate = (total_failures / total_audited * 100.0) if total_audited > 0 else 0.0

        return AggregateFailureSummary(
            total_audited=total_audited,
            total_failures=total_failures,
            failure_rate_pct=round(failure_rate, 2),
            category_counts=category_counts,
            ticket_reports=reports
        )
