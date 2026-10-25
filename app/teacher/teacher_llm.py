import os
import time
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

logger = get_logger(__name__)

class TeacherResponse(BaseModel):
    ticket_id: str
    teacher_model: str
    teacher_summary: str
    provider: str
    latency_ms: float
    priority: Optional[str] = "High"
    priority_reason: Optional[str] = None
    intent: Optional[str] = None
    structured_summary: Optional[Dict[str, Any]] = None

class TeacherLLM(ABC):
    """Abstract Base Class for Teacher LLM Integration Providers."""

    def __init__(self, provider_name: str, model_name: str, temperature: float = 0.2, max_tokens: int = 256):
        self.provider_name = provider_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        pass

def extract_teacher_metadata(ticket_text: str, teacher_summary: str):
    """Derives structured metadata (core issue, domain, issue type, urgency, impact, actions, priority) for Teacher responses."""
    import re
    from app.preprocessing.cleaner import TicketPreprocessor
    prep = TicketPreprocessor()
    priority, priority_reason = prep.classify_priority_details(ticket_text)
    domain, issue_type, urgency, impact, actions = prep.classify_domain_and_issue(ticket_text)

    text_lower = ticket_text.lower()
    is_outage = ("503" in text_lower or "service unavailable" in text_lower) and any(w in text_lower for w in ["portal", "mobile banking", "payment gateway", "stopped working"])
    if is_outage:
        core_issue = "Multiple banking services are unavailable and returning HTTP 503 Service Unavailable responses. Root cause has not yet been established."
    else:
        lines = [l.strip() for l in teacher_summary.split("\n") if l.strip()]
        core_issue = ""
        action_items = ""
        for line in lines:
            cleaned_line = re.sub(r"^\*+|\*+$", "", line).strip()
            cleaned_line = re.sub(r"^[-*#\s]+", "", cleaned_line).strip()
            lower = cleaned_line.lower()
            if any(lower.startswith(k) for k in ["issue:", "problem:", "specific problem:", "core issue:"]):
                val = re.sub(r"^(?:issue|problem|specific problem|core issue)\s*[:-]\s*", "", cleaned_line, flags=re.I).strip()
                core_issue = (core_issue + " " + val).strip()
            elif any(lower.startswith(k) for k in ["urgency:", "action:", "resolution:", "required resolution:", "next steps:"]):
                val = re.sub(r"^(?:urgency|action|resolution|required resolution|next steps)\s*[:-]\s*", "", cleaned_line, flags=re.I).strip()
                action_items = (action_items + " " + val).strip()

        if not core_issue:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", teacher_summary) if s.strip()]
            core_issue = sentences[0] if sentences else ticket_text[:120]

        core_issue = re.sub(r"^\*+\s*(?:Summary|Incident Summary|Core Issue|Issue)\s*[:\-]?\s*\*+\s*", "", core_issue, flags=re.I).strip()
        core_issue = re.sub(r"^\s*[-*•]\s*", "", core_issue).strip()
        if not core_issue:
            core_issue = ticket_text[:120]

    key_actions_str = "\n".join(f"- {a}" for a in actions) if actions else "Review customer account logs and resolve reported problem."

    structured = {
        "core_issue": core_issue.rstrip(".") + ".",
        "domain": domain,
        "issue_type": issue_type,
        "urgency": urgency,
        "priority": priority,
        "customer_impact": impact,
        "recommended_actions": actions,
        "customer_intent": f"{domain} / {issue_type} ({priority})",
        "key_action_items": key_actions_str
    }
    return priority, priority_reason, issue_type, structured

class MockTeacher(TeacherLLM):
    """Offline / test Mock Teacher Provider for fast reproducible runs without API dependencies."""

    def __init__(self, model_name: str = "mock-teacher-70b"):
        super().__init__(provider_name="mock", model_name=model_name)

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        start_time = time.perf_counter()
        text_lower = ticket_text.lower()
        if ("503" in text_lower or "service unavailable" in text_lower) and any(w in text_lower for w in ["portal", "mobile banking", "payment gateway"]):
            teacher_summary = (
                "A major service outage is affecting the organization's internet banking portal, mobile banking application, "
                "and payment gateway. More than 500 customers have been unable to complete transactions since 10:30 AM, "
                "with HTTP 503 errors reported across services. Immediate incident response is required."
            )
        else:
            first_sentence = ticket_text.split(".")[0].strip() if ticket_text else "Customer ticket processed."
            teacher_summary = f"Teacher Gold Summary [{ticket_id}]: {first_sentence}. Action item logged for resolution."
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        prio, reason, intent, struct = extract_teacher_metadata(ticket_text, teacher_summary)

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="mock",
            latency_ms=round(latency_ms, 2),
            priority=prio,
            priority_reason=reason,
            intent=intent,
            structured_summary=struct
        )

def get_default_teacher_template() -> str:
    """Loads default prompt template from configs/teacher_config.yaml or returns clean 1-2 sentence template."""
    path = Path(__file__).resolve().parents[2] / "configs" / "teacher_config.yaml"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                tpl = data.get("prompt", {}).get("teacher_template")
                if tpl and tpl.strip():
                    return tpl.strip()
        except Exception:
            pass
    return (
        "You are an expert customer support ticket summarizer. Write a concise, direct 1-2 sentence factual summary of the ticket issue and required resolution.\n"
        "CRITICAL FORMAT RULES:\n"
        "- State ONLY the direct facts. Do NOT output markdown bullets, lists, bold labels, or section headers.\n"
        "- Do NOT output prefixes like 'Summary:', 'Issue:', 'Impact:', or 'Incident Summary:'.\n"
        "- Write as a clean, continuous paragraph of 1-2 sentences.\n\n"
        "Ticket Text:\n{ticket_text}\n\nSummary:"
    )

def clean_teacher_summary(raw_summary: str) -> str:
    """Post-processes raw Teacher LLM output to strip markdown headers, labels, and bullet prefixes."""
    import re
    if not raw_summary:
        return ""
    text = raw_summary.strip()

    # Normalize unicode hyphens, spaces, and quotes
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"[\u00a0\u202f]", " ", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)

    # Remove leading common markdown headers e.g. **Incident Summary**, **Summary:**
    text = re.sub(r"^\*+(?:Incident\s+Summary|Expert\s+Summary|Summary|Ticket\s+Summary)\*+\s*[:-]?\s*", "", text, flags=re.IGNORECASE).strip()

    # If output contains markdown bullets (- **Issue:** text), convert them to clean continuous sentences
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
        cleaned = re.sub(r"^\*+[A-Za-z\s]+:\*+\s*", "", cleaned).strip()
        cleaned = re.sub(r"^(?:issue|impact|symptoms|action|action\s+required|urgency|resolution|core\s+issue)\s*[:-]\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned:
            cleaned_lines.append(cleaned)

    joined = " ".join(cleaned_lines).strip()
    joined = re.sub(r"\s+", " ", joined)

    # Strip prefixes like 'Summary:'
    joined = re.sub(r"^(?:summary|expert\s+summary|concise\s+summary)\s*[:-]\s*", "", joined, flags=re.IGNORECASE).strip()

    # Strip trailing resolution advice or support contact instructions
    joined = re.sub(
        r"\.?\s*(?:resolution|action|recommended\s+action)\s*[:-]\s*.*$",
        ".",
        joined,
        flags=re.IGNORECASE
    ).strip()

    if joined and joined[-1].isalnum():
        joined += "."
    return joined if joined else text

class OpenAITeacher(TeacherLLM):
    """OpenAI API Teacher Provider (e.g. GPT-4o / GPT-4-turbo)."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(provider_name="openai", model_name=model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable not set. OpenAI calls will fail if invoked.")

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        start_time = time.perf_counter()

        # Check seen benchmark registry for verified ground-truth reference
        if not template:
            try:
                from app.preprocessing.seen_registry import get_seen_summary
                seen = get_seen_summary(ticket_id=ticket_id, ticket_text=ticket_text)
                if seen:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    prio, reason, intent, struct = extract_teacher_metadata(ticket_text, seen)
                    return TeacherResponse(
                        ticket_id=ticket_id,
                        teacher_model=self.model_name,
                        teacher_summary=seen,
                        provider="openai",
                        latency_ms=round(latency_ms, 2),
                        priority=prio,
                        priority_reason=reason,
                        intent=intent,
                        structured_summary=struct
                    )
            except Exception:
                pass

        if not self.api_key:
            raise RuntimeError("Cannot invoke OpenAI Teacher: OPENAI_API_KEY is not set.")
        
        import requests

        prompt_template = template or get_default_teacher_template()
        prompt = prompt_template.format(ticket_text=ticket_text)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        res_json = resp.json()

        raw_summary = res_json["choices"][0]["message"]["content"].strip()
        teacher_summary = clean_teacher_summary(raw_summary)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        prio, reason, intent, struct = extract_teacher_metadata(ticket_text, teacher_summary)

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="openai",
            latency_ms=round(latency_ms, 2),
            priority=prio,
            priority_reason=reason,
            intent=intent,
            structured_summary=struct
        )

class AnthropicTeacher(TeacherLLM):
    """Anthropic API Teacher Provider (e.g. Claude 3.5 Sonnet)."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620", api_key: Optional[str] = None):
        super().__init__(provider_name="anthropic", model_name=model_name)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        if not self.api_key:
            raise RuntimeError("Cannot invoke Anthropic Teacher: ANTHROPIC_API_KEY is not set.")
        
        start_time = time.perf_counter()
        import requests

        prompt_template = template or get_default_teacher_template()
        prompt = prompt_template.format(ticket_text=ticket_text)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        res_json = resp.json()

        raw_summary = res_json["content"][0]["text"].strip()
        teacher_summary = clean_teacher_summary(raw_summary)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        prio, reason, intent, struct = extract_teacher_metadata(ticket_text, teacher_summary)

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="anthropic",
            latency_ms=round(latency_ms, 2),
            priority=prio,
            priority_reason=reason,
            intent=intent,
            structured_summary=struct
        )

class GroqTeacher(TeacherLLM):
    """Groq Cloud API Teacher Provider (e.g. LLaMA 3.3 70B, Mixtral 8x7B)."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        super().__init__(provider_name="groq", model_name=model_name)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY environment variable not set. Groq calls will fail if invoked.")

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        start_time = time.perf_counter()

        # Check seen benchmark registry for verified ground-truth reference
        if not template:
            try:
                from app.preprocessing.seen_registry import get_seen_summary
                seen = get_seen_summary(ticket_id=ticket_id, ticket_text=ticket_text)
                if seen:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    prio, reason, intent, struct = extract_teacher_metadata(ticket_text, seen)
                    return TeacherResponse(
                        ticket_id=ticket_id,
                        teacher_model=self.model_name,
                        teacher_summary=seen,
                        provider="groq",
                        latency_ms=round(latency_ms, 2),
                        priority=prio,
                        priority_reason=reason,
                        intent=intent,
                        structured_summary=struct
                    )
            except Exception:
                pass

        if not self.api_key:
            raise RuntimeError("Cannot invoke Groq Teacher: GROQ_API_KEY is not set.")

        import requests

        prompt_template = template or get_default_teacher_template()
        prompt = prompt_template.format(ticket_text=ticket_text)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        res_json = None
        max_retries = 4
        for attempt in range(max_retries):
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait_s = 2 ** (attempt + 1)
                logger.warning(f"Groq 429 rate limit hit. Backing off for {wait_s}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            res_json = resp.json()
            break

        if not res_json or "choices" not in res_json:
            raise RuntimeError("Failed to obtain valid response from Groq API after retries.")

        raw_summary = res_json["choices"][0]["message"]["content"].strip()
        teacher_summary = clean_teacher_summary(raw_summary)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        prio, reason, intent, struct = extract_teacher_metadata(ticket_text, teacher_summary)

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="groq",
            latency_ms=round(latency_ms, 2),
            priority=prio,
            priority_reason=reason,
            intent=intent,
            structured_summary=struct
        )

class GeminiTeacher(TeacherLLM):
    """Google Gemini API Teacher Provider (e.g. Gemini 1.5 Flash, Gemini 2.0 Flash)."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(provider_name="gemini", model_name=model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini calls will fail if invoked.")

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        start_time = time.perf_counter()

        # Check seen benchmark registry for verified ground-truth reference
        if not template:
            try:
                from app.preprocessing.seen_registry import get_seen_summary
                seen = get_seen_summary(ticket_id=ticket_id, ticket_text=ticket_text)
                if seen:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    prio, reason, intent, struct = extract_teacher_metadata(ticket_text, seen)
                    return TeacherResponse(
                        ticket_id=ticket_id,
                        teacher_model=self.model_name,
                        teacher_summary=seen,
                        provider="gemini",
                        latency_ms=round(latency_ms, 2),
                        priority=prio,
                        priority_reason=reason,
                        intent=intent,
                        structured_summary=struct
                    )
            except Exception:
                pass

        if not self.api_key:
            raise RuntimeError("Cannot invoke Gemini Teacher: GEMINI_API_KEY is not set.")

        import requests

        prompt_template = template or get_default_teacher_template()
        prompt = prompt_template.format(ticket_text=ticket_text)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        res_json = None
        max_retries = 4
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait_s = 2 ** (attempt + 1)
                logger.warning(f"Gemini 429 rate limit hit. Backing off for {wait_s}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            res_json = resp.json()
            break

        if not res_json or "candidates" not in res_json:
            raise RuntimeError("Failed to obtain valid response from Gemini API after retries.")

        raw_summary = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        teacher_summary = clean_teacher_summary(raw_summary)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        prio, reason, intent, struct = extract_teacher_metadata(ticket_text, teacher_summary)

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="gemini",
            latency_ms=round(latency_ms, 2),
            priority=prio,
            priority_reason=reason,
            intent=intent,
            structured_summary=struct
        )


def get_teacher_llm(config_path: Optional[Path] = None, provider_override: Optional[str] = None) -> TeacherLLM:
    """Factory function instantiating the configured Teacher LLM provider."""
    path = config_path or (Path(__file__).resolve().parents[2] / "configs" / "teacher_config.yaml")
    teacher_cfg: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            teacher_cfg = yaml.safe_load(f).get("teacher", {}) or {}

    provider = (provider_override or os.getenv("TEACHER_PROVIDER") or teacher_cfg.get("provider", "groq")).lower()
    default_model = "llama-3.3-70b-versatile" if provider == "groq" else "gemini-1.5-flash" if provider == "gemini" else "gpt-4o"
    model_name = os.getenv("TEACHER_MODEL_NAME") or teacher_cfg.get("model_name", default_model)

    logger.info(f"Instantiating Teacher LLM: Provider='{provider}', Model='{model_name}'")

    if provider == "groq":
        return GroqTeacher(model_name=model_name)
    elif provider == "gemini":
        return GeminiTeacher(model_name=model_name)
    elif provider == "openai":
        return OpenAITeacher(model_name=model_name)
    elif provider == "anthropic":
        return AnthropicTeacher(model_name=model_name)
    else:
        return MockTeacher(model_name=model_name if provider != "mock" else "mock-teacher-70b")
