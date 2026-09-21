import os
import time
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)

class TeacherResponse(BaseModel):
    ticket_id: str
    teacher_model: str
    teacher_summary: str
    provider: str
    latency_ms: float

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

class MockTeacher(TeacherLLM):
    """Offline / test Mock Teacher Provider for fast reproducible runs without API dependencies."""

    def __init__(self, model_name: str = "mock-teacher-70b"):
        super().__init__(provider_name="mock", model_name=model_name)

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        start_time = time.perf_counter()
        # High quality synthetic teacher response derived from key ticket sentences
        first_sentence = ticket_text.split(".")[0].strip() if ticket_text else "Customer ticket processed."
        teacher_summary = f"Teacher Gold Summary [{ticket_id}]: {first_sentence}. Action item logged for resolution."
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="mock",
            latency_ms=round(latency_ms, 2)
        )

class OpenAITeacher(TeacherLLM):
    """OpenAI API Teacher Provider (e.g. GPT-4o / GPT-4-turbo)."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(provider_name="openai", model_name=model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable not set. OpenAI calls will fail if invoked.")

    def generate_summary(self, ticket_id: str, ticket_text: str, template: Optional[str] = None) -> TeacherResponse:
        if not self.api_key:
            raise RuntimeError("Cannot invoke OpenAI Teacher: OPENAI_API_KEY is not set.")
        
        start_time = time.perf_counter()
        import requests

        prompt = template.format(ticket_text=ticket_text) if template else f"Summarize concisely:\n{ticket_text}"
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

        teacher_summary = res_json["choices"][0]["message"]["content"].strip()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="openai",
            latency_ms=round(latency_ms, 2)
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

        prompt = template.format(ticket_text=ticket_text) if template else f"Summarize concisely:\n{ticket_text}"
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

        teacher_summary = res_json["content"][0]["text"].strip()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TeacherResponse(
            ticket_id=ticket_id,
            teacher_model=self.model_name,
            teacher_summary=teacher_summary,
            provider="anthropic",
            latency_ms=round(latency_ms, 2)
        )

def get_teacher_llm(config_path: Optional[Path] = None, provider_override: Optional[str] = None) -> TeacherLLM:
    """Factory function instantiating the configured Teacher LLM provider."""
    path = config_path or (Path(__file__).resolve().parents[2] / "configs" / "teacher_config.yaml")
    teacher_cfg: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            teacher_cfg = yaml.safe_load(f).get("teacher", {}) or {}

    provider = (provider_override or os.getenv("TEACHER_PROVIDER") or teacher_cfg.get("provider", "mock")).lower()
    model_name = os.getenv("TEACHER_MODEL_NAME") or teacher_cfg.get("model_name", "gpt-4o")

    logger.info(f"Instantiating Teacher LLM: Provider='{provider}', Model='{model_name}'")

    if provider == "openai":
        return OpenAITeacher(model_name=model_name)
    elif provider == "anthropic":
        return AnthropicTeacher(model_name=model_name)
    else:
        return MockTeacher(model_name=model_name if provider != "mock" else "mock-teacher-70b")
