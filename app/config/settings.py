import os
import yaml
import random
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default_config.yaml"

class ModelSettings(BaseModel):
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    precision: str = "float16"
    device: str = "auto"
    trust_remote_code: bool = True
    fallback_cpu_model: str = "sshleifer/tiny-gpt2"

class ContextSettings(BaseModel):
    max_input_tokens: int = 1024
    max_output_tokens: int = 256
    truncation_side: str = "right"

class GenerationSettings(BaseModel):
    max_new_tokens: int = 256
    min_new_tokens: int = 10
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 1
    repetition_penalty: float = 1.1

class PromptSettings(BaseModel):
    template: str = Field(
        default="Summarize the following customer support ticket concisely:\n\nTicket:\n{ticket_text}\n\nSummary:"
    )

class EvaluationSettings(BaseModel):
    rouge_types: list = Field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    hourly_hardware_cost_usd: float = 0.85

class ReproducibilitySettings(BaseModel):
    seed: int = 42

class Settings(BaseModel):
    model: ModelSettings = Field(default_factory=ModelSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    prompt: PromptSettings = Field(default_factory=PromptSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    reproducibility: ReproducibilitySettings = Field(default_factory=ReproducibilitySettings)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        path = config_path or DEFAULT_CONFIG_PATH
        config_dict: Dict[str, Any] = {}
        
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        # Allow Environment Variable overrides
        if os.getenv("MODEL_NAME"):
            config_dict.setdefault("model", {})["name"] = os.getenv("MODEL_NAME")
        if os.getenv("PRECISION"):
            config_dict.setdefault("model", {})["precision"] = os.getenv("PRECISION")
        if os.getenv("DEVICE"):
            config_dict.setdefault("model", {})["device"] = os.getenv("DEVICE")
        if os.getenv("MAX_INPUT_TOKENS"):
            config_dict.setdefault("context", {})["max_input_tokens"] = int(os.getenv("MAX_INPUT_TOKENS"))
        if os.getenv("MAX_OUTPUT_TOKENS"):
            config_dict.setdefault("context", {})["max_output_tokens"] = int(os.getenv("MAX_OUTPUT_TOKENS"))
        if os.getenv("RANDOM_SEED"):
            config_dict.setdefault("reproducibility", {})["seed"] = int(os.getenv("RANDOM_SEED"))

        return cls(**config_dict)

    def set_seed(self) -> None:
        """Enforce random seed across random, numpy, and torch if available."""
        seed = self.reproducibility.seed
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

_settings_instance: Optional[Settings] = None

def get_settings(config_path: Optional[Path] = None, force_reload: bool = False) -> Settings:
    global _settings_instance
    if _settings_instance is None or force_reload:
        _settings_instance = Settings.load(config_path)
        _settings_instance.set_seed()
    return _settings_instance
