import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

class QuantizationConfig:
    """Manages quantization configs (FP16, BF16, INT8, INT4) for model inference optimization."""

    def __init__(self, precision: str = "float16"):
        self.precision = precision.lower()

    def get_quantization_kwargs(self) -> Dict[str, Any]:
        """Builds Hugging Face kwargs for model loading under selected quantization precision."""
        kwargs: Dict[str, Any] = {}

        if self.precision in ["int8", "8bit"]:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Configured INT8 8-bit quantization via BitsAndBytesConfig.")
            except ImportError:
                logger.warning("bitsandbytes package not available. Falling back to default float16 precision.")

        elif self.precision in ["int4", "4bit", "nf4"]:
            try:
                import torch
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Configured INT4 4-bit NormalFloat4 (nf4) quantization via BitsAndBytesConfig.")
            except ImportError:
                logger.warning("bitsandbytes package not available. Falling back to default float16 precision.")

        return kwargs

def load_deployment_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or (Path(__file__).resolve().parents[2] / "configs" / "deployment.yaml")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
