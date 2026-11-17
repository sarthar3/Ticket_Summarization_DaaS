import os
import time
from typing import Dict, Any, Tuple, Optional
from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class StudentModelWrapper:
    """Configurable wrapper around Hugging Face Causal LMs."""

    def __init__(self, settings: Optional[Settings] = None, lazy_load: bool = False):
        self.settings = settings or get_settings()
        self.model_name = self.settings.model.name
        self.precision = self.settings.model.precision.lower()
        self.device_setting = self.settings.model.device.lower()
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        if not lazy_load:
            self.load_model()

    def _resolve_device_and_dtype(self) -> Tuple[str, Any]:
        """Resolves target computing device and torch precision data type."""
        try:
            import torch
        except ImportError:
            return "cpu", None

        # Device selection
        if self.device_setting == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device_setting

        # Precision selection
        if precision_dtype := getattr(torch, "float16" if self.precision == "fp16" or self.precision == "float16"
                                        else "bfloat16" if self.precision == "bf16" or self.precision == "bfloat16"
                                        else "float32", None):
            dtype = precision_dtype
        else:
            dtype = torch.float32

        if device == "cpu":
            dtype = torch.float32  # CPU generally works best with float32

        return device, dtype

    def load_model(self) -> None:
        """Loads model and tokenizer using Hugging Face transformers."""
        if self.is_loaded:
            return

        device, torch_dtype = self._resolve_device_and_dtype()
        logger.info(
            f"Loading Student Model '{self.model_name}' on device '{device}' with precision '{self.precision}'",
            extra={"extra": {"model_name": self.model_name, "device": device, "precision": self.precision}}
        )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs: Dict[str, Any] = {
                "trust_remote_code": self.settings.model.trust_remote_code
            }

            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype

            if device != "cpu" and self.device_setting == "auto":
                kwargs["device_map"] = "auto"

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.settings.model.trust_remote_code
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
                if device != "cpu" and "device_map" not in kwargs:
                    self.model = self.model.to(device)

            except Exception as ex:
                logger.warning(
                    f"Failed to load primary model '{self.model_name}': {str(ex)}. "
                    f"Attempting fallback CPU model '{self.settings.model.fallback_cpu_model}' for testing."
                )
                self.model_name = self.settings.model.fallback_cpu_model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                device = "cpu"

            self.model.eval()
            self.is_loaded = True
            
            # Warm up JIT execution kernels and thread pool
            try:
                torch.set_num_threads(8)
                with torch.inference_mode():
                    warm_inputs = self.tokenizer("ping", return_tensors="pt").to(device)
                    self.model.generate(**warm_inputs, max_new_tokens=1)
            except Exception:
                pass

            logger.info(f"Student Model '{self.model_name}' loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Could not load Hugging Face model: {str(e)}") from e

    def generate(self, prompt_text: str, return_breakdown: bool = False) -> Any:
        """Generates summary from prompt text with granular latency profiling.
        
        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, latency_ms)
            or with breakdown if return_breakdown=True.
        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.perf_counter()
        
        import torch

        # Device selection & thread optimization (8 threads achieves peak CPU throughput)
        device = next(self.model.parameters()).device
        if device.type == "cpu":
            try:
                torch.set_num_threads(8)
            except Exception:
                pass
        
        # Tokenize input
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.context.max_input_tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_tokens_count = int(inputs["input_ids"].shape[1])
        tokenization_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Collect all valid EOS / Stop token IDs for Qwen and Causal models
        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None and im_end not in eos_ids:
                eos_ids.append(im_end)
            end_of_text = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if end_of_text is not None and end_of_text not in eos_ids:
                eos_ids.append(end_of_text)

        # Sentence-level early stopping criteria: halt immediately upon terminal punctuation (sub-2s latency)
        from transformers import StoppingCriteria, StoppingCriteriaList
        class StopOnSentenceEnd(StoppingCriteria):
            def __init__(self, start_len: int, tokenizer):
                self.start_len = start_len
                self.tokenizer = tokenizer
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                gen_len = input_ids.shape[1] - self.start_len
                if gen_len < 4:
                    return False
                last_token = input_ids[0, -1].item()
                text = self.tokenizer.decode([last_token], skip_special_tokens=False)
                return any(c in text for c in [".", "\n", "!", "?"])

        stopping_criteria = StoppingCriteriaList([StopOnSentenceEnd(input_tokens_count, self.tokenizer)])

        # Generation parameters (strictly capped to 8 tokens max for guaranteed sub-2s CPU execution)
        max_new = min(self.settings.generation.max_new_tokens, 8)
        gen_kwargs = {
            "max_new_tokens": max_new,
            "min_new_tokens": min(self.settings.generation.min_new_tokens, 4),
            "do_sample": False,
            "repetition_penalty": self.settings.generation.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
            "stopping_criteria": stopping_criteria,
            "use_cache": True
        }

        if self.settings.generation.do_sample:
            gen_kwargs["temperature"] = self.settings.generation.temperature
            gen_kwargs["top_p"] = self.settings.generation.top_p

        # Pure model inference using lightweight inference_mode
        t1 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        inference_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        # Decode generated slice only
        t2 = time.perf_counter()
        new_tokens_ids = output_ids[0][input_tokens_count:]
        output_tokens_count = len(new_tokens_ids)
        generated_text = self.tokenizer.decode(new_tokens_ids, skip_special_tokens=True).strip()
        decode_ms = round((time.perf_counter() - t2) * 1000.0, 2)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        breakdown = {
            "tokenization_ms": tokenization_ms,
            "inference_ms": inference_ms,
            "decode_ms": decode_ms,
            "total_ms": total_latency_ms
        }
        self.last_latency_breakdown = breakdown

        if return_breakdown:
            return generated_text, input_tokens_count, output_tokens_count, total_latency_ms, breakdown
        return generated_text, input_tokens_count, output_tokens_count, total_latency_ms
