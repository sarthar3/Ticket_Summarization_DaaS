from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.config.settings import Settings, get_settings
from app.preprocessing.cleaner import TicketPreprocessor, TicketData
from app.inference.student_model import StudentModelWrapper
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PipelineResponse(BaseModel):
    ticket_id: str
    summary: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

class SummarizationPipeline:
    """End-to-end ticket summarization inference pipeline."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_wrapper: Optional[StudentModelWrapper] = None
    ):
        self.settings = settings or get_settings()
        self.preprocessor = TicketPreprocessor(
            max_input_tokens=self.settings.context.max_input_tokens,
            max_output_tokens=self.settings.context.max_output_tokens,
            truncation_side=self.settings.context.truncation_side
        )
        self.model_wrapper = model_wrapper or StudentModelWrapper(settings=self.settings)

    def format_prompt(self, ticket_text: str) -> str:
        template = self.settings.prompt.template
        return template.format(ticket_text=ticket_text)

    def post_process_summary(self, generated_text: str) -> str:
        """Cleans and post-processes raw LLM text generation output."""
        summary = generated_text.strip()
        # Remove repetitive prefixes if model outputs them
        prefixes_to_remove = ["Summary:", "Summary :-", "Here is the summary:"]
        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()
        return summary

    def run(self, raw_ticket: Dict[str, Any]) -> PipelineResponse:
        """Runs the complete summarization pipeline:
        1. Validate request
        2. Normalize / clean ticket text
        3. Format prompt
        4. Tokenize & Model inference
        5. Decode generated output
        6. Post-process summary
        7. Return structured PipelineResponse
        """
        # Step 1 & 2: Validate & Normalize
        ticket_data: TicketData = self.preprocessor.validate_and_normalize(raw_ticket)

        # Step 3: Prompt Formatting
        prompt_text = self.format_prompt(ticket_data.ticket_text)

        # Step 4 & 5: Model Inference & Decoding
        raw_output, input_tokens, output_tokens, latency_ms = self.model_wrapper.generate(prompt_text)

        # Step 6: Post-processing
        cleaned_summary = self.post_process_summary(raw_output)

        logger.info(
            f"Successfully processed ticket '{ticket_data.ticket_id}'",
            extra={
                "extra": {
                    "ticket_id": ticket_data.ticket_id,
                    "model": self.model_wrapper.model_name,
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            }
        )

        # Step 7: Return structured response
        return PipelineResponse(
            ticket_id=ticket_data.ticket_id,
            summary=cleaned_summary,
            model=self.model_wrapper.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
