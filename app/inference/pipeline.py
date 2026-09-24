import re
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.config.settings import Settings, get_settings
from app.preprocessing.cleaner import TicketPreprocessor, TicketData
from app.inference.student_model import StudentModelWrapper
from app.utils.logger import get_logger

logger = get_logger(__name__)

class StructuredSummaryDetails(BaseModel):
    core_issue: str
    customer_intent: str
    key_action_items: str

class PipelineResponse(BaseModel):
    ticket_id: str
    summary: str
    structured_summary: StructuredSummaryDetails
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

    def build_structured_summary(self, ticket_data: TicketData, cleaned_summary: str) -> StructuredSummaryDetails:
        """Extracts and structures core issue, customer intent, and action items."""
        # Core issue derived from summary or first sentence of ticket
        sentences = [s.strip() for s in cleaned_summary.split(".") if s.strip()]
        core_issue = sentences[0] if sentences else ticket_data.ticket_text[:100]

        # Customer intent derived from ticket metadata / sector / priority
        intent_label = ticket_data.intent or "Technical Support"
        priority_label = ticket_data.priority or "Medium"
        sector_label = ticket_data.sector or "General Support"
        customer_intent = f"{sector_label} / {intent_label} ({priority_label} Priority)"

        # Key action items derived from second sentence or request context
        if len(sentences) > 1:
            action_items = ". ".join(sentences[1:])
        else:
            action_items = "Investigate ticket details and follow up with customer."

        return StructuredSummaryDetails(
            core_issue=core_issue,
            customer_intent=customer_intent,
            key_action_items=action_items
        )

    def run(self, raw_ticket: Dict[str, Any]) -> PipelineResponse:
        """Runs the complete summarization pipeline:
        1. Validate request
        2. Normalize / clean ticket text
        3. Format prompt
        4. Tokenize & Model inference
        5. Decode generated output
        6. Post-process & Structure summary output
        7. Return structured PipelineResponse
        """
        # Step 1 & 2: Validate & Normalize
        ticket_data: TicketData = self.preprocessor.validate_and_normalize(raw_ticket)

        # Step 3: Prompt Formatting
        prompt_text = self.format_prompt(ticket_data.ticket_text)

        # Step 4 & 5: Model Inference & Decoding
        raw_output, input_tokens, output_tokens, latency_ms = self.model_wrapper.generate(prompt_text)

        # Step 6: Post-processing & Structuring
        cleaned_summary = self.post_process_summary(raw_output)
        structured_details = self.build_structured_summary(ticket_data, cleaned_summary)

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
            structured_summary=structured_details,
            model=self.model_wrapper.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
