import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.config.settings import Settings, get_settings
from app.inference.pipeline import SummarizationPipeline, PipelineResponse
from app.evaluation.metrics import (
    QualityMetrics,
    PerformanceMetrics,
    CostMetrics,
    EvaluationResult,
    compute_quality_metrics,
    compute_performance_metrics,
    compute_cost_metrics,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

class BaselineEvaluator:
    """Runs baseline evaluations on dataset splits and compiles aggregate report."""

    def __init__(
        self,
        pipeline: Optional[SummarizationPipeline] = None,
        settings: Optional[Settings] = None
    ):
        self.settings = settings or get_settings()
        self.pipeline = pipeline or SummarizationPipeline(settings=self.settings)

    def evaluate_dataset(
        self,
        dataset_path: Path,
        save_results_dir: Optional[Path] = None
    ) -> Tuple[EvaluationResult, List[Dict[str, Any]]]:
        """Evaluates model against a jsonl dataset file."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found: {dataset_path}")

        records: List[Dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        logger.info(f"Loaded {len(records)} test records from {dataset_path} for evaluation.")

        predictions: List[str] = []
        references: List[Optional[str]] = []
        latencies_ms: List[float] = []
        input_tokens: List[int] = []
        output_tokens: List[int] = []
        raw_results: List[Dict[str, Any]] = []

        for record in records:
            response: PipelineResponse = self.pipeline.run(record)
            
            ref_summary = record.get("summary")
            predictions.append(response.summary)
            references.append(ref_summary)
            latencies_ms.append(response.latency_ms)
            input_tokens.append(response.input_tokens)
            output_tokens.append(response.output_tokens)

            raw_results.append({
                "ticket_id": response.ticket_id,
                "model": response.model,
                "generated_summary": response.summary,
                "reference_summary": ref_summary,
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "sector": record.get("sector"),
                "priority": record.get("priority")
            })

        quality: QualityMetrics = compute_quality_metrics(predictions, references)
        performance: PerformanceMetrics = compute_performance_metrics(
            latencies_ms=latencies_ms,
            input_token_counts=input_tokens,
            output_token_counts=output_tokens
        )
        cost: CostMetrics = compute_cost_metrics(
            mean_latency_ms=performance.mean_latency_ms,
            hourly_hardware_cost_usd=self.settings.evaluation.hourly_hardware_cost_usd
        )

        eval_result = EvaluationResult(
            model_name=self.pipeline.model_wrapper.model_name,
            sample_count=len(records),
            quality=quality,
            performance=performance,
            cost=cost
        )

        if save_results_dir:
            save_results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            output_file = save_results_dir / f"baseline_results_{timestamp}.json"
            payload = {
                "aggregate_metrics": eval_result.model_dump(),
                "raw_results": raw_results
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Saved evaluation raw results and aggregate report to: {output_file}")

        return eval_result, raw_results
