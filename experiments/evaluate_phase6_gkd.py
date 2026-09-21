import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import Settings, get_settings
from app.inference.pipeline import SummarizationPipeline
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.tracker import ExperimentTracker
from app.utils.logger import get_logger

logger = get_logger("evaluate_phase6")

def main():
    parser = argparse.ArgumentParser(description="Phase 6 GKD Distilled Model Evaluation")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--test_dataset", type=str, default="data/processed/test.jsonl", help="Path to fixed test set")
    parser.add_argument("--distill_dir", type=str, default="checkpoints/gkd_student", help="Path to GKD checkpoints")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory to save evaluation results")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    test_dataset_path = PROJECT_ROOT / args.test_dataset
    distill_dir = PROJECT_ROOT / args.distill_dir
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 65)
    print("    TICKET SUMMARIZATION DaaS — PHASE 6 GKD DISTILLED MODEL EVALUATION")
    print("=" * 65)

    settings = get_settings(config_path=config_path, force_reload=True)
    print(f"Student Model Target : {settings.model.name}")
    print(f"GKD Adapter Dir      : {distill_dir}")
    print(f"Fixed Test Set       : {test_dataset_path}")

    pipeline = SummarizationPipeline(settings=settings)
    evaluator = BaselineEvaluator(pipeline=pipeline, settings=settings)

    eval_result, _ = evaluator.evaluate_dataset(
        dataset_path=test_dataset_path,
        save_results_dir=output_dir
    )

    tracker = ExperimentTracker(results_dir=output_dir)
    tracker.log_experiment(
        experiment_id="phase6_gkd_distilled_student",
        phase_name="Phase 6 - Generalized Knowledge Distillation (GKD)",
        eval_result=eval_result,
        extra_metadata={
            "distill_dir": str(distill_dir),
            "dataset_split": "test.jsonl"
        }
    )

    print("\n" + "=" * 65)
    print("            PHASE 6 GKD EVALUATION REPORT")
    print("=" * 65)
    print(f"Model Name           : {eval_result.model_name} (+ GKD Distillation)")
    print(f"Test Set Size        : {eval_result.sample_count} records")

    print("\n[QUALITY BENCHMARK]")
    print(f"  ROUGE-1 Score      : {eval_result.quality.rouge1}")
    print(f"  ROUGE-2 Score      : {eval_result.quality.rouge2}")
    print(f"  ROUGE-L Score      : {eval_result.quality.rougeL}")
    print(f"  Key Info Coverage  : {eval_result.quality.key_info_coverage}")

    print("\n[PERFORMANCE BENCHMARK]")
    print(f"  P50 Latency        : {eval_result.performance.p50_latency_ms} ms")
    print(f"  P95 Latency        : {eval_result.performance.p95_latency_ms} ms")
    print(f"  Mean Latency       : {eval_result.performance.mean_latency_ms} ms")
    print(f"  Throughput         : {eval_result.performance.throughput_tokens_per_sec} tokens/sec")

    print("\n[ESTIMATED HARDWARE COST]")
    print(f"  Cost / Ticket      : ${eval_result.cost.estimated_cost_per_ticket_usd:.6f}")
    print(f"  Cost / 1k Tickets  : ${eval_result.cost.estimated_cost_per_1k_tickets_usd:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
