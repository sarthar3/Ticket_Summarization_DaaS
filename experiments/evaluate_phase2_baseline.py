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

logger = get_logger("evaluate_phase2")

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Fixed Test Set Baseline Evaluation")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--test_dataset", type=str, default="data/processed/test.jsonl", help="Path to fixed test dataset")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory to save evaluation results")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    test_dataset_path = PROJECT_ROOT / args.test_dataset
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 65)
    print("    TICKET SUMMARIZATION DaaS — PHASE 2 BASELINE EVALUATION")
    print("=" * 65)

    if not test_dataset_path.exists():
        print(f"Error: Test dataset file not found at '{test_dataset_path}'. Run scripts/prepare_dataset.py first!")
        sys.exit(1)

    # 1. Load Settings
    settings = get_settings(config_path=config_path, force_reload=True)
    print(f"Model Under Test   : {settings.model.name}")
    print(f"Precision          : {settings.model.precision}")
    print(f"Target Device      : {settings.model.device}")
    print(f"Fixed Test Set     : {test_dataset_path}")

    # 2. Run Pipeline & Baseline Evaluation
    print("\n--- RUNNING EVALUATION ON FIXED TEST SET ---")
    pipeline = SummarizationPipeline(settings=settings)
    evaluator = BaselineEvaluator(pipeline=pipeline, settings=settings)

    eval_result, raw_records = evaluator.evaluate_dataset(
        dataset_path=test_dataset_path,
        save_results_dir=output_dir
    )

    # 3. Save Experiment Tracking Log
    tracker = ExperimentTracker(results_dir=output_dir)
    tracker.log_experiment(
        experiment_id="phase2_student_baseline",
        phase_name="Phase 2 - Baseline Evaluation",
        eval_result=eval_result,
        extra_metadata={
            "dataset_split": "test.jsonl",
            "seed": settings.reproducibility.seed,
            "max_input_tokens": settings.context.max_input_tokens,
            "max_output_tokens": settings.context.max_output_tokens
        }
    )

    # 4. Print Summary Report
    print("\n" + "=" * 65)
    print("          PHASE 2 BASELINE EVALUATION REPORT")
    print("=" * 65)
    print(f"Model Name           : {eval_result.model_name}")
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
    print(f"  Avg Input Tokens   : {eval_result.performance.avg_input_tokens}")
    print(f"  Avg Output Tokens  : {eval_result.performance.avg_output_tokens}")
    if eval_result.performance.max_vram_mb is not None:
        print(f"  Max VRAM Memory    : {eval_result.performance.max_vram_mb} MB")

    print("\n[ESTIMATED HARDWARE COST]")
    print(f"  Cost / Ticket      : ${eval_result.cost.estimated_cost_per_ticket_usd:.6f}")
    print(f"  Cost / 1k Tickets  : ${eval_result.cost.estimated_cost_per_1k_tickets_usd:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
