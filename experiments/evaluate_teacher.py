import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.teacher.teacher_llm import get_teacher_llm, TeacherLLM
from app.evaluation.metrics import (
    compute_quality_metrics,
    compute_performance_metrics,
    compute_cost_metrics,
    EvaluationResult
)
from app.evaluation.tracker import ExperimentTracker
from app.utils.logger import get_logger

logger = get_logger("evaluate_teacher")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Teacher LLM Upper-Bound Benchmark Quality")
    parser.add_argument("--provider", type=str, default=None, help="Teacher provider override")
    parser.add_argument("--test_dataset", type=str, default="data/processed/test.jsonl", help="Path to test set")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory to save evaluation results")
    args = parser.parse_args()

    test_dataset_path = PROJECT_ROOT / args.test_dataset
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 65)
    print("    TICKET SUMMARIZATION DaaS — TEACHER LLM BENCHMARK EVALUATION")
    print("=" * 65)

    teacher = get_teacher_llm(provider_override=args.provider)
    print(f"Teacher Provider   : {teacher.provider_name}")
    print(f"Teacher Model      : {teacher.model_name}")
    print(f"Test Dataset Path  : {test_dataset_path}")

    if not test_dataset_path.exists():
        print(f"Error: Test dataset file not found at '{test_dataset_path}'")
        sys.exit(1)

    records = []
    with open(test_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    predictions = []
    references = []
    latencies = []
    input_token_counts = []
    output_token_counts = []

    for rec in records:
        t_id = rec.get("ticket_id", "UNKNOWN")
        t_text = rec.get("ticket_text", "")
        ref = rec.get("summary")

        t_resp = teacher.generate_summary(t_id, t_text)
        predictions.append(t_resp.teacher_summary)
        references.append(ref)
        latencies.append(t_resp.latency_ms)
        input_token_counts.append(len(t_text.split()))
        output_token_counts.append(len(t_resp.teacher_summary.split()))

    quality = compute_quality_metrics(predictions, references)
    performance = compute_performance_metrics(latencies, input_token_counts, output_token_counts)
    cost = compute_cost_metrics(performance.mean_latency_ms, hourly_hardware_cost_usd=2.50)

    eval_result = EvaluationResult(
        model_name=f"Teacher-{teacher.model_name}",
        sample_count=len(records),
        quality=quality,
        performance=performance,
        cost=cost
    )

    tracker = ExperimentTracker(results_dir=output_dir)
    tracker.log_experiment(
        experiment_id="phase5_teacher_upper_bound",
        phase_name="Phase 5 - Teacher LLM Integration",
        eval_result=eval_result,
        extra_metadata={
            "provider": teacher.provider_name,
            "teacher_model": teacher.model_name
        }
    )

    print("\n" + "=" * 65)
    print("          TEACHER LLM UPPER-BOUND REPORT")
    print("=" * 65)
    print(f"Teacher Model Name   : {eval_result.model_name}")
    print(f"Test Set Size        : {eval_result.sample_count} records")

    print("\n[QUALITY UPPER-BOUND BENCHMARK]")
    print(f"  ROUGE-1 Score      : {eval_result.quality.rouge1}")
    print(f"  ROUGE-2 Score      : {eval_result.quality.rouge2}")
    print(f"  ROUGE-L Score      : {eval_result.quality.rougeL}")
    print(f"  Key Info Coverage  : {eval_result.quality.key_info_coverage}")

    print("\n[PERFORMANCE SUMMARY]")
    print(f"  Mean Latency       : {eval_result.performance.mean_latency_ms} ms")
    print(f"  Throughput         : {eval_result.performance.throughput_tokens_per_sec} tokens/sec")
    print("=" * 65)

if __name__ == "__main__":
    main()
