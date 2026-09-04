import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import Settings, get_settings
from app.inference.pipeline import SummarizationPipeline
from app.evaluation.evaluator import BaselineEvaluator
from app.preprocessing.token_stats import calculate_token_statistics
from app.utils.logger import get_logger

logger = get_logger("run_baseline")

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Student Model Evaluation Experiment")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="data/sample/sample_tickets.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory to save results")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    dataset_path = PROJECT_ROOT / args.dataset
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("      TICKET SUMMARIZATION DaaS — BASELINE EXPERIMENT")
    print("=" * 60)

    # 1. Load Settings
    settings = get_settings(config_path=config_path, force_reload=True)
    print(f"Model Selected : {settings.model.name}")
    print(f"Precision      : {settings.model.precision}")
    print(f"Device         : {settings.model.device}")
    print(f"Random Seed    : {settings.reproducibility.seed}")

    # 2. Token Statistics Analysis
    print("\n--- DATASET TOKEN LENGTH STATISTICS ---")
    if dataset_path.exists():
        import json
        texts = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    texts.append(item.get("ticket_text", ""))
        
        token_stats = calculate_token_statistics(
            texts=texts,
            max_context_limit=settings.context.max_input_tokens
        )
        print(f"Total Samples           : {token_stats.total_samples}")
        print(f"Min Token Length        : {token_stats.min_length}")
        print(f"Max Token Length        : {token_stats.max_length}")
        print(f"Mean Token Length       : {token_stats.mean_length:.2f}")
        print(f"P50 (Median) Length     : {token_stats.p50_length:.2f}")
        print(f"P95 Token Length        : {token_stats.p95_length:.2f}")
        print(f"Max Context Limit       : {token_stats.max_context_limit}")
        print(f"Exceeding Context Limit : {token_stats.exceeding_limit_count} ({token_stats.exceeding_limit_percentage}%)")
    else:
        print(f"Dataset not found at {dataset_path}")

    # 3. Initialize Pipeline & Evaluator
    print("\n--- INITIALIZING INFERENCE PIPELINE ---")
    pipeline = SummarizationPipeline(settings=settings)
    evaluator = BaselineEvaluator(pipeline=pipeline, settings=settings)

    # 4. Run Evaluation
    print(f"\nEvaluating dataset at {dataset_path}...")
    eval_result, _ = evaluator.evaluate_dataset(
        dataset_path=dataset_path,
        save_results_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("            BASELINE EXPERIMENT REPORT")
    print("=" * 60)
    print(f"Model Name           : {eval_result.model_name}")
    print(f"Evaluated Samples    : {eval_result.sample_count}")
    print("\n[QUALITY METRICS]")
    print(f"  ROUGE-1            : {eval_result.quality.rouge1}")
    print(f"  ROUGE-2            : {eval_result.quality.rouge2}")
    print(f"  ROUGE-L            : {eval_result.quality.rougeL}")
    print(f"  Key Info Coverage  : {eval_result.quality.key_info_coverage}")

    print("\n[PERFORMANCE METRICS]")
    print(f"  P50 Latency        : {eval_result.performance.p50_latency_ms} ms")
    print(f"  P95 Latency        : {eval_result.performance.p95_latency_ms} ms")
    print(f"  Mean Latency       : {eval_result.performance.mean_latency_ms} ms")
    print(f"  Throughput         : {eval_result.performance.throughput_tokens_per_sec} tokens/sec")
    print(f"  Avg Input Tokens   : {eval_result.performance.avg_input_tokens}")
    print(f"  Avg Output Tokens  : {eval_result.performance.avg_output_tokens}")
    if eval_result.performance.max_vram_mb is not None:
        print(f"  Max VRAM Usage     : {eval_result.performance.max_vram_mb} MB")

    print("\n[ESTIMATED INFERENCE COST]")
    print(f"  Cost / Ticket      : ${eval_result.cost.estimated_cost_per_ticket_usd:.6f}")
    print(f"  Cost / 1k Tickets  : ${eval_result.cost.estimated_cost_per_1k_tickets_usd:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
