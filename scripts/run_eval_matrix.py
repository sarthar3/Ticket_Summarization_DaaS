"""
DaaS Comprehensive Evaluation Matrix Runner
Computes Quality, Latency, Classification Accuracy, and Financial Cost Savings
across model checkpoints and test datasets.
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set up encoding and paths
sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.inference.pipeline import SummarizationPipeline
from app.evaluation.metrics import (
    compute_quality_metrics,
    compute_performance_metrics,
    compute_cost_metrics
)
from app.preprocessing.cleaner import TicketPreprocessor

def run_evaluation_matrix(dataset_path: Optional[Path] = None, max_samples: int = 50) -> Dict[str, Any]:
    settings = get_settings()
    pipeline = SummarizationPipeline(settings=settings)
    preprocessor = TicketPreprocessor()

    test_path = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    sample_path = PROJECT_ROOT / "data" / "sample" / "sample_tickets.jsonl"
    if not dataset_path or not dataset_path.exists():
        dataset_path = test_path if test_path.exists() else sample_path

    print("\n" + "=" * 78)
    print("      DISTILLATION-AS-A-SERVICE (DaaS) COMPREHENSIVE EVALUATION MATRIX")
    print("=" * 78)
    print(f"Target Model      : {settings.model.name} (Student Edge)")
    print(f"Evaluation Dataset: {dataset_path.name}")

    records: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= max_samples:
                break

    print(f"Evaluated Samples : {len(records)} tickets")
    print("-" * 78)

    predictions: List[str] = []
    references: List[Optional[str]] = []
    latencies_ms: List[float] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    priority_matches = 0
    origin_matches = 0

    for idx, rec in enumerate(records, start=1):
        t_id = rec.get("ticket_id", f"TICK-{idx:04d}")
        text = rec.get("ticket_text") or rec.get("customer_issue") or ""
        ref_summary = rec.get("summary") or rec.get("target_summary")
        ref_prio = rec.get("priority", "Medium")
        
        t0 = time.perf_counter()
        resp = pipeline.run({
            "ticket_id": t_id,
            "ticket_text": text,
            "sector": rec.get("sector") or "IT & Tech Support"
        })
        lat = (time.perf_counter() - t0) * 1000.0

        predictions.append(resp.summary)
        references.append(ref_summary)
        latencies_ms.append(lat)
        input_tokens.append(resp.input_tokens)
        output_tokens.append(resp.output_tokens)

        # Priority check
        if rec.get("priority") and resp.priority:
            if rec["priority"].lower() in resp.priority.lower() or resp.priority.lower() in rec["priority"].lower():
                priority_matches += 1

        # Origin check
        expected_origin = rec.get("issue_origin") or "COMPANY_SIDE"
        if getattr(resp, "issue_origin", None) == expected_origin:
            origin_matches += 1

    # Quality metrics
    qual = compute_quality_metrics(predictions, references)
    perf = compute_performance_metrics(latencies_ms, input_tokens, output_tokens)
    cost = compute_cost_metrics(perf.mean_latency_ms)

    prio_acc = (priority_matches / max(len(records), 1)) * 100.0
    orig_acc = (origin_matches / max(len(records), 1)) * 100.0

    # Theoretical Teacher API Cost for same workload (Groq LLaMA 3.3 70B: ~$0.69/M tokens avg)
    teacher_theoretical_cost = (sum(input_tokens) * 0.00000059) + (sum(output_tokens) * 0.00000079)
    student_actual_cost = 0.00  # Local inference
    cost_savings_pct = 100.0

    # Load historical baseline from previous run to calculate comparative deltas
    eval_dir = PROJECT_ROOT / "experiments" / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_dir / "eval_matrix_latest.json"
    
    prev_results = None
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                prev_results = json.load(f)
        except Exception:
            prev_results = None

    # Display Report
    print(f"\n1. SUMMARIZATION QUALITY METRICS (N-Gram & Coverage)")
    print(f"   ├─ ROUGE-1 F1 Score           : {qual.rouge1:.4f}  (Target: > 0.40)")
    print(f"   ├─ ROUGE-2 F1 Score           : {qual.rouge2:.4f}  (Target: > 0.20)")
    print(f"   ├─ ROUGE-L F1 Score           : {qual.rougeL:.4f}  (Target: > 0.35)")
    print(f"   └─ Key Info Coverage          : {qual.key_info_coverage * 100:.1f}%")

    print(f"\n2. OPERATIONAL LATENCY & THROUGHPUT (Target: < 2.0s on CPU)")
    print(f"   ├─ P50 Median Latency         : {perf.p50_latency_ms:.1f}ms  {'[PASS]' if perf.p50_latency_ms < 2000 else '[WARN]'}")
    print(f"   ├─ P95 Tail Latency           : {perf.p95_latency_ms:.1f}ms  {'[PASS]' if perf.p95_latency_ms < 2500 else '[WARN]'}")
    print(f"   ├─ Mean Latency               : {perf.mean_latency_ms:.1f}ms")
    print(f"   ├─ Generation Throughput      : {perf.throughput_tokens_per_sec:.1f} tokens/sec")
    print(f"   └─ Average Tokens             : {perf.avg_input_tokens:.1f} in / {perf.avg_output_tokens:.1f} out")

    print(f"\n3. TRIAGE & ROOT-CAUSE CLASSIFICATION ACCURACY")
    print(f"   ├─ Priority Match Accuracy    : {prio_acc:.1f}%")
    print(f"   └─ Issue Origin Accuracy      : {orig_acc:.1f}%")

    print(f"\n4. DaaS FINANCIAL & TOKEN EFFICIENCY")
    print(f"   ├─ Student Token API Cost     : $0.00000 USD (Zero API Cost)")
    print(f"   ├─ Teacher Tokens Consumed    : 0 tokens (Bypassed during Edge Serving)")
    print(f"   ├─ Avoided Teacher API Cost   : ${teacher_theoretical_cost:.5f} USD")
    print(f"   └─ Cost Reduction vs Cloud API: {cost_savings_pct:.1f}%")

    # 5. Checkpoint Comparison Delta Section (shows whether metrics increased or decreased)
    if prev_results and "quality" in prev_results:
        pq = prev_results.get("quality", {})
        pp = prev_results.get("performance", {})
        pc = prev_results.get("classification", {})
        prev_time = prev_results.get("timestamp", "Previous Checkpoint")

        print(f"\n5. HISTORICAL EVALUATION COMPARISON (Earlier Run: {prev_time} ──► Current Run)")
        print("   " + "-" * 74)

        def print_point_delta(label, prev_val, curr_val):
            delta = curr_val - prev_val
            pct = (delta / max(abs(prev_val), 1e-6)) * 100.0 if prev_val else 0.0
            status = "INCREASED ▲" if delta > 0.0001 else ("DECREASED ▼" if delta < -0.0001 else "STEADY =")
            print(f"   ├─ {label:<28} : {prev_val:.4f} ──► {curr_val:.4f} ({delta:+.4f} / {pct:+.1f}%) [{status}]")

        def print_pct_delta(label, prev_val, curr_val):
            delta = curr_val - prev_val
            status = "INCREASED ▲" if delta > 0.05 else ("DECREASED ▼" if delta < -0.05 else "STEADY =")
            print(f"   ├─ {label:<28} : {prev_val:.1f}% ──► {curr_val:.1f}% ({delta:+.1f}%) [{status}]")

        def print_lat_delta(label, prev_val, curr_val):
            delta = curr_val - prev_val
            status = "FASTER / IMPROVED ▼" if delta < -1.0 else ("SLOWER ▲" if delta > 1.0 else "STEADY =")
            print(f"   ├─ {label:<28} : {prev_val:.1f}ms ──► {curr_val:.1f}ms ({delta:+.1f}ms) [{status}]")

        print_point_delta("ROUGE-1 (Unigram Points)", pq.get("rouge1", 0.0), qual.rouge1)
        print_point_delta("ROUGE-L (Syntax Points)", pq.get("rougeL", 0.0), qual.rougeL)
        print_pct_delta("Key Information Coverage", pq.get("key_info_coverage", 0.0) * (100.0 if pq.get("key_info_coverage", 0.0) <= 1.0 else 1.0), qual.key_info_coverage * 100.0)
        print_pct_delta("Priority Classification", pc.get("priority_accuracy_pct", 0.0), prio_acc)
        print_pct_delta("Issue Origin Accuracy", pc.get("origin_accuracy_pct", 0.0), orig_acc)
        print_lat_delta("P50 Median Latency", pp.get("p50_latency_ms", 0.0), perf.p50_latency_ms)
        print("   " + "-" * 74)

    print("=" * 78 + "\n")

    results = {
        "model_name": settings.model.name,
        "evaluated_samples": len(records),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality": {
            "rouge1": qual.rouge1,
            "rouge2": qual.rouge2,
            "rougeL": qual.rougeL,
            "key_info_coverage": qual.key_info_coverage
        },
        "performance": {
            "p50_latency_ms": perf.p50_latency_ms,
            "p95_latency_ms": perf.p95_latency_ms,
            "mean_latency_ms": perf.mean_latency_ms,
            "throughput_tokens_per_sec": perf.throughput_tokens_per_sec,
            "avg_input_tokens": perf.avg_input_tokens,
            "avg_output_tokens": perf.avg_output_tokens
        },
        "classification": {
            "priority_accuracy_pct": prio_acc,
            "origin_accuracy_pct": orig_acc
        },
        "financial_efficiency": {
            "student_api_cost_usd": 0.0,
            "avoided_teacher_cost_usd": teacher_theoretical_cost,
            "cost_reduction_pct": 100.0
        }
    }

    # Save to experiments directory
    eval_dir = PROJECT_ROOT / "experiments" / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_dir / "eval_matrix_latest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved latest evaluation matrix results to: {out_file}\n")
    return results

if __name__ == "__main__":
    run_evaluation_matrix()
