import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class QualityMetrics(BaseModel):
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    key_info_coverage: Optional[float] = None

class PerformanceMetrics(BaseModel):
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    throughput_tokens_per_sec: float
    avg_input_tokens: float
    avg_output_tokens: float
    max_vram_mb: Optional[float] = None

class CostMetrics(BaseModel):
    estimated_cost_per_ticket_usd: float
    estimated_cost_per_1k_tickets_usd: float

class EvaluationResult(BaseModel):
    model_name: str
    sample_count: int
    quality: QualityMetrics
    performance: PerformanceMetrics
    cost: CostMetrics

def compute_quality_metrics(
    predictions: List[str],
    references: List[Optional[str]]
) -> QualityMetrics:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L, and key information coverage."""
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if r and r.strip()]
    
    if not valid_pairs:
        return QualityMetrics()

    preds = [p for p, r in valid_pairs]
    refs = [r for p, r in valid_pairs]

    r1_list, r2_list, rl_list, coverage_list = [], [], [], []

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)
    except ImportError:
        # Fallback simple token overlap metric if rouge-score is not installed
        for pred, ref in zip(preds, refs):
            p_words = set(pred.lower().split())
            r_words = set(ref.lower().split())
            overlap = len(p_words.intersection(r_words)) / max(len(r_words), 1)
            rl_list.append(overlap)
            r1_list.append(overlap)
            r2_list.append(overlap)

    # Simple Key Information Coverage (proportion of reference content words appearing in prediction)
    for pred, ref in zip(preds, refs):
        ref_words = [w.lower() for w in ref.split() if len(w) > 3]
        if ref_words:
            matched = sum(1 for w in ref_words if w in pred.lower())
            coverage_list.append(matched / len(ref_words))
        else:
            coverage_list.append(1.0)

    return QualityMetrics(
        rouge1=round(float(np.mean(r1_list)), 4),
        rouge2=round(float(np.mean(r2_list)), 4) if r2_list else None,
        rougeL=round(float(np.mean(rl_list)), 4),
        key_info_coverage=round(float(np.mean(coverage_list)), 4)
    )

def compute_performance_metrics(
    latencies_ms: List[float],
    input_token_counts: List[int],
    output_token_counts: List[int]
) -> PerformanceMetrics:
    """Computes latency percentiles (P50, P95, Mean), throughput, and average token counts."""
    if not latencies_ms:
        return PerformanceMetrics(
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            mean_latency_ms=0.0,
            throughput_tokens_per_sec=0.0,
            avg_input_tokens=0.0,
            avg_output_tokens=0.0
        )

    lat_arr = np.array(latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    mean_lat = float(np.mean(lat_arr))

    total_output_tokens = sum(output_token_counts)
    total_time_sec = sum(latencies_ms) / 1000.0
    throughput = (total_output_tokens / total_time_sec) if total_time_sec > 0 else 0.0

    avg_input = float(np.mean(input_token_counts)) if input_token_counts else 0.0
    avg_output = float(np.mean(output_token_counts)) if output_token_counts else 0.0

    max_vram_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            max_vram_bytes = torch.cuda.max_memory_allocated()
            max_vram_mb = round(max_vram_bytes / (1024 * 1024), 2)
    except Exception:
        pass

    return PerformanceMetrics(
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        mean_latency_ms=round(mean_lat, 2),
        throughput_tokens_per_sec=round(throughput, 2),
        avg_input_tokens=round(avg_input, 2),
        avg_output_tokens=round(avg_output, 2),
        max_vram_mb=max_vram_mb
    )

def compute_cost_metrics(
    mean_latency_ms: float,
    hourly_hardware_cost_usd: float = 0.85
) -> CostMetrics:
    """Computes estimated inference cost based on hardware hourly rate and mean latency."""
    # Cost per millisecond = hourly_rate / (3600 * 1000)
    cost_per_ms = hourly_hardware_cost_usd / 3600000.0
    cost_per_ticket = mean_latency_ms * cost_per_ms
    cost_per_1k = cost_per_ticket * 1000.0

    return CostMetrics(
        estimated_cost_per_ticket_usd=round(cost_per_ticket, 6),
        estimated_cost_per_1k_tickets_usd=round(cost_per_1k, 4)
    )
