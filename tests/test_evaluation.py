from app.evaluation.metrics import (
    compute_quality_metrics,
    compute_performance_metrics,
    compute_cost_metrics,
)

def test_compute_quality_metrics():
    preds = ["User unable to login to mobile banking app."]
    refs = ["Customer cannot log in to mobile banking application."]
    quality = compute_quality_metrics(preds, refs)
    assert quality.rougeL is not None
    assert quality.rougeL > 0.0
    assert quality.key_info_coverage is not None
    assert quality.key_info_coverage > 0.0

def test_compute_performance_metrics():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    input_tokens = [50, 60, 70, 80, 90]
    output_tokens = [10, 20, 30, 40, 50]
    
    perf = compute_performance_metrics(latencies, input_tokens, output_tokens)
    assert perf.p50_latency_ms == 300.0
    assert perf.p95_latency_ms == 480.0
    assert perf.mean_latency_ms == 300.0
    assert perf.avg_input_tokens == 70.0
    assert perf.avg_output_tokens == 30.0
    assert perf.throughput_tokens_per_sec == 100.0  # 150 tokens / 1.5 sec

def test_compute_cost_metrics():
    mean_latency_ms = 1000.0  # 1 second
    hourly_rate = 3.60        # $3.60 / hour => $0.001 / second
    cost = compute_cost_metrics(mean_latency_ms, hourly_rate)
    assert round(cost.estimated_cost_per_ticket_usd, 3) == 0.001
    assert round(cost.estimated_cost_per_1k_tickets_usd, 2) == 1.00
