"""Evaluation framework for model quality, performance, latency, and inference cost."""
from app.evaluation.metrics import (
    QualityMetrics,
    PerformanceMetrics,
    CostMetrics,
    EvaluationResult,
    compute_quality_metrics,
    compute_performance_metrics,
    compute_cost_metrics,
)
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.tracker import ExperimentTracker
from app.evaluation.failure_analysis import (
    FailureAnalyzer,
    TicketFailureReport,
    AggregateFailureSummary,
)

__all__ = [
    "QualityMetrics",
    "PerformanceMetrics",
    "CostMetrics",
    "EvaluationResult",
    "compute_quality_metrics",
    "compute_performance_metrics",
    "compute_cost_metrics",
    "BaselineEvaluator",
    "ExperimentTracker",
    "FailureAnalyzer",
    "TicketFailureReport",
    "AggregateFailureSummary",
]
