import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from app.evaluation.metrics import EvaluationResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ExperimentTracker:
    """Utility for logging, saving, and comparing experiment runs."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        experiment_id: str,
        phase_name: str,
        eval_result: EvaluationResult,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        timestamp = int(time.time())
        file_path = self.results_dir / f"{experiment_id}_{timestamp}.json"

        payload = {
            "experiment_id": experiment_id,
            "phase": phase_name,
            "timestamp": timestamp,
            "metadata": extra_metadata or {},
            "metrics": eval_result.model_dump()
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Experiment '{experiment_id}' recorded successfully at: {file_path}")
        return file_path
