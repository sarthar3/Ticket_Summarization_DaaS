import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from app.preprocessing.cleaner import TicketPreprocessor, TicketData
from app.preprocessing.token_stats import calculate_token_statistics, TokenStatsResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DatasetSplitter:
    """Handles dataset loading, validation, deterministic splitting, and persistence."""

    def __init__(self, seed: int = 42, preprocessor: Optional[TicketPreprocessor] = None):
        self.seed = seed
        self.preprocessor = preprocessor or TicketPreprocessor()

    def split_dataset(
        self,
        records: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validates and deterministically splits dataset into train, validation, and test sets."""
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0. Got {train_ratio} + {val_ratio} + {test_ratio}")

        validated_records: List[Dict[str, Any]] = []
        for rec in records:
            try:
                ticket_data = self.preprocessor.validate_and_normalize(rec)
                validated_records.append(ticket_data.model_dump())
            except Exception as e:
                logger.warning(f"Skipping invalid record '{rec.get('ticket_id')}': {str(e)}")

        if not validated_records:
            raise ValueError("No valid records found in dataset to split.")

        # Shuffle deterministically
        rng = random.Random(self.seed)
        shuffled = list(validated_records)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_split = shuffled[:n_train]
        val_split = shuffled[n_train:n_train + n_val]
        test_split = shuffled[n_train + n_val:]

        logger.info(
            f"Dataset split complete: Total={n_total}, Train={len(train_split)}, Val={len(val_split)}, Test={len(test_split)}"
        )
        return train_split, val_split, test_split

    def save_split(self, split_data: List[Dict[str, Any]], output_file: Path) -> None:
        """Saves split records to a JSONL file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for record in split_data:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved {len(split_data)} records to {output_file}")
