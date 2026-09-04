import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TokenStatsResult(BaseModel):
    total_samples: int
    min_length: int
    max_length: int
    mean_length: float
    p50_length: float
    p95_length: float
    max_context_limit: int
    exceeding_limit_count: int
    exceeding_limit_percentage: float

def calculate_token_statistics(
    texts: List[str],
    tokenizer: Optional[Any] = None,
    max_context_limit: int = 1024
) -> TokenStatsResult:
    """Computes comprehensive token length statistics over a dataset.
    
    If tokenizer is provided, uses tokenizer token counts; otherwise falls back to word count.
    """
    if not texts:
        return TokenStatsResult(
            total_samples=0,
            min_length=0,
            max_length=0,
            mean_length=0.0,
            p50_length=0.0,
            p95_length=0.0,
            max_context_limit=max_context_limit,
            exceeding_limit_count=0,
            exceeding_limit_percentage=0.0
        )

    token_counts: List[int] = []
    for text in texts:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts.append(len(tokens))
        else:
            # Simple word count approximation if no tokenizer available
            token_counts.append(len(text.split()))

    counts_arr = np.array(token_counts)
    total_samples = len(token_counts)
    exceeding = int(np.sum(counts_arr > max_context_limit))
    exceeding_pct = float((exceeding / total_samples) * 100.0) if total_samples > 0 else 0.0

    return TokenStatsResult(
        total_samples=total_samples,
        min_length=int(np.min(counts_arr)),
        max_length=int(np.max(counts_arr)),
        mean_length=float(np.mean(counts_arr)),
        p50_length=float(np.percentile(counts_arr, 50)),
        p95_length=float(np.percentile(counts_arr, 95)),
        max_context_limit=max_context_limit,
        exceeding_limit_count=exceeding,
        exceeding_limit_percentage=round(exceeding_pct, 2)
    )
