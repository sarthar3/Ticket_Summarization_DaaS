"""
Preprocesses and splits the 20,000 IT & Cloud Support dataset into
standardized train.jsonl, val.jsonl, and test.jsonl splits.
"""
import sys
import re
import json
from pathlib import Path
import pandas as pd

# Set up project root and stdout encoding
sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.dataset import DatasetSplitter
from app.preprocessing.token_stats import calculate_token_statistics
from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger("prepare_dataset")

PRIORITY_MAP = {
    "critical": "P1 · Critical",
    "p1": "P1 · Critical",
    "high": "P2 · High",
    "p2": "P2 · High",
    "medium": "P3 · Medium",
    "p3": "P3 · Medium",
    "low": "P4 / Low",
    "p4": "P4 / Low",
}

def clean_ticket_prefix(text: str) -> str:
    """Removes simulated ticket system prefixes e.g. [NET-20240], [IAM-33992], #44221 from text."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.strip()
    # Strip bracketed system tags like [NET-20240], [IAM-33992], [INC-CLD-13634]
    cleaned = re.sub(r"^\[[A-Za-z0-9_\-]+\]\s*", "", cleaned).strip()
    # Strip leading issue numbers like #44221:
    cleaned = re.sub(r"^#\d+\s*[:-]?\s*", "", cleaned).strip()
    return cleaned

def clean_summary_text(summary: str) -> str:
    """Ensures clean 1-sentence summary ending with period, stripping filler prefixes."""
    if not summary or not isinstance(summary, str):
        return ""
    s = summary.strip()
    s = re.sub(r"^(?:summary|ticket summary|resolution)\s*[:-]\s*", "", s, flags=re.IGNORECASE).strip()
    if s and s[-1].isalnum():
        s += "."
    return s

def prepare_20k_dataset(csv_path: Optional[Path] = None):
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "raw" / "it_cloud_support_tickets_20k.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Source raw dataset not found at: {csv_path}")

    print("=" * 70)
    print("   PREPARING 20,000 IT & CLOUD SUPPORT DATASET FOR STUDENT TRAINING")
    print("=" * 70)
    print(f"Source CSV       : {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Total Rows Ingested: {len(df):,}")

    records = []
    for idx, row in df.iterrows():
        t_id = f"IT-{idx+1:05d}"
        clean_text = clean_ticket_prefix(str(row.get("ticket_text", "")))
        clean_sum = clean_summary_text(str(row.get("summary", "")))
        
        raw_prio = str(row.get("priority", "Medium")).strip().lower()
        prio = PRIORITY_MAP.get(raw_prio, "P3 · Medium")
        
        cat = str(row.get("category", "Cloud Infrastructure")).strip()
        intent = str(row.get("intent", "Configuration Error")).strip()
        origin = str(row.get("issue_origin", "COMPANY_SIDE")).strip().upper()
        if origin not in ["COMPANY_SIDE", "CLIENT_SIDE"]:
            origin = "COMPANY_SIDE"

        records.append({
            "ticket_id": t_id,
            "ticket_text": clean_text,
            "summary": clean_sum,
            "priority": prio,
            "category": cat,
            "intent": intent,
            "sector": "IT & Cloud Operations",
            "domain": "IT & Cloud Operations",
            "issue_origin": origin
        })

    print(f"Cleaned & Normalized: {len(records):,} records")
    print(f"Sample Row [IT-00004]:")
    print(f"  Text   : {records[3]['ticket_text'][:90]}...")
    print(f"  Summary: {records[3]['summary']}")
    print(f"  Prio   : {records[3]['priority']} | Category: {records[3]['category']} | Origin: {records[3]['issue_origin']}")

    # Split dataset: 70% Train (14,000), 15% Val (3,000), 15% Test (3,000)
    splitter = DatasetSplitter(seed=42)
    train_records, val_records, test_records = splitter.split_dataset(
        records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = processed_dir / "test.jsonl"

    splitter.save_split(train_records, train_path)
    splitter.save_split(val_records, val_path)
    splitter.save_split(test_records, test_path)

    print("\n" + "-" * 70)
    print("DETERMINISTIC DATASET SPLIT COMPLETE (70/15/15):")
    print(f"  Train Set      : {len(train_records):,} rows -> {train_path}")
    print(f"  Validation Set : {len(val_records):,} rows -> {val_path}")
    print(f"  Test Set       : {len(test_records):,} rows -> {test_path}")

    # Compute Token Statistics
    print("\n" + "-" * 70)
    print("COMPUTING TOKEN DISTRIBUTION STATISTICS:")
    ticket_texts = [r["ticket_text"] for r in records]
    summary_texts = [r["summary"] for r in records]
    input_stats = calculate_token_statistics(ticket_texts)
    target_stats = calculate_token_statistics(summary_texts)
    print(f"  Total Tickets  : {input_stats.total_samples:,}")
    print(f"  Avg Input Len  : {input_stats.mean_length:.1f} words (P95: {input_stats.p95_length:.0f})")
    print(f"  Avg Summary Len: {target_stats.mean_length:.1f} words (P95: {target_stats.p95_length:.0f})")
    print(f"  Max Input Len  : {input_stats.max_length} words")
    print("=" * 70 + "\n")

    return {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "input_stats": input_stats.model_dump(),
        "target_stats": target_stats.model_dump()
    }

if __name__ == "__main__":
    prepare_20k_dataset()
