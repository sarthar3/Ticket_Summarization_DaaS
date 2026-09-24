"""
Script to ingest multi-sector raw CSV files from data/raw/
into standardized train.jsonl, val.jsonl, and test.jsonl dataset files.
"""
import sys
import glob
import json
import random
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SEED = 42

def ingest_all_csvs(max_records_per_file: int = None):
    print("=" * 60)
    print("INGESTING MULTI-SECTOR CSV DATASETS")
    print("=" * 60)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV dataset files in {RAW_DIR}\n")

    all_records = []

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file)
            if max_records_per_file:
                df = df.head(max_records_per_file)

            # Map column names if needed
            if "customer_issue" in df.columns and "ticket_text" not in df.columns:
                df.rename(columns={"customer_issue": "ticket_text"}, inplace=True)

            required_cols = ["ticket_id", "ticket_text", "summary"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"  ⚠️ Skipping {csv_file.name}: missing columns {missing}")
                continue

            for _, row in df.iterrows():
                ticket_id = str(row.get("ticket_id", "")).strip()
                ticket_text = str(row.get("ticket_text", "")).strip()
                summary = str(row.get("summary", "")).strip()

                if not ticket_id or not ticket_text or not summary or ticket_text == "nan" or summary == "nan":
                    continue

                rec = {
                    "ticket_id": ticket_id,
                    "ticket_text": ticket_text,
                    "summary": summary,
                    "sector": str(row.get("sector", "")).strip() if "sector" in row else None,
                    "intent": str(row.get("intent", "")).strip() if "intent" in row else None,
                    "category": str(row.get("category", "")).strip() if "category" in row else None,
                    "priority": str(row.get("priority", "")).strip() if "priority" in row else None,
                }
                all_records.append(rec)

            print(f"  ✅ Added records. Running total: {len(all_records):,}")

        except Exception as e:
            print(f"  ❌ Error reading {csv_file.name}: {e}")

    print("-" * 60)
    print(f"Total Clean Records Collected: {len(all_records):,}")

    if not all_records:
        print("❌ No valid records collected!")
        return

    # Shuffle deterministically
    rng = random.Random(SEED)
    rng.shuffle(all_records)

    n_total = len(all_records)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_records = all_records[:n_train]
    val_records = all_records[n_train:n_train + n_val]
    test_records = all_records[n_train + n_val:]

    print(f"Split Summary (70% / 15% / 15%):")
    print(f"  • Train Set : {len(train_records):,} records")
    print(f"  • Val Set   : {len(val_records):,} records")
    print(f"  • Test Set  : {len(test_records):,} records")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(records, filename):
        out_path = PROCESSED_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved: {out_path} ({len(records):,} lines)")

    write_jsonl(train_records, "train.jsonl")
    write_jsonl(val_records, "val.jsonl")
    write_jsonl(test_records, "test.jsonl")

    print("=" * 60)
    print("✅ DATASET INGESTION & PROCESSING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest CSV datasets into train/val/test jsonl")
    parser.add_argument("--max_per_file", type=int, default=None, help="Limit records per CSV file for fast testing")
    args = parser.parse_args()

    ingest_all_csvs(max_records_per_file=args.max_per_file)
