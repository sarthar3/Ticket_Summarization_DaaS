"""
Script to evaluate the Priority Classifier's 4-Tier Operational Routing accuracy
across the complete 24,751 tickets in data/processed/test.jsonl.
"""
import sys
import json
import time
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.cleaner import TicketPreprocessor

PRIORITY_NORM_MAP = {
    "P1 · Critical": "Critical",
    "P1": "Critical",
    "Critical": "Critical",
    "P2 · High": "High",
    "P2": "High",
    "High": "High",
    "P3 · Medium": "Medium",
    "P3": "Medium",
    "Medium": "Medium",
    "P4 / Low": "Low",
    "P4": "Low",
    "Low": "Low",
}

def main():
    test_dataset_path = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    if not test_dataset_path.exists():
        print(f"Error: Dataset file not found at {test_dataset_path}")
        sys.exit(1)

    print("=" * 65)
    print("  PRIORITY CLASSIFIER 4-TIER OPERATIONAL ROUTING BENCHMARK")
    print(f"  Test Dataset: {test_dataset_path}")
    print("=" * 65)

    cleaner = TicketPreprocessor()

    total = 0
    correct = 0
    tier_totals = Counter()
    tier_correct = Counter()

    t_start = time.perf_counter()

    with open(test_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            true_priority = rec.get("priority", "Medium").strip()
            ticket_id = rec.get("ticket_id")
            ticket_text = rec.get("ticket_text", "")

            # Predict using preprocessor priority engine with ticket context
            pred_label, _ = cleaner.classify_priority_details(
                ticket_text=ticket_text,
                ticket_id=ticket_id,
                sector=rec.get("sector"),
                category=rec.get("category"),
                intent=rec.get("intent")
            )

            norm_pred = PRIORITY_NORM_MAP.get(pred_label, pred_label)
            norm_true = PRIORITY_NORM_MAP.get(true_priority, true_priority)

            total += 1
            tier_totals[norm_true] += 1
            if norm_pred.lower() == norm_true.lower():
                correct += 1
                tier_correct[norm_true] += 1

    t_elapsed = time.perf_counter() - t_start
    overall_acc = (correct / total * 100.0) if total > 0 else 0.0

    print("\n[EVALUATION RESULTS]")
    print(f"  Total Tickets Evaluated : {total:,}")
    print(f"  Total Correct Routings  : {correct:,}")
    print(f"  Overall Accuracy        : {overall_acc:.2f}%")
    print(f"  Evaluation Latency      : {t_elapsed:.2f}s ({(t_elapsed / total * 1000.0):.4f} ms/ticket)")

    print("\n[PER-TIER ACCURACY BREAKDOWN]")
    for tier in ["Critical", "High", "Medium", "Low"]:
        tot = tier_totals[tier]
        cor = tier_correct[tier]
        acc = (cor / tot * 100.0) if tot > 0 else 0.0
        print(f"  • {tier:<10}: {cor:,} / {tot:,} ({acc:.2f}%)")

    print("=" * 65)
    assert overall_acc >= 75.0, f"Priority accuracy {overall_acc:.2f}% is below target 75%!"
    print("✅ TARGET REACHED: Priority Classifier accuracy >= 75.00%!")
    print("=" * 65)

if __name__ == "__main__":
    main()
