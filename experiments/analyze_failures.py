import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.failure_analysis import FailureAnalyzer
from app.utils.logger import get_logger

logger = get_logger("analyze_failures")

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Failure-Case Analysis CLI")
    parser.add_argument("--results_file", type=str, default=None, help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory to save failure report")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.output_dir

    # Find latest evaluation results file containing raw_results
    if args.results_file:
        target_files = [Path(args.results_file)]
    else:
        target_files = sorted(
            list(results_dir.glob("baseline_results_*.json")) + list(results_dir.glob("phase*.json")),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

    raw_results = []
    target_file = None
    for f_path in target_files:
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("raw_results") or data.get("raw_records") or []
        if records:
            raw_results = records
            target_file = f_path
            break

    if not raw_results:
        print("Error: No raw ticket results found in experiments/results/")
        sys.exit(1)

    print(f"Loaded {len(raw_results)} records for failure audit.")

    analyzer = FailureAnalyzer()
    report = analyzer.analyze_dataset_results(raw_results)

    # Save summary report to output_dir
    report_output_path = results_dir / "failure_analysis_report.json"
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    print("\n" + "=" * 65)
    print("             FAILURE-CASE ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Total Audited Samples: {report.total_audited}")
    print(f"Total Failure Count  : {report.total_failures}")
    print(f"Overall Failure Rate : {report.failure_rate_pct}%")
    
    print("\n[FAILURE CATEGORY BREAKDOWN]")
    for cat, count in report.category_counts.items():
        print(f"  - {cat:20s}: {count} occurrences")

    print("\n[PER-TICKET FAILURE TRACES]")
    for r in report.ticket_reports:
        if r.has_failure:
            print(f"  Ticket ID: {r.ticket_id}")
            print(f"    Types  : {', '.join(r.failure_types)}")
            for d in r.details:
                print(f"    Detail : {d}")
        else:
            print(f"  Ticket ID: {r.ticket_id} -> [CLEAN]")

    print("=" * 65)
    print(f"Full Failure Report saved to: {report_output_path}")

if __name__ == "__main__":
    main()
