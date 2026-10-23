import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.teacher.teacher_llm import get_teacher_llm, TeacherLLM, TeacherResponse
from app.utils.logger import get_logger

logger = get_logger("generate_teacher_labels")

def process_file_with_teacher(teacher: TeacherLLM, input_file: Path, output_file: Path, max_samples: Optional[int] = None):
    if not input_file.exists():
        logger.warning(f"Input split file not found: {input_file}")
        return

    records: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples and max_samples > 0:
        records = records[:max_samples]

    logger.info(f"Generating Teacher LLM labels for {len(records)} records from {input_file}...")

    distill_records: List[Dict[str, Any]] = []
    for i, rec in enumerate(records, 1):
        t_id = rec.get("ticket_id", "UNKNOWN")
        t_text = rec.get("ticket_text", "")

        teacher_resp: TeacherResponse = teacher.generate_summary(t_id, t_text)
        print(f"[{i}/{len(records)}] Ticket {t_id} processed by {teacher_resp.teacher_model} ({teacher_resp.latency_ms:.1f}ms)")

        # Augment record with teacher supervision label
        augmented = dict(rec)
        augmented["teacher_summary"] = teacher_resp.teacher_summary
        augmented["teacher_model"] = teacher_resp.teacher_model
        augmented["teacher_provider"] = teacher_resp.provider
        augmented["teacher_latency_ms"] = teacher_resp.latency_ms

        distill_records.append(augmented)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in distill_records:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Saved {len(distill_records)} teacher-augmented records to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Teacher LLM Distillation Supervision Labels")
    parser.add_argument("--provider", type=str, default=None, help="Teacher provider override (groq, gemini, openai, anthropic, mock)")
    parser.add_argument("--max_samples", type=int, default=25, help="Max records per split to label (default: 25, use -1 for all)")
    args = parser.parse_args()

    print("=" * 65)
    print("      TICKET SUMMARIZATION DaaS — TEACHER LABEL GENERATION")
    print("=" * 65)

    teacher = get_teacher_llm(provider_override=args.provider)
    print(f"Teacher Provider   : {teacher.provider_name}")
    print(f"Teacher Model      : {teacher.model_name}")
    print(f"Max Samples/Split  : {args.max_samples if args.max_samples > 0 else 'ALL'}")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_in = processed_dir / "train.jsonl"
    val_in = processed_dir / "val.jsonl"

    train_out = processed_dir / "train_teacher_distill.jsonl"
    val_out = processed_dir / "val_teacher_distill.jsonl"

    process_file_with_teacher(teacher, train_in, train_out, max_samples=args.max_samples)
    process_file_with_teacher(teacher, val_in, val_out, max_samples=max(5, args.max_samples // 5) if args.max_samples > 0 else None)

    print("\n" + "=" * 65)
    print("          TEACHER DISTILLATION DATASETS GENERATED")
    print("=" * 65)
    print(f"Train Distill Dataset : {train_out}")
    print(f"Val Distill Dataset   : {val_out}")
    print("=" * 65)

if __name__ == "__main__":
    main()
