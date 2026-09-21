import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.training.distill_trainer import GKDSFTTrainer
from app.utils.logger import get_logger

logger = get_logger("train_gkd")

def main():
    parser = argparse.ArgumentParser(description="Generalized Knowledge Distillation (GKD) for Student Model")
    parser.add_argument("--config", type=str, default="configs/gkd_config.yaml", help="Path to GKD YAML config")
    parser.add_argument("--train_path", type=str, default="data/processed/train_teacher_distill.jsonl", help="Path to train teacher distill split")
    parser.add_argument("--val_path", type=str, default="data/processed/val_teacher_distill.jsonl", help="Path to validation teacher distill split")
    args = parser.parse_args()

    gkd_config_path = PROJECT_ROOT / args.config
    train_path = PROJECT_ROOT / args.train_path
    val_path = PROJECT_ROOT / args.val_path

    print("=" * 65)
    print("    TICKET SUMMARIZATION DaaS — PHASE 6 KNOWLEDGE DISTILLATION (GKD)")
    print("=" * 65)

    settings = get_settings()
    print(f"Student Model Target : {settings.model.name}")
    print(f"GKD Config File      : {gkd_config_path}")
    print(f"Teacher Train Path   : {train_path}")
    print(f"Teacher Val Path     : {val_path}")

    trainer = GKDSFTTrainer(config_path=gkd_config_path, settings=settings)
    saved_dir = trainer.train(train_path=train_path, val_path=val_path)

    print("\n" + "=" * 65)
    print("         GKD KNOWLEDGE DISTILLATION COMPLETE")
    print("=" * 65)
    print(f"Saved Distilled Model Checkpoints to: {saved_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
