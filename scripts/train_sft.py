import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.training.sft_trainer import StudentSFTTrainer
from app.utils.logger import get_logger

logger = get_logger("train_sft")

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for Student Model")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml", help="Path to SFT YAML config")
    parser.add_argument("--train_path", type=str, default="data/processed/train.jsonl", help="Path to train split")
    parser.add_argument("--val_path", type=str, default="data/processed/val.jsonl", help="Path to validation split")
    args = parser.parse_args()

    sft_config_path = PROJECT_ROOT / args.config
    train_path = PROJECT_ROOT / args.train_path
    val_path = PROJECT_ROOT / args.val_path

    print("=" * 65)
    print("    TICKET SUMMARIZATION DaaS — PHASE 3 SUPERVISED FINE-TUNING (SFT)")
    print("=" * 65)

    settings = get_settings()
    print(f"Base Model Target  : {settings.model.name}")
    print(f"SFT Config File    : {sft_config_path}")
    print(f"Training Dataset   : {train_path}")
    print(f"Validation Dataset : {val_path}")

    trainer = StudentSFTTrainer(config_path=sft_config_path, settings=settings)
    saved_dir = trainer.train(train_path=train_path, val_path=val_path)

    print("\n" + "=" * 65)
    print("            SFT TRAINING COMPLETE")
    print("=" * 65)
    print(f"Saved LoRA Adapter Checkpoints to: {saved_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
