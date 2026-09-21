from pathlib import Path
from app.training.sft_trainer import StudentSFTTrainer, format_sft_sample

def test_format_sft_sample():
    ticket = "User unable to reset password."
    summary = "Password reset issue."
    formatted = format_sft_sample(ticket, summary)
    assert ticket in formatted
    assert summary in formatted
    assert "Summary:" in formatted

def test_sft_trainer_init(tmp_path):
    trainer = StudentSFTTrainer()
    assert trainer.sft_config is not None
    assert "training" in trainer.sft_config
    assert "peft" in trainer.sft_config
