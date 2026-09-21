from app.training.distill_trainer import GKDSFTTrainer

def test_gkd_trainer_init():
    trainer = GKDSFTTrainer()
    assert trainer.gkd_config is not None
    assert "gkd" in trainer.gkd_config
    assert trainer.gkd_config["gkd"]["distill_alpha"] == 0.6
