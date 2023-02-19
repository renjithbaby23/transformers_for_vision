"""Model training entrypoint."""

import logging
from pathlib import Path

from omegaconf import OmegaConf

from model.train import Trainer, plot_loss
from utils.config_parser import parse_config
from utils.configure_logger import configure_logger

if __name__ == "__main__":
    configure_logger()
    logger = logging.getLogger()

    config_path = Path("../config/model")
    config_name = "unet"
    cfg = parse_config(config_path, config_name)
    logger.info("Starting training with the following configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    losses = trainer.train()

    plot_loss(losses=losses)
