"""Sample hyper parameter tuning."""

import logging
from pathlib import Path

import numpy as np
import optuna
from omegaconf import OmegaConf

from model.train import Trainer
from utils.config_parser import parse_config
from utils.configure_logger import configure_logger

configure_logger()
logger = logging.getLogger(__name__)


def train(params) -> float:
    """Model training with the optuna suggested parameters."""
    logger.info(OmegaConf.to_yaml(params))
    trainer = Trainer(params)
    losses = trainer.train()
    # best validation loss for the current model
    bet_val_loss = min(np.array(losses)[:, 1])
    return bet_val_loss


def objective(trial) -> float:
    """Objective function that optuna is trying to optimise.

    Must return the value that optuna is trying to maximise or minimise.
    """
    config_path = Path("../config/model")
    config_name = "unet"

    logger.info(f"Loading config {config_name} from {config_path}")

    params = parse_config(config_path, config_name)

    params.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params.loss.gamma = trial.suggest_float("gamma", 0.5, 3, log=False)

    params.loss.alpha = trial.suggest_categorical(
        "alpha", [[1, 2, 1, 2, 1, 1], [1, 2, 1, 3, 1, 1]]
    )

    min_loss = train(params)
    logger.info(f"Loss for current trial: {min_loss:.4f}")

    return min_loss


def run_study():
    """Optimisation."""
    logger.info("Starting optuna hyper parameter tuning.")
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=50)


if __name__ == "__main__":
    run_study()
