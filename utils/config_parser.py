"""Configuration parser."""

from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig


def parse_config(path: Path, name: str) -> DictConfig:
    """Parse model parameters."""
    with initialize(config_path=path, version_base=None):
        cfg = compose(config_name=name)
    return cfg.params


if __name__ == "__main__":
    config_path = Path("../config/model")
    config_name = "unet"
    config = parse_config(path=config_path, name=config_name)
    print(config)
