"""Logger configuration."""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def configure_logger(
    log_path: Optional[Path] = None, level=logging.INFO
) -> None:
    """Configure the logging."""
    if log_path is None:
        log_path = Path("../.logs/logs.log")
    handler1: logging.Handler = TimedRotatingFileHandler(
        log_path, when="D", interval=1, backupCount=3
    )
    handler2: logging.Handler = logging.StreamHandler()
    handlers = [handler1, handler2]

    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d] [PID-%(process)d] "
        "[%(funcName)s-%(lineno)d]-[%(levelname)s] : %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=handlers,
    )
