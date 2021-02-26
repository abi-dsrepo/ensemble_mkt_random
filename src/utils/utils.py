import logging
from logging import Logger
import sys


def get_logger(name) -> Logger:
    """Initializes logger

    Returns:
        [Logger]: [logger instance]
    """
    logger = logging.getLogger(f"Delivery_Hero_{name}")
    logger.setLevel("INFO")
    format = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - [%(name)s] : %(message)s"
    )
    loginStreamHandler = logging.StreamHandler(sys.stdout)
    loginStreamHandler.setFormatter(format)
    logger.addHandler(loginStreamHandler)
    return logger
