"""Logging utilities."""

import logging


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Sets up logging for the PyTorch project.

    Args:
        log_level (int): The logging level to use.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s,%(msecs)03d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
