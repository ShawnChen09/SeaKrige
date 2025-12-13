import logging
import sys


def setup_logger(name):
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("==%(levelname)s== %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
