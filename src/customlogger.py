import logging
from pathlib import Path
from src.customformater import CustomFormatter


def get_logger(logger_name="logger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    Path("logs/").mkdir(exist_ok=True)

    fh = logging.FileHandler("logs/" + logger_name + ".log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    fh.setFormatter(CustomFormatter())

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
