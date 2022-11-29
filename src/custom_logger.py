import logging
from pathlib import Path


class CustomFilter(logging.Filter):
    def __init__(self, url_name=""):
        self.url_name = url_name

    def filter(self, record):
        if (not hasattr(record, "url_name")) or record.url_name == "":
            record.url_name = self.url_name
        return True


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)s %(process)6d [%(funcName)14s] (%(filename)s:%(lineno)3d): %(url_name)-16.16s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(logger_name="logger", url_name=""):
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
    logger.addFilter(CustomFilter(url_name))
    return logger
