import sys
from typing import Union

from loguru import logger

LOG_LEVEL = None


def set_logger(level: str = "INFO", filename: str = "matten.log", stderr: bool = True):
    """
    Set up loguru loggers.

    The default logger has a log level of `INFO` and logs to stderr. Call this function
    at the beginning of your script to override it.

    Args:
        level: log level, e.g. DEBUG, INFO, WARNING, ERROR, and CRITICAL.
        filename: name of the file to log to
        stderr: whether to log to  stderr.
    """
    global LOG_LEVEL

    level = level.upper()
    LOG_LEVEL = level

    file_handler = {"sink": filename, "level": level}
    stderr_handler = {"sink": sys.stderr, "level": level}

    if stderr:
        config = {"handlers": [stderr_handler, file_handler]}
    else:
        config = {"handlers": [file_handler]}

    logger.configure(**config)


def get_log_level() -> Union[None, str]:
    """
    Get the current log level.

    Returns:
        Log level in str, one of DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    """

    return LOG_LEVEL
