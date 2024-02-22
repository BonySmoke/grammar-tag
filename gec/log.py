import logging
import sys

LOG_LEVEL = 'DEBUG'


def create_logger(name='main'):
    """
    Create a logging object if it doesn't exist
    """
    # check if the logger exists
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(LOG_LEVEL)
    fmt = '%(asctime)s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt=fmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    return logger
