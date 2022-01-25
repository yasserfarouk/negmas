#!/usr/bin/env python
"""
A set of utilities that can be used by agents developed for the platform.

This set of utlities can be extended but must be backward compatible for at
least two versions
"""
from __future__ import annotations

import datetime
import logging
import os
import sys

import colorlog

__all__ = [
    "create_loggers",
]
COMMON_LOG_FILE_NAME = "./logs/{}_{}.txt".format(
    "log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

MODULE_LOG_FILE_NAME: dict[str, str] = dict()

LOGS_BASE_DIR = "./logs"


def create_loggers(
    file_name: str | None = None,
    module_name: str | None = None,
    screen_level: int | None = logging.WARNING,
    file_level: int | None = logging.DEBUG,
    format_str: str = "%(asctime)s - %(levelname)s - %(message)s",
    colored: bool = True,
    app_wide_log_file: bool = True,
    module_wide_log_file: bool = False,
) -> logging.Logger:
    """
    Create a set of loggers to report feedback.

    The logger created can log to both a file and the screen at the  same time
    with adjustable level for each of them. The default is to log everything to
    the file and to log WARNING at least to the screen

    Args:
        module_wide_log_file:
        app_wide_log_file:
        file_name: The file to export_to the logs to. If None only the screen
                    is used for logging. If empty, a time-stamp is used
        module_name: The module name to use. If not given the file name
                    without .py is used
        screen_level: level of the screen logger
        file_level: level of the file logger
        format_str: the format of logged items
        colored: whether or not to try using colored logs

    Returns:
        logging.Logger: The logger

    """
    if module_name is None:
        module_name = __file__.split("/")[-1][:-3]
    # create logger if it does not already exist
    logger = None
    if module_wide_log_file or app_wide_log_file:
        logger = logging.getLogger(module_name)
        if len(logger.handlers) > 0:
            return logger
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
    # create formatter
    file_formatter = logging.Formatter(format_str)
    if colored and "colorlog" in sys.modules and os.isatty(2) and screen_level:
        date_format = "%Y-%m-%d %H:%M:%S"
        cformat = "%(log_color)s" + format_str
        screen_formatter = colorlog.ColoredFormatter(
            cformat,
            date_format,
            log_colors={
                "DEBUG": "magenta",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        screen_formatter = logging.Formatter(format_str)
    if screen_level is not None and (module_wide_log_file or app_wide_log_file):
        # create console handler and set level to logdebug
        screen_logger = logging.StreamHandler()
        screen_logger.setLevel(screen_level)
        # add formatter to ch
        screen_logger.setFormatter(screen_formatter)
        # add ch to logger
        logger.addHandler(screen_logger)
    if file_name is not None and file_level is not None:
        file_name = str(file_name)
        if logger is None:
            logger = logging.getLogger(file_name)
            logger.setLevel(file_level)
        if len(file_name) == 0:
            if app_wide_log_file:
                file_name = COMMON_LOG_FILE_NAME
            elif module_wide_log_file and module_name in MODULE_LOG_FILE_NAME.keys():
                file_name = MODULE_LOG_FILE_NAME[module_name]
            else:
                file_name = "{}/{}_{}.txt".format(
                    LOGS_BASE_DIR,
                    module_name,
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
                MODULE_LOG_FILE_NAME[module_name] = file_name

            os.makedirs(f"{LOGS_BASE_DIR}", exist_ok=True)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)  # type: ignore
        file_logger = logging.FileHandler(file_name)
        file_logger.setLevel(file_level)
        file_logger.setFormatter(file_formatter)
        logger.addHandler(file_logger)
    return logger
