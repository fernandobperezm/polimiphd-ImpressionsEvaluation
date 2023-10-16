import logging.config
import sys
import os
from typing import Optional

import toml


def configure_logger(
    filename_logs: Optional[str] = None,
) -> None:
    """Method used to ensure that we configure the logger config."""

    dir_logging = os.path.join(os.getcwd(), ".logs", "")
    filename_logging = "impressions_evaluation.log"
    filename_errors_logging = "impressions_evaluation.errors.log"

    os.makedirs(dir_logging, exist_ok=True)

    filepath_logging = os.path.join(dir_logging, filename_logging)
    filepath_error_logging = os.path.join(dir_logging, filename_errors_logging)

    # project_conf = os.path.join(os.getcwd(), "pyproject.toml")
    # with open(project_conf, "r") as project_file:
    #     pyproject_logs_config = toml.load(f=project_file)["logs"]

    # _filename = (
    #     filename_logs
    #     if filename_logs is not None
    #     else pyproject_logs_config["filename_logs"]
    # )
    # _filename_errors = f"{_filename}_errors"
    #
    # _dir_logger = os.path.join(
    #     os.getcwd(),
    #     pyproject_logs_config["dir_logs"],
    #     "",
    # )
    # _filename_logger = os.path.join(
    #     _dir_logger,
    #     _filename,
    # )

    # Definition of config dict seen at:
    # https://docs.python.org/3.9/library/logging.config.html#dictionary-schema-details
    conf = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "main_formatter": {
                "format": "%(process)d|%(asctime)s|%(levelname)s"
                "|%(name)s|%(module)s|%(filename)s|%(funcName)s|%(lineno)d"
                "|%(message)s",
                "validate": True,
            },
            "test_formatter": {
                "format": "%(levelname)s|TEST-FORMATTER|%(message)s",
                "validate": True,
            },
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "main_formatter",
                "filename": filepath_logging,
                "level": logging.DEBUG,
                "when": "midnight",
                "utc": True,
                "encoding": "utf-8",
            },
            "file_errors": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "main_formatter",
                "filename": filepath_error_logging,
                "level": logging.WARNING,
                "when": "midnight",
                "utc": True,
                "encoding": "utf-8",
            },
            "console_out": {
                "class": "logging.StreamHandler",
                "formatter": "main_formatter",
                "stream": sys.stdout,
                "level": logging.DEBUG,
            },
            "console_error": {
                "class": "logging.StreamHandler",
                "formatter": "main_formatter",
                "stream": sys.stderr,
                "level": logging.WARNING,
            },
            "test_out": {
                "class": "logging.StreamHandler",
                "formatter": "test_formatter",
                "stream": sys.stdout,
                "level": logging.DEBUG,
            },
        },
        "loggers": {
            "__main__": {
                "level": logging.DEBUG,
                "propagate": True,
                "handlers": ["file", "file_errors", "console_out", "console_error"],
            },
            "impressions_evaluation": {
                "level": logging.DEBUG,
                "propagate": True,
                "handlers": ["file", "file_errors", "console_out", "console_error"],
            },
            "recsys_framework_extensions": {
                "level": logging.WARNING,
                "propagate": True,
                "handlers": ["file", "file_errors", "console_out", "console_error"],
            },
            # If we want to leave the module logging defaults, then leave it commented. Else, to debug logging calls, then uncomment and modify it according to our needs.
            # "recsys_framework_extensions": {
            #     "level": logging.CRITICAL,
            #     "propagate": False,
            #     "handlers": [
            #         "file",
            #         "console_out",
            #         "console_error",
            #     ],
            # },
        },
    }

    logging.config.dictConfig(conf)

    # Ensure that main loggers are created.
    logger_main = logging.getLogger("__main__")
    logger_impressions_evaluation = logging.getLogger("impressions_evaluation")
    logger_recsys_framework_extensions = logging.getLogger(
        "recsys_framework_extensions"
    )

    logger_main.info("Created main logger")
    logger_impressions_evaluation.info("Created impressions_evaluation logger")
    logger_recsys_framework_extensions.info(
        "Created recsys_framework_extensions logger"
    )
