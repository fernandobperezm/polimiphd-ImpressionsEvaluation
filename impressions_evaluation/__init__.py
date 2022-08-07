import datetime
import logging.config
import sys
import os
import toml


with open(os.path.join(os.getcwd(), "pyproject.toml"), "r") as project_file:
    pyproject_logs_config = toml.load(
        f=project_file
    )["logs"]

_dir_logger = os.path.join(
    os.getcwd(),
    pyproject_logs_config['dir_logs'],
    "",
)
_filename_logger = os.path.join(
    _dir_logger,
    pyproject_logs_config['filename_logs'],
)

os.makedirs(_dir_logger, exist_ok=True)

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
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "main_formatter",
            "mode": "a",
            "filename": _filename_logger,
            "level": logging.DEBUG,
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
        }
    },
    "loggers": {
        "impressions_evaluation": {
            "level": logging.DEBUG,
            "propagate": False,
            "handlers": [
                "file",
                "console_out",
                "console_error",
            ]
        },
        "__main__": {
            "level": logging.DEBUG,
            "propagate": False,
            "handlers": [
                "file",
                "console_out",
                "console_error",
            ],
        },
    },
    "root": {
        "level": logging.WARNING,
        "handlers": [
            "file",
            "console_out",
            "console_error",
        ],
    }
}

logging.config.dictConfig(conf)
