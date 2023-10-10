#!/usr/bin/env python3
from __future__ import annotations

import logging
import signal

from impressions_evaluation import configure_logger
from recsys_framework_extensions.dask import configure_dask_cluster
from tap import Tap

from dotenv import load_dotenv


load_dotenv()


class ConsoleArguments(Tap):
    setup_dask_local_cluster: bool = False


if __name__ == "__main__":
    input_flags = ConsoleArguments().parse_args()

    configure_logger()
    logger = logging.getLogger("__main__")

    if input_flags.setup_dask_local_cluster:
        logger.info(f"Creating dask local cluster")

        while True:
            dask_interface = configure_dask_cluster()

            # Putting thread to sleep until a signal arrives, which should only be kills from the OS when restarting the AWS instance.
            # See https://stackoverflow.com/a/31577274
            logger.info(
                "Successfully created/connected to dask cluster with config %(config)s. Now, the thread is going to pause.",
                {"config": dask_interface._config},
            )
            signal.pause()

    logger.info(f"Finished running script: {__file__}")
