#!/usr/bin/env python3
from __future__ import annotations

from tap import Tap

from FINNNoReader import FinnNoSlatesConfig
from MINDReader import MINDSmallConfig, MINDLargeConfig
from ContentWiseImpressionsReader import ContentWiseImpressionsConfig
from experiments.commons import create_necessary_folders, DatasetInterface, Benchmarks, EvaluationStrategy
from experiments.evaluation import (
    run_evaluation_experiments,
    plot_popularity_of_datasets,
)
from recsys_framework_extensions.dask import configure_dask_cluster
from recsys_framework_extensions.logging import get_logger


class ConsoleArguments(Tap):
    run_evaluation: bool = False
    """Run Hyper-parameter tuning of recommenders on the Ciao, ML100K, and ML1M datasets. Which recommenders are 
    tuned depend on the presence of the options --include_baselines and --include_cfgan.
    """

    include_baselines: bool = False
    """Include baselines in the hyper-parameter tuning"""

    print_evaluation_results: bool = False
    """Print LaTeX tables containing the accuracy and beyond accuracy metrics of the hyper-parameter tuned 
    recommenders."""

    plot_popularity_of_datasets: bool = False
    """Creates plots depicting the popularity of each dataset split."""


####################################################################################################
####################################################################################################
#                                            MAIN                                                  #
####################################################################################################
####################################################################################################
if __name__ == '__main__':
    input_flags = ConsoleArguments().parse_args()

    logger = get_logger(__name__)

    dask_interface = configure_dask_cluster()

    # Training statistics.
    # CW - UserKNN - 3 GB Training - 250 sec/it
    # CW - ItemKNN - 3 GB Training - 320 sec/it
    # CW - PureSVD - 2 GB Training - 170 sec/it
    # CW - EASE-R - 28 GB Training - 400 sec/it
    # MINDSmall - EASE_R - 16GB Training - 80sec/it
    # MINDLarge - EASE_R - 29.3GB Training - 450sec/it
    # FINNNoSlates - EASE R - 12.4TB Training - No Training.
    dataset_interface = DatasetInterface(
        priorities=[
            # 40,
            20,
            # 30,
            20,
        ],
        benchmarks=[
            # Benchmarks.MINDLarge,
            Benchmarks.MINDSmall,
            # Benchmarks.FINNNoSlates,
            Benchmarks.ContentWiseImpressions,
        ],
        configs=[
            # MINDLargeConfig(),
            MINDSmallConfig(),
            # FinnNoSlatesConfig(),
            ContentWiseImpressionsConfig(),
        ],
        evaluations=[
            # EvaluationStrategy.LEAVE_LAST_K_OUT,
            # EvaluationStrategy.LEAVE_LAST_K_OUT,
            EvaluationStrategy.LEAVE_LAST_K_OUT,
            EvaluationStrategy.LEAVE_LAST_K_OUT,
        ]
    )

    create_necessary_folders(
        benchmarks=dataset_interface.benchmarks,
        evaluation_strategies=dataset_interface.evaluation_strategies,
    )

    if input_flags.run_evaluation:
        run_evaluation_experiments(
            include_baselines=input_flags.include_baselines,
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )

    dask_interface.wait_for_jobs()

    if input_flags.plot_popularity_of_datasets:
        plot_popularity_of_datasets(
            dataset_interface=dataset_interface,
        )

    # if input_flags.print_evaluation_results:
    #     print_reproducibility_results(
    #         dataset_interface=dataset_interface,
    #     )

    dask_interface.close()
