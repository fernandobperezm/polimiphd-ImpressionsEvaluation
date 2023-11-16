#!/usr/bin/env python3
from __future__ import annotations

import logging

from dotenv import load_dotenv
from tap import Tap

from impressions_evaluation.experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    ensure_datasets_exist,
    EHyperParameterTuningParameters,
    RecommenderImpressions,
)
from impressions_evaluation.experiments.graph_based import (
    _run_frequency_impressions_hyper_parameter_tuning,
    run_experiments_sequentially,
)

from Recommenders.MatrixFactorization.Cython.MatrixFactorizationImpressions_Cython import (
    MatrixFactorization_BPR_Cython,
)


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if 
    possible and then processes the data to create the splits."""

    include_impressions: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the collaborative+impressions recommenders"""

    use_gpu: bool = False
    """"""

    compute_confidence_intervals: bool = False
    """TODO: fernando-debugger"""

    compute_statistical_tests: bool = False
    """TODO: fernando-debugger"""

    print_evaluation_results: bool = False
    """Export to CSV and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of 
    all tuned recommenders."""

    print_datasets_statistics: bool = False
    """Export to CSV statistics on the different sparse matrices existing for each dataset."""

    plot_popularity_of_datasets: bool = False
    """Creates plots depicting the popularity of each dataset split."""

    send_email: bool = False
    """Send a notification email via GMAIL when experiments start and finish."""


####################################################################################################
####################################################################################################
#                                            MAIN                                                  #
####################################################################################################
####################################################################################################
_TO_USE_BENCHMARKS = [
    Benchmarks.ContentWiseImpressions,
    Benchmarks.MINDSmall,
    Benchmarks.FINNNoSlates,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS = [
    RecommenderImpressions.SOFT_FREQUENCY_CAPPING,
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
    # EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_5_2,
]

_TO_USE_TRAINING_FUNCTIONS_FREQUENCY_IMPRESSIONS = [
    _run_frequency_impressions_hyper_parameter_tuning,
]


if __name__ == "__main__":
    load_dotenv()

    input_flags = ConsoleArguments().parse_args()

    logger = logging.getLogger(__name__)

    common_hyper_parameter_tuning_parameters = HyperParameterTuningParameters()

    experiments_impressions_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_FREQUENCY_IMPRESSIONS,
    )

    create_necessary_folders(
        benchmarks=experiments_impressions_interface.benchmarks,
        evaluation_strategies=experiments_impressions_interface.evaluation_strategies,
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(to_use_benchmarks=experiments_impressions_interface)

    if input_flags.include_impressions:
        run_experiments_sequentially(
            experiment_cases_interface=experiments_impressions_interface,
        )
