#!/usr/bin/env python3
from __future__ import annotations

from typing import Union, Sequence

from dotenv import load_dotenv

load_dotenv()

import logging

from tap import Tap

from impressions_evaluation.experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    ensure_datasets_exist,
    EHyperParameterTuningParameters,
    RecommenderBaseline,
    RecommenderImpressions,
)
from impressions_evaluation.experiments.graph_based import (
    _run_collaborative_filtering_hyper_parameter_tuning,
    _run_pure_impressions_hyper_parameter_tuning,
    run_experiments_sequentially,
)
from impressions_evaluation.experiments.graph_based.results import print_results


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if 
    possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the pure collaborative recommenders"""

    include_impressions: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the collaborative+impressions recommenders"""

    print_evaluation_results: bool = False
    """Export to CSV and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of 
    all tuned recommenders."""

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
    # Benchmarks.FINNNoSlates,
]


_TO_USE_RECOMMENDERS_BASELINES = [
    # RecommenderBaseline.P3_ALPHA,
    # RecommenderBaseline.RP3_BETA,
    RecommenderBaseline.LIGHT_GCN,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS = [
    # RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
    # RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    # RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
    # RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS,
    RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS,
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
    # EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_5_2,
]

_TO_USE_TRAINING_FUNCTIONS_BASELINES = [
    _run_collaborative_filtering_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_PURE_IMPRESSIONS = [
    _run_pure_impressions_hyper_parameter_tuning,
]

_TO_PRINT_RECOMMENDERS: list[
    tuple[
        Benchmarks,
        EHyperParameterTuningParameters,
        list[Sequence[Union[RecommenderBaseline, RecommenderImpressions]]],
    ]
] = [
    (
        Benchmarks.MINDSmall,
        EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
        [
            (
                RecommenderBaseline.P3_ALPHA,
                RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
                RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            ),
            (
                RecommenderBaseline.RP3_BETA,
                RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
                RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            ),
        ],
    ),
    (
        Benchmarks.ContentWiseImpressions,
        EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
        [
            (
                RecommenderBaseline.P3_ALPHA,
                RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
                RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            ),
            (
                RecommenderBaseline.RP3_BETA,
                RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
                RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            ),
        ],
    ),
    # (
    #     Benchmarks.FINNNoSlates,
    #     EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
    #     [
    #         (
    #             RecommenderBaseline.P3_ALPHA,
    #             RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
    #             RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    #         ),
    #         (
    #             RecommenderBaseline.RP3_BETA,
    #             RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
    #             RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    #         ),
    #     ],
    # ),
]


if __name__ == "__main__":
    input_flags = ConsoleArguments().parse_args()

    logger = logging.getLogger(__name__)

    common_hyper_parameter_tuning_parameters = HyperParameterTuningParameters()

    experiments_interface_baselines = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_BASELINES,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_BASELINES,
    )

    experiments_impressions_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_PURE_IMPRESSIONS,
    )

    create_necessary_folders(
        benchmarks=experiments_interface_baselines.benchmarks,
        evaluation_strategies=experiments_interface_baselines.evaluation_strategies,
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_baselines:
        run_experiments_sequentially(
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_impressions:
        run_experiments_sequentially(
            experiment_cases_interface=experiments_impressions_interface,
        )

    if input_flags.print_evaluation_results:
        print_results(
            results_interface=_TO_PRINT_RECOMMENDERS,
        )

    logger.info(f"Finished running script: {__file__}")
