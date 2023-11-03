#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Union

from dotenv import load_dotenv

load_dotenv()

import logging

from tap import Tap

from impressions_evaluation import configure_logger
from impressions_evaluation.experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    ensure_datasets_exist,
    EHyperParameterTuningParameters,
    RecommenderBaseline,
    RecommenderImpressions,
    DIR_TRAINED_MODELS,
)
from impressions_evaluation.experiments.graph_based import (
    _run_collaborative_filtering_hyper_parameter_tuning,
    _run_pure_impressions_hyper_parameter_tuning,
    run_experiments_sequentially,
    _run_frequency_impressions_hyper_parameter_tuning,
)
from impressions_evaluation.experiments.graph_based.results import (
    process_results,
    export_evaluation_results,
    DIR_PARQUET_RESULTS,
    DIR_CSV_RESULTS,
    DIR_LATEX_RESULTS,
)
from impressions_evaluation.experiments.hyperparameters import (
    DIR_ANALYSIS_HYPER_PARAMETERS,
    plot_parallel_hyper_parameters_recommenders,
    distribution_hyper_parameters_graph_based_impression_aware_recommenders,
)


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if 
    possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the pure collaborative recommenders"""

    include_impressions: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the collaborative+impressions recommenders"""

    include_impressions_frequency: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the collaborative+impressions frequency recommenders"""

    print_evaluation_results: bool = False
    """Export to CSV and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of 
    all tuned recommenders."""

    analyze_hyper_parameters: bool = False
    """"""

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


_TO_USE_RECOMMENDERS_BASELINES = [
    RecommenderBaseline.P3_ALPHA,
    RecommenderBaseline.RP3_BETA,
    # RecommenderBaseline.LIGHT_GCN,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS = [
    RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
    RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
    RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS,
    # RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_FREQUENCY = [
    RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS_FREQUENCY,
    RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
    RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS_FREQUENCY,
    RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
    # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS_FREQUENCY,
    # RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
]

_TO_USE_TRAINING_FUNCTIONS_BASELINES = [
    _run_collaborative_filtering_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_PURE_IMPRESSIONS = [
    _run_pure_impressions_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_IMPRESSIONS_FREQUENCY = [
    _run_frequency_impressions_hyper_parameter_tuning,
]


_TO_PRINT_RECOMMENDERS: tuple[
    list[Benchmarks],
    list[EHyperParameterTuningParameters],
    list[
        Union[
            RecommenderBaseline,
            RecommenderImpressions,
            tuple[RecommenderBaseline, RecommenderImpressions],
        ]
    ],
] = (
    _TO_USE_BENCHMARKS,
    _TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
    [
        *[
            RecommenderBaseline.P3_ALPHA,
            RecommenderBaseline.RP3_BETA,
            RecommenderBaseline.LIGHT_GCN,
        ],
        *[
            RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
            RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
            RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS,
            RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS,
        ],
        *[
            RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
        ],
    ],
)


_TO_USE_BENCHMARKS_RESULTS = [
    Benchmarks.ContentWiseImpressions,
    Benchmarks.MINDSmall,
    Benchmarks.FINNNoSlates,
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16
]


if __name__ == "__main__":
    input_flags = ConsoleArguments().parse_args()

    configure_logger()
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

    experiments_impressions_frequency_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_FREQUENCY,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSIONS_FREQUENCY,
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

    if input_flags.include_impressions_frequency:
        run_experiments_sequentially(
            experiment_cases_interface=experiments_impressions_frequency_interface,
        )

    if input_flags.print_evaluation_results:
        dir_trained_models = os.path.join(
            DIR_TRAINED_MODELS,
            "script_graph_based_recommenders_with_impressions",
            "",
        )

        dir_latex_results = DIR_LATEX_RESULTS
        dir_csv_results = DIR_CSV_RESULTS
        dir_parquet_results = DIR_PARQUET_RESULTS

        process_results(
            results_interface=_TO_PRINT_RECOMMENDERS,
            dir_csv_results=dir_csv_results,
            dir_latex_results=dir_latex_results,
            dir_parquet_results=dir_parquet_results,
        )
        export_evaluation_results(
            benchmarks=_TO_USE_BENCHMARKS,
            hyper_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        )

    if input_flags.analyze_hyper_parameters:
        recommenders = [
            "P3alphaRecommender",
            "RP3betaRecommender",
            #
            "ExtendedP3AlphaRecommender",
            "ExtendedRP3BetaRecommender",
            "ExtendedLightGCNRecommender",
            #
            "ImpressionsProfileP3AlphaRecommender",
            "ImpressionsProfileWithFrequencyP3AlphaRecommender",
            "ImpressionsDirectedP3AlphaRecommender",
            "ImpressionsDirectedWithFrequencyP3AlphaRecommender",
            #
            "ImpressionsProfileRP3BetaRecommender",
            "ImpressionsProfileWithFrequencyRP3BetaRecommender",
            "ImpressionsDirectedRP3BetaRecommender",
            "ImpressionsDirectedWithFrequencyRP3BetaRecommender",
            #
            "ImpressionsProfileLightGCNRecommender",
            "ImpressionsProfileWithFrequencyLightGCNRecommender",
            "ImpressionsDirectedLightGCNRecommender",
            "ImpressionsDirectedWithFrequencyLightGCNRecommender",
        ]

        metrics_to_optimize = ["COVERAGE_ITEM", "NDCG"]
        cutoff_to_optimize = 10

        dir_analysis_hyper_parameters = os.path.join(
            DIR_ANALYSIS_HYPER_PARAMETERS,
            "script_graph_based_recommenders_with_impressions",
            "",
        )

        dir_parquet_results = DIR_PARQUET_RESULTS

        # No need to call the `distribution_hyper_parameters_graph_based_impression_aware_recommenders` function because
        # we only have five values in each hyper-parameter. Not enough to create a distribution map.
        # distribution_hyper_parameters_graph_based_impression_aware_recommenders(
        #     benchmarks=_TO_USE_BENCHMARKS_RESULTS,
        #     hyper_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS,
        #     dir_parquet_results=dir_parquet_results,
        #     dir_analysis_hyper_parameters=dir_analysis_hyper_parameters,
        # )

        # We can plot parallel coordinates.
        plot_parallel_hyper_parameters_recommenders(
            benchmarks=_TO_USE_BENCHMARKS_RESULTS,
            hyper_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS,
            recommenders=recommenders,
            metrics_to_optimize=metrics_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
            dir_analysis_hyper_parameters=dir_analysis_hyper_parameters,
        )

    logger.info(f"Finished running script: {__file__}")
