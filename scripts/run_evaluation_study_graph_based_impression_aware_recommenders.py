#!/usr/bin/env python3
from __future__ import annotations

import logging

from typing import Union
from tap import Tap

from dotenv import load_dotenv

load_dotenv()

from impressions_evaluation import configure_logger
from impressions_evaluation.experiments.commons import (
    ExperimentCasesGraphBasedStatisticalTestInterface,
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    ensure_datasets_exist,
    EHyperParameterTuningParameters,
    RecommenderBaseline,
    RecommenderImpressions,
)
from impressions_evaluation.experiments.hyperparameters import (
    DIR_ANALYSIS_HYPER_PARAMETERS,
    plot_parallel_hyper_parameters_recommenders,
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
from impressions_evaluation.experiments.graph_based.statistical_tests import (
    compute_graph_based_statistical_tests,
    export_graph_based_statistical_tests,
)


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """Ensures that datasets exists, i.e., it downloads the datasets if possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """Tunes the hyper-parameters of the pure collaborative recommenders"""

    include_impressions: bool = False
    """Tunes the hyper-parameters of graph-based recommenders using impressions, i.e., the UIM."""

    include_impressions_frequency: bool = False
    """Tunes the hyper-parameters of graph-based recommenders using impressions frequency, i.e., the UIM-F"""

    compute_statistical_tests: bool = False
    """Compute statistical significance tests comparing the performance of impression-aware recommenders against collaborative filtering baselines."""

    print_statistical_tests: bool = False
    """Exports to CSV the p-values of several statistical significance tests comparing the performance of impression-aware recommenders against collaborative filtering baselines."""

    print_evaluation_results: bool = False
    """Exports to Parquet, CSV, and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of all tuned recommenders."""

    analyze_hyper_parameters: bool = False
    """Exports to Parquet, CSV, Tikz, PDF, and PNG the parallel plots of hyper-parameters."""

    send_email: bool = False
    """Send a notification email via GMAIL when experiments start and finish."""


####################################################################################################
####################################################################################################
#                                            MAIN                                                  #
####################################################################################################
####################################################################################################
TO_USE_BENCHMARKS = [
    Benchmarks.ContentWiseImpressions,
    Benchmarks.MINDSmall,
    Benchmarks.FINNNoSlates,
]


TO_USE_RECOMMENDERS_BASELINES = [
    RecommenderBaseline.P3_ALPHA,
    RecommenderBaseline.RP3_BETA,
    # RecommenderBaseline.LIGHT_GCN,
]

TO_USE_RECOMMENDERS_IMPRESSIONS = [
    RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
    RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
    RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
    # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS,
    # RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS,
]

TO_USE_RECOMMENDERS_IMPRESSIONS_FREQUENCY = [
    RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS_FREQUENCY,
    RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
    RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS_FREQUENCY,
    RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
    # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS_FREQUENCY,
    # RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
]

TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
]

TO_USE_TRAINING_FUNCTIONS_BASELINES = [
    _run_collaborative_filtering_hyper_parameter_tuning,
]

TO_USE_TRAINING_FUNCTIONS_PURE_IMPRESSIONS = [
    _run_pure_impressions_hyper_parameter_tuning,
]

TO_USE_TRAINING_FUNCTIONS_IMPRESSIONS_FREQUENCY = [
    _run_frequency_impressions_hyper_parameter_tuning,
]


TO_PRINT_RECOMMENDERS: tuple[
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
    TO_USE_BENCHMARKS,
    TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
    [
        *[
            RecommenderBaseline.P3_ALPHA,
            RecommenderBaseline.RP3_BETA,
            # RecommenderBaseline.LIGHT_GCN,
        ],
        *[
            RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
            RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
            RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
            # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS,
            # RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS,
        ],
        *[
            RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS_FREQUENCY,
            RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
            # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS_FREQUENCY,
            # RecommenderImpressions.LIGHT_GCN_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
        ],
    ],
)


TO_USE_BENCHMARKS_RESULTS = [
    Benchmarks.ContentWiseImpressions,
    Benchmarks.MINDSmall,
    Benchmarks.FINNNoSlates,
]

TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16
]

TO_USE_SCRIPT_NAME = "script_evaluation_study_graph_based_impression_aware"


if __name__ == "__main__":
    input_flags = ConsoleArguments().parse_args()

    configure_logger()
    logger = logging.getLogger(__name__)

    common_hyper_parameter_tuning_parameters = HyperParameterTuningParameters()

    experiments_interface_baselines = ExperimentCasesInterface(
        to_use_benchmarks=TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=TO_USE_RECOMMENDERS_BASELINES,
        to_use_training_functions=TO_USE_TRAINING_FUNCTIONS_BASELINES,
    )

    experiments_impressions_interface = ExperimentCasesInterface(
        to_use_benchmarks=TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=TO_USE_RECOMMENDERS_IMPRESSIONS,
        to_use_training_functions=TO_USE_TRAINING_FUNCTIONS_PURE_IMPRESSIONS,
    )

    experiments_impressions_frequency_interface = ExperimentCasesInterface(
        to_use_benchmarks=TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=TO_USE_RECOMMENDERS_IMPRESSIONS_FREQUENCY,
        to_use_training_functions=TO_USE_TRAINING_FUNCTIONS_IMPRESSIONS_FREQUENCY,
    )

    experiments_statistical_tests_interface = ExperimentCasesGraphBasedStatisticalTestInterface(
        to_use_benchmarks=TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_script_name=TO_USE_SCRIPT_NAME,
        to_use_recommenders_baselines=[
            RecommenderBaseline.P3_ALPHA,
            RecommenderBaseline.RP3_BETA,
        ],
        to_use_recommenders_impressions=[
            [
                RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS,
                RecommenderImpressions.P3_ALPHA_ONLY_IMPRESSIONS_FREQUENCY,
                RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS,
                RecommenderImpressions.P3_ALPHA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
            ],
            [
                RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS,
                RecommenderImpressions.RP3_BETA_ONLY_IMPRESSIONS_FREQUENCY,
                RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS,
                RecommenderImpressions.RP3_BETA_DIRECTED_INTERACTIONS_IMPRESSIONS_FREQUENCY,
            ],
        ],
    )

    create_necessary_folders(
        benchmarks=experiments_interface_baselines.benchmarks,
        evaluation_strategies=experiments_interface_baselines.evaluation_strategies,
        script_name=TO_USE_SCRIPT_NAME,
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(
            to_use_benchmarks=TO_USE_BENCHMARKS,
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

    if input_flags.compute_statistical_tests:
        compute_graph_based_statistical_tests(
            experiment_cases_statistical_tests_interface=experiments_statistical_tests_interface,
        )

    if input_flags.print_statistical_tests:
        export_graph_based_statistical_tests(
            experiment_cases_statistical_tests_interface=experiments_statistical_tests_interface,
        )

    if input_flags.print_evaluation_results:
        process_results(
            results_interface=TO_PRINT_RECOMMENDERS,
            dir_csv_results=DIR_CSV_RESULTS,
            dir_latex_results=DIR_LATEX_RESULTS,
            dir_parquet_results=DIR_PARQUET_RESULTS,
            script_name=TO_USE_SCRIPT_NAME,
        )
        export_evaluation_results(
            benchmarks=TO_USE_BENCHMARKS,
            hyper_parameters=TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
            script_name=TO_USE_SCRIPT_NAME,
        )

    if input_flags.analyze_hyper_parameters:
        recommenders = [
            "P3alphaRecommender",
            "RP3betaRecommender",
            #
            "ExtendedP3AlphaRecommender",
            "ExtendedRP3BetaRecommender",
            # "ExtendedLightGCNRecommender",
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
            # "ImpressionsProfileLightGCNRecommender",
            # "ImpressionsProfileWithFrequencyLightGCNRecommender",
            # "ImpressionsDirectedLightGCNRecommender",
            # "ImpressionsDirectedWithFrequencyLightGCNRecommender",
        ]

        metrics_to_optimize = ["COVERAGE_ITEM", "NDCG"]
        cutoff_to_optimize = 10

        dir_analysis_hyper_parameters = DIR_ANALYSIS_HYPER_PARAMETERS.format(
            script_name=TO_USE_SCRIPT_NAME,
        )

        dir_parquet_results = DIR_PARQUET_RESULTS

        plot_parallel_hyper_parameters_recommenders(
            benchmarks=TO_USE_BENCHMARKS_RESULTS,
            hyper_parameters=TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS,
            recommenders=recommenders,
            metrics_to_optimize=metrics_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
            dir_analysis_hyper_parameters=dir_analysis_hyper_parameters,
        )

    logger.info(
        "Finished running script: %(script_name)s",
        {"script_name": __file__},
    )
