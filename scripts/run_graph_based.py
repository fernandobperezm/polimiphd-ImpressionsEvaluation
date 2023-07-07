#!/usr/bin/env python3
from __future__ import annotations

import logging
from tap import Tap


from impressions_evaluation.experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    plot_popularity_of_datasets,
    ensure_datasets_exist,
    EHyperParameterTuningParameters,
    RecommenderBaseline,
    RecommenderImpressions,
)
from impressions_evaluation.experiments.graph_based import (
    _run_collaborative_filtering_hyper_parameter_tuning,
    _run_pure_impressions_hyper_parameter_tuning,
    run_experiments,
)
from impressions_evaluation.experiments.confidence_intervals import (
    compute_confidence_intervals,
)
from impressions_evaluation.experiments.print_results import print_results
from impressions_evaluation.experiments.print_statistics import (
    print_datasets_statistics,
)
from impressions_evaluation.experiments.statistical_tests import (
    compute_statistical_tests,
)
from recsys_framework_extensions.dask import configure_dask_cluster


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if 
    possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the pure collaborative recommenders"""

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
    # Benchmarks.MINDSmall,
    # Benchmarks.FINNNoSlates,
]

_TO_USE_RECOMMENDERS_BASELINES = [
    RecommenderBaseline.LIGHT_GCN,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS = [
    # RecommenderImpressions.LIGHT_GCN_ONLY_IMPRESSIONS,
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


if __name__ == "__main__":
    input_flags = ConsoleArguments().parse_args()

    logger = logging.getLogger(__name__)

    dask_interface = configure_dask_cluster()

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
        run_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_impressions:
        run_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_impressions_interface,
        )

    dask_interface.wait_for_jobs()

    if input_flags.compute_confidence_intervals:
        compute_confidence_intervals(
            dask_interface=dask_interface,
            experiment_cases_interface_baselines=experiments_interface_baselines,
            experiment_cases_interface_impressions_heuristics=experiments_impressions_heuristics_interface,
            experiment_cases_interface_impressions_re_ranking=experiments_impressions_re_ranking_interface,
            experiment_cases_interface_impressions_user_profiles=experiments_impressions_user_profiles_interface,
        )

    if input_flags.compute_statistical_tests:
        compute_statistical_tests(
            dask_interface=dask_interface,
            experiment_cases_interface_baselines=experiments_interface_baselines,
        )

    dask_interface.wait_for_jobs()

    if input_flags.plot_popularity_of_datasets:
        plot_popularity_of_datasets(
            experiments_interface=experiments_interface_baselines
        )

    if input_flags.print_datasets_statistics:
        print_datasets_statistics(
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.print_evaluation_results:
        print_results(
            baseline_experiment_cases_interface=experiments_interface_baselines,
            impressions_heuristics_experiment_cases_interface=experiments_impressions_heuristics_interface,
            ablation_re_ranking_experiment_cases_interface=experiments_ablation_impressions_re_ranking_interface,
            re_ranking_experiment_cases_interface=experiments_impressions_re_ranking_interface,
            user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
        )

    dask_interface.close()