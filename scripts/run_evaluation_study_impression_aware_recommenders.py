#!/usr/bin/env python3
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from recsys_framework_extensions.dask import configure_dask_cluster
from tap import Tap

from impressions_evaluation import configure_logger
from impressions_evaluation.experiments.hyperparameters import (
    DIR_ANALYSIS_HYPER_PARAMETERS,
    distribution_hyper_parameters_plug_in_impression_aware_recommenders,
    plot_parallel_hyper_parameters_plug_in_impression_aware_recommenders,
)
from impressions_evaluation.experiments.print_results import (
    process_evaluation_results,
    export_evaluation_results,
    DIR_PARQUET_RESULTS,
)

load_dotenv()

from impressions_evaluation.experiments.baselines import (
    run_baselines_experiments,
    run_baselines_folded,
    _run_baselines_folded_hyper_parameter_tuning,
)
from impressions_evaluation.experiments.impression_aware.heuristics import (
    run_impressions_heuristics_experiments,
    _run_impressions_heuristics_hyper_parameter_tuning,
)
from impressions_evaluation.experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    ensure_datasets_exist,
    RecommenderImpressions,
    EHyperParameterTuningParameters,
    RecommenderBaseline,
    ExperimentCasesSignalAnalysisInterface,
)
from impressions_evaluation.experiments.graph_based import (
    _run_collaborative_filtering_hyper_parameter_tuning,
)
from impressions_evaluation.experiments.impression_aware.re_ranking import (
    run_impressions_re_ranking_experiments,
    run_ablation_impressions_re_ranking_experiments,
    run_signal_analysis_impressions_re_ranking_experiments,
    _run_impressions_re_ranking_hyper_parameter_tuning,
    _run_ablation_impressions_re_ranking_hyper_parameter_tuning,
    _run_signal_analysis_ablation_impressions_re_ranking_hyper_parameter_tuning,
    run_signal_analysis_ablation_impressions_re_ranking_experiments,
)
from impressions_evaluation.experiments.impression_aware.user_profiles import (
    run_impressions_user_profiles_experiments,
    _run_impressions_user_profiles_hyper_parameter_tuning,
)


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if 
    possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the base recommenders, e.g., ItemKNN, 
    UserKNN, SLIM ElasticNet."""

    include_folded: bool = False
    """If the flag is included, then the script folds the tuned matrix-factorization base recommenders. If the 
    recommenders are not previously tuned, then this flag fails."""

    include_impressions_heuristics: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of time-aware impressions recommenders: 
    Last Impressions, Recency, and Frequency & Recency. These recommenders do not need base recommenders to be tuned."""

    include_impressions_reranking: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of re-ranking impressions recommenders: 
    Cycling and Impressions Discounting. These recommenders need base recommenders to be tuned, if they aren't then 
    the method fails."""

    include_ablation_impressions_reranking: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of re-ranking impressions recommenders: 
    Impressions Discounting with only impressions frequency. These recommenders need base recommenders to be tuned, 
    if they aren't then the method fails."""

    include_signal_analysis: bool = False
    """TODO: fernando-debugger"""

    include_signal_analysis_reranking: bool = False
    """TODO: fernando-debugger"""

    include_impressions_profile: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of impressions as user profiles recommenders. 
    These recommenders need similarity-based recommenders to be tuned, if they aren't then the method fails."""

    compute_confidence_intervals: bool = False
    """TODO: fernando-debugger"""

    compute_statistical_tests: bool = False
    """TODO: fernando-debugger"""

    print_evaluation_results: bool = False
    """Export to CSV and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of 
    all tuned recommenders."""

    analyze_hyper_parameters: bool = False
    """TODO: fernando-debugger"""

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

_TO_USE_BENCHMARKS_RESULTS = [
    Benchmarks.ContentWiseImpressions,
    Benchmarks.MINDSmall,
    Benchmarks.FINNNoSlates,
]

_TO_USE_RECOMMENDERS_BASELINE = [
    RecommenderBaseline.RANDOM,
    RecommenderBaseline.TOP_POPULAR,
    #
    RecommenderBaseline.USER_KNN,
    RecommenderBaseline.ITEM_KNN,
    #
    RecommenderBaseline.P3_ALPHA,
    RecommenderBaseline.RP3_BETA,
    #
    RecommenderBaseline.PURE_SVD,
    RecommenderBaseline.NMF,
    RecommenderBaseline.MF_BPR,
    RecommenderBaseline.SVDpp,
    #
    RecommenderBaseline.SLIM_ELASTIC_NET,
    RecommenderBaseline.SLIM_BPR,
    #
    RecommenderBaseline.LIGHT_FM,
    RecommenderBaseline.EASE_R,
    RecommenderBaseline.MULT_VAE,
]

_TO_USE_RECOMMENDERS_BASELINE_FOLDED = [
    RecommenderBaseline.PURE_SVD,
    RecommenderBaseline.NMF,
    # RecommenderBaseline.MF_BPR,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS = [
    RecommenderImpressions.LAST_IMPRESSIONS,
    RecommenderImpressions.FREQUENCY_RECENCY,
    RecommenderImpressions.RECENCY,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING = [
    RecommenderImpressions.CYCLING,
    RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
    RecommenderImpressions.HARD_FREQUENCY_CAPPING,
]

_TO_USE_RECOMMENDERS_ABLATION_IMPRESSIONS_RE_RANKING = [
    RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES = [
    RecommenderImpressions.USER_WEIGHTED_USER_PROFILE,
    RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_SIGNAL_ANALYSIS = [
    (
        RecommenderImpressions.HARD_FREQUENCY_CAPPING,
        "SIGNAL_ANALYSIS_LESS_OR_EQUAL_THRESHOLD",
    ),
    (
        RecommenderImpressions.HARD_FREQUENCY_CAPPING,
        "SIGNAL_ANALYSIS_GREAT_OR_EQUAL_THRESHOLD",
    ),
    (
        RecommenderImpressions.CYCLING,
        "SIGNAL_ANALYSIS_SIGN_POSITIVE",
    ),
    (
        RecommenderImpressions.CYCLING,
        "SIGNAL_ANALYSIS_SIGN_NEGATIVE",
    ),
    (
        RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
        "SIGNAL_ANALYSIS_SIGN_ALL_POSITIVE",
    ),
    (
        RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
        "SIGNAL_ANALYSIS_SIGN_ALL_NEGATIVE",
    ),
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
    # EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_5_2,
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16
]

_TO_USE_TRAINING_FUNCTIONS_BASELINES = [
    # _run_baselines_hyper_parameter_tuning,
    _run_collaborative_filtering_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_BASELINES_FOLDED = [
    _run_baselines_folded_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_HEURISTICS = [
    _run_impressions_heuristics_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING = [
    _run_impressions_re_ranking_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING_ABLATION = [
    _run_ablation_impressions_re_ranking_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING_SIGNAL_ANALYSIS = [
    _run_signal_analysis_ablation_impressions_re_ranking_hyper_parameter_tuning,
]

_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_PROFILES = [
    _run_impressions_user_profiles_hyper_parameter_tuning,
]


if __name__ == "__main__":
    input_flags = ConsoleArguments().parse_args()

    configure_logger()
    logger = logging.getLogger(__name__)

    print(f"Running script: {__file__} with arguments: {input_flags.as_dict()}")

    logger.info(
        "Running script: %(script_name)s with arguments: %(args)s",
        {"script_name": __file__, "args": input_flags.as_dict()},
    )

    dask_interface = configure_dask_cluster()

    experiments_interface_baselines = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_BASELINE,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_BASELINES,
    )

    experiments_interface_baselines_folded = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_BASELINE_FOLDED,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_BASELINES_FOLDED,
    )

    experiments_impressions_heuristics_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_HEURISTICS,
    )

    experiments_impressions_re_ranking_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING,
    )

    experiments_ablation_impressions_re_ranking_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_ABLATION_IMPRESSIONS_RE_RANKING,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING_ABLATION,
    )

    experiments_signal_analysis_ablation_impressions_re_ranking_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_ABLATION_IMPRESSIONS_RE_RANKING,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING_SIGNAL_ANALYSIS,
    )

    experiments_signal_analysis_impressions_re_ranking_interface = ExperimentCasesSignalAnalysisInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders_baselines=_TO_USE_RECOMMENDERS_BASELINE,
        to_use_recommenders_impressions=_TO_USE_RECOMMENDERS_IMPRESSIONS_SIGNAL_ANALYSIS,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_RE_RANKING_SIGNAL_ANALYSIS,
    )

    experiments_impressions_user_profiles_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES,
        to_use_training_functions=_TO_USE_TRAINING_FUNCTIONS_IMPRESSION_AWARE_PROFILES,
    )

    create_necessary_folders(
        benchmarks=experiments_interface_baselines.benchmarks,
        evaluation_strategies=experiments_interface_baselines.evaluation_strategies,
    )

    if input_flags.create_datasets:
        ensure_datasets_exist(
            to_use_benchmarks=_TO_USE_BENCHMARKS,
        )

    if input_flags.include_baselines:
        run_baselines_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_folded:
        run_baselines_folded(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface_baselines_folded,
        )

    if input_flags.include_impressions_heuristics:
        run_impressions_heuristics_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_impressions_heuristics_interface,
        )

    if input_flags.include_impressions_reranking:
        run_impressions_re_ranking_experiments(
            dask_interface=dask_interface,
            re_ranking_experiment_cases_interface=experiments_impressions_re_ranking_interface,
            baseline_experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_impressions_profile:
        run_impressions_user_profiles_experiments(
            dask_interface=dask_interface,
            user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
            baseline_experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_signal_analysis:
        run_signal_analysis_impressions_re_ranking_experiments(
            dask_interface=dask_interface,
            signal_analysis_re_ranking_experiment_cases_interface=experiments_signal_analysis_impressions_re_ranking_interface,
        )

    if input_flags.include_ablation_impressions_reranking:
        run_ablation_impressions_re_ranking_experiments(
            dask_interface=dask_interface,
            ablation_re_ranking_experiment_cases_interface=experiments_ablation_impressions_re_ranking_interface,
            baseline_experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_signal_analysis_reranking:
        run_signal_analysis_ablation_impressions_re_ranking_experiments(
            dask_interface=dask_interface,
            ablation_re_ranking_experiment_cases_interface=experiments_signal_analysis_ablation_impressions_re_ranking_interface,
            baseline_experiment_cases_interface=experiments_interface_baselines,
        )

    dask_interface.wait_for_jobs()

    # if input_flags.compute_confidence_intervals:
    #     compute_confidence_intervals(
    #         dask_interface=dask_interface,
    #         experiment_cases_interface_baselines=experiments_interface_baselines,
    #         experiment_cases_interface_impressions_heuristics=experiments_impressions_heuristics_interface,
    #         experiment_cases_interface_impressions_re_ranking=experiments_impressions_re_ranking_interface,
    #         experiment_cases_interface_impressions_user_profiles=experiments_impressions_user_profiles_interface,
    #     )
    #
    # if input_flags.compute_statistical_tests:
    #     compute_statistical_tests(
    #         dask_interface=dask_interface,
    #         experiment_cases_interface_baselines=experiments_interface_baselines,
    #     )
    #
    # dask_interface.wait_for_jobs()
    if input_flags.analyze_hyper_parameters:
        baseline_recommenders = [
            "ItemKNNCFRecommender_asymmetric",
            "ItemKNNCFRecommender_cosine",
            "ItemKNNCFRecommender_dice",
            "ItemKNNCFRecommender_jaccard",
            "ItemKNNCFRecommender_tversky",
            "UserKNNCFRecommender_asymmetric",
            "UserKNNCFRecommender_cosine",
            "UserKNNCFRecommender_dice",
            "UserKNNCFRecommender_jaccard",
            "UserKNNCFRecommender_tversky",
            "P3alphaRecommender",
            "RP3betaRecommender",
            "PureSVDRecommender",
            "NMFRecommender",
            "MatrixFactorization_FunkSVD_Cython_Recommender",
            "MatrixFactorization_BPR_Cython_Recommender",
            "SLIMElasticNetRecommender",
            "SLIM_BPR_Recommender",
            "LightFMCFRecommender",
            "EASE_R_Recommender",
        ]
        impression_aware_recommenders = [
            "CyclingRecommender",
            "HardFrequencyCappingRecommender",
            "ImpressionsDiscountingRecommender",
            "ItemWeightedUserProfileRecommender",
            "UserWeightedUserProfileRecommender",
        ]

        metrics_to_optimize = ["COVERAGE_ITEM", "NDCG"]
        cutoff_to_optimize = 10

        dir_analysis_hyper_parameters = os.path.join(
            DIR_ANALYSIS_HYPER_PARAMETERS,
            "script_2022_evaluation_study_impression_aware_recommenders",
            "",
        )

        dir_parquet_results = DIR_PARQUET_RESULTS

        distribution_hyper_parameters_plug_in_impression_aware_recommenders(
            benchmarks=_TO_USE_BENCHMARKS_RESULTS,
            hyper_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS,
            dir_parquet_results=dir_parquet_results,
            dir_analysis_hyper_parameters=dir_analysis_hyper_parameters,
        )
        plot_parallel_hyper_parameters_plug_in_impression_aware_recommenders(
            benchmarks=_TO_USE_BENCHMARKS_RESULTS,
            hyper_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS,
            baseline_recommenders=baseline_recommenders,
            impression_aware_recommenders=impression_aware_recommenders,
            metrics_to_optimize=metrics_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
            dir_analysis_hyper_parameters=dir_analysis_hyper_parameters,
        )

    if input_flags.print_evaluation_results:
        process_evaluation_results(
            baseline_experiment_cases_interface=experiments_interface_baselines,
            impressions_heuristics_experiment_cases_interface=experiments_impressions_heuristics_interface,
            ablation_re_ranking_experiment_cases_interface=experiments_ablation_impressions_re_ranking_interface,
            re_ranking_experiment_cases_interface=experiments_impressions_re_ranking_interface,
            user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
        )

        export_evaluation_results(
            benchmarks=_TO_USE_BENCHMARKS_RESULTS,
            hyper_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS_RESULTS,
        )

    logger.info(f"Finished running script: {__file__}")
