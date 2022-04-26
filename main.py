#!/usr/bin/env python3
from __future__ import annotations

from recsys_framework_extensions.dask import configure_dask_cluster
from recsys_framework_extensions.logging import get_logger
from tap import Tap

from experiments.baselines import run_baselines_experiments, run_baselines_folded, print_baselines_results
from experiments.commons import (
    create_necessary_folders,
    ExperimentCasesInterface,
    Benchmarks,
    HyperParameterTuningParameters,
    plot_popularity_of_datasets,
    ensure_datasets_exist, RecommenderBaseline, RecommenderImpressions, EHyperParameterTuningParameters,
)
from experiments.heuristics import run_impressions_heuristics_experiments, print_impressions_heuristics_results
from experiments.print_results import print_results
from experiments.re_ranking import run_impressions_re_ranking_experiments, print_impressions_re_ranking_results
from experiments.user_profiles import run_impressions_user_profiles_experiments, print_impressions_user_profiles_results


class ConsoleArguments(Tap):
    create_datasets: bool = False
    """TODO: fernando-debugger."""

    include_baselines: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_folded: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_impressions_heuristics: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_impressions_reranking: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_impressions_profile: bool = False
    """Include baselines in the hyper-parameter tuning"""

    print_evaluation_results: bool = False
    """Print LaTeX tables containing the accuracy and beyond accuracy metrics of the hyper-parameter tuned 
    recommenders."""

    plot_popularity_of_datasets: bool = False
    """Creates plots depicting the popularity of each dataset split."""

    send_email: bool = False
    """Send a notification email via GMAIL when experiments finish."""


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

_TO_USE_RECOMMENDERS_BASELINE = [
    RecommenderBaseline.RANDOM,
    RecommenderBaseline.TOP_POPULAR,
    RecommenderBaseline.GLOBAL_EFFECTS,

    RecommenderBaseline.USER_KNN,
    RecommenderBaseline.ITEM_KNN,

    RecommenderBaseline.PURE_SVD,
    RecommenderBaseline.NMF,
    RecommenderBaseline.MF_BPR,
    RecommenderBaseline.IALS,
    RecommenderBaseline.FUNK_SVD,
    RecommenderBaseline.ASYMMETRIC_SVD,

    RecommenderBaseline.P3_ALPHA,
    RecommenderBaseline.RP3_BETA,

    RecommenderBaseline.SLIM_ELASTIC_NET,
    RecommenderBaseline.SLIM_BPR,

    RecommenderBaseline.LIGHT_FM,
    RecommenderBaseline.MULT_VAE,
    RecommenderBaseline.EASE_R,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS = [
    RecommenderImpressions.LAST_IMPRESSIONS,
    RecommenderImpressions.FREQUENCY_RECENCY,
    RecommenderImpressions.RECENCY,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING = [
    RecommenderImpressions.CYCLING,
    RecommenderImpressions.IMPRESSIONS_DISCOUNTING,
]

_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES = [
    RecommenderImpressions.USER_WEIGHTED_USER_PROFILE,
    RecommenderImpressions.ITEM_WEIGHTED_USER_PROFILE,
]

_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS = [
    EHyperParameterTuningParameters.LEAVE_LAST_OUT_BAYESIAN_50_16,
]


if __name__ == '__main__':
    input_flags = ConsoleArguments().parse_args()

    logger = get_logger(__name__)

    if input_flags.send_email:
        from recsys_framework_extensions.data.io import ExtendedJSONEncoderDecoder
        from recsys_framework_extensions.email.gmail import GmailEmailNotifier
        import json

        GmailEmailNotifier.send_email(
            subject="[Impressions Datasets] Execution Started",
            body=f"""An execution with the following properties just started:
            \n\t* Input Flags: {json.dumps(input_flags.as_dict(), default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Benchmarks: {json.dumps(_TO_USE_BENCHMARKS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Hyper-parameters: {json.dumps(_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Baselines: {json.dumps(_TO_USE_RECOMMENDERS_BASELINE, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions Heuristics: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions Re Ranking: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions User Profiles: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES, default=ExtendedJSONEncoderDecoder.to_json)} 
            """,
            sender="mistermaurera@gmail.com",
            receivers=[
                "fperezmaurera@gmail.com",
                "fernandobenjamin.perez@polimi.it",
            ],
        )

    dask_interface = configure_dask_cluster()

    common_hyper_parameter_tuning_parameters = HyperParameterTuningParameters()

    # Training statistics.
    # CW - UserKNN - 3 GB Training - 250 sec/it
    # CW - ItemKNN - 3 GB Training - 320 sec/it
    # CW - PureSVD - 2 GB Training - 170 sec/it
    # CW - EASE-R - 28 GB Training - 400 sec/it
    # MINDSmall - EASE_R - 16GB Training - 80sec/it
    # MINDLarge - EASE_R - 29.3GB Training - 450sec/it
    # FINNNoSlates - EASE R - 12.4TB Training - No Training.
    experiments_interface_baselines = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_BASELINE,
    )

    experiments_impressions_heuristics_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS,
    )

    experiments_impressions_re_ranking_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING,
    )

    experiments_impressions_user_profiles_interface = ExperimentCasesInterface(
        to_use_benchmarks=_TO_USE_BENCHMARKS,
        to_use_hyper_parameter_tuning_parameters=_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS,
        to_use_recommenders=_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES,
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
        run_baselines_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface_baselines,
        )

    if input_flags.include_impressions_heuristics:
        run_impressions_heuristics_experiments(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_impressions_heuristics_interface,
        )

    dask_interface.wait_for_jobs()

    if input_flags.include_folded:
        run_baselines_folded(
            dask_interface=dask_interface,
            experiment_cases_interface=experiments_interface_baselines,
        )

    dask_interface.wait_for_jobs()

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

    dask_interface.wait_for_jobs()

    if input_flags.plot_popularity_of_datasets:
        plot_popularity_of_datasets(experiments_interface=experiments_interface_baselines)

    if input_flags.print_evaluation_results:
        print_results(
            baseline_experiment_cases_interface=experiments_interface_baselines,
            impressions_heuristics_experiment_cases_interface=experiments_impressions_heuristics_interface,
            re_ranking_experiment_cases_interface=experiments_impressions_re_ranking_interface,
            user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
        )
        # print_baselines_results(
        #     baseline_experiment_cases_interface=experiments_interface_baselines,
        # )
        # print_impressions_heuristics_results(
        #     impressions_heuristics_experiment_cases_interface=experiments_impressions_heuristics_interface,
        # )
        # print_impressions_re_ranking_results(
        #     baseline_experiment_cases_interface=experiments_interface_baselines,
        #     re_ranking_experiment_cases_interface=experiments_impressions_re_ranking_interface
        # )
        # print_impressions_user_profiles_results(
        #     baseline_experiment_cases_interface=experiments_interface_baselines,
        #     user_profiles_experiment_cases_interface=experiments_impressions_user_profiles_interface,
        # )

    if input_flags.send_email:
        from recsys_framework_extensions.data.io import ExtendedJSONEncoderDecoder
        from recsys_framework_extensions.email.gmail import GmailEmailNotifier
        import json

        GmailEmailNotifier.send_email(
            subject="[Impressions Datasets] Execution finished",
            body=f"""An execution with the following properties just finished:
            \n\t* Input Flags: {json.dumps(input_flags.as_dict(), default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Benchmarks: {json.dumps(_TO_USE_BENCHMARKS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Hyper-parameters: {json.dumps(_TO_USE_HYPER_PARAMETER_TUNING_PARAMETERS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Baselines: {json.dumps(_TO_USE_RECOMMENDERS_BASELINE, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions Heuristics: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_HEURISTICS, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions Re Ranking: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_RE_RANKING, default=ExtendedJSONEncoderDecoder.to_json)} 
            \n\t* Impressions User Profiles: {json.dumps(_TO_USE_RECOMMENDERS_IMPRESSIONS_USER_PROFILES, default=ExtendedJSONEncoderDecoder.to_json)} 
            """,
            sender="mistermaurera@gmail.com",
            receivers=[
                "fperezmaurera@gmail.com",
                "fernandobenjamin.perez@polimi.it",
            ],
        )

    dask_interface.close()
